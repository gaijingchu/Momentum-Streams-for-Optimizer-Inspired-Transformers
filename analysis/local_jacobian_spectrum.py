"""Sandbox A: per-layer local Jacobian spectrum.

Implements the experimental plan from docs/momentum_transformer_theory.md §7.
For each layer l of an OWT-pretrained variant, we treat the residual stream
as input to a "canonical residual oracle"

    R_l(X) = A_l(LN(X)) + M_l(LN(X + A_l(LN(X))))

where A_l and M_l are the layer's actual attention and MLP modules and X is
the state the model's forward pass feeds into the attention sublayer:
  - Vanilla / Adam / AdamW: X = block input.
  - Yurii / TMM:            X = x + mu_attn * v  (the attention lookahead).

We then estimate the local Jacobian J_l = dR_l/dX at the captured X, matrix-free,
via JVP/VJP through power iteration and randomized range-finding:

  sigma_max          power iter on J^T J (20 iters)
  top-k SVs (k=64)   randomized SVD: Y = J Omega, QR, B = Q^T J, SVD(B)
  sigma_min^eff      smallest of the top-k SVs (so kappa_eff = sigma_1 / sigma_k)
  ||J||_F^2          Hutchinson with Rademacher probes (8 probes)
  stable rank        ||J||_F^2 / sigma_max^2

Outputs spectrum_results/<variant>.json with per-layer + per-batch arrays.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Variant registry: (model_module, model_class, ckpt_subdir, lookahead_kind)
#   lookahead_kind: "x"  — attention input is the block input itself
#                   "xv" — attention input is x + mu_attn * v  (Yurii/TMM)
# All OWT-pretrained checkpoints live at ${CACHE}/<ckpt_subdir>/best.pt.
# ────────────────────────────────────────────────────────────────────────────
VARIANTS = {
    "vanilla":   ("vanilla_model", "VanillaTransformer", "checkpoints_vanilla_owt",   "x"),
    "adam":      ("adam_model",    "AdamFormer",         "checkpoints_adam_owt",      "x"),
    "adamw":     ("adamw_model",   "AdamWFormer",        "checkpoints_adamw_owt",     "x"),
    "yurii":     ("model",         "YuriiFormer",        "checkpoints_yurii_owt",     "xv"),
    "tmm":       ("tmm_model",     "TMMFormer",          "checkpoints_tmm_owt",       "xv"),
    "yurii-sam": ("model",         "YuriiFormer",        "checkpoints_yurii_sam_owt", "xv"),
    "yurii-wsd": ("model",         "YuriiFormer",        "checkpoints_yurii_wsd_owt", "xv"),
    "tmm-sam":   ("tmm_model",     "TMMFormer",          "checkpoints_tmm_sam_owt",   "xv"),
    "tmm-wsd":   ("tmm_model",     "TMMFormer",          "checkpoints_tmm_wsd_owt",   "xv"),
}


def force_math_sdpa():
    """Disable flash/memefficient SDPA: they do not support double-backward
    paths used by JVP, and JVP through them returns wrong results silently."""
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    )


def load_model(variant: str, device: str):
    mod_name, cls_name, ckpt_subdir, lookahead_kind = VARIANTS[variant]
    cache = os.environ.get("CACHE", "./checkpoints_cache")
    ckpt_path = os.path.join(cache, ckpt_subdir, "best.pt")
    print(f"[{variant}] loading {ckpt_path}", flush=True)

    mod = __import__(mod_name)
    cls = getattr(mod, cls_name)
    model = cls().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    # Disable grad for everything; we only need autograd through R_l(X).
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  step={ckpt.get('step')}, val_loss={ckpt.get('val_loss')}", flush=True)
    return model, lookahead_kind


def make_R_fn(layer):
    """Canonical residual oracle R_l(X) using the layer's actual modules."""
    attn = layer.attn
    mlp = layer.mlp
    ln_attn = layer.ln_attn
    ln_mlp = layer.ln_mlp

    def R(X):
        a = attn(ln_attn(X))
        m = mlp(ln_mlp(X + a))
        return a + m

    return R


@torch.no_grad()
def capture_inputs(model, val_tokens: np.ndarray, lookahead_kind: str,
                   n_batches: int, batch_size: int, block_size: int,
                   device: str, seed: int = 0):
    """Run the model on n_batches validation batches; per layer, record the
    state fed into ln_attn (Vanilla: x; Yurii/TMM: x + mu_a*v; Adam(W): x).

    Returns: list of length n_layers, each a list of n_batches captured tensors
             of shape (B, T, d).
    """
    n_layers = len(model.layers)
    captured = [[None] * n_batches for _ in range(n_layers)]

    hooks = []
    current_batch = {"i": 0}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # ln_attn input is exactly the desired X across all variants.
            captured[layer_idx][current_batch["i"]] = inputs[0].detach().to(device)
        return hook

    for li, layer in enumerate(model.layers):
        h = layer.ln_attn.register_forward_hook(make_hook(li))
        hooks.append(h)

    rng = np.random.default_rng(seed)
    val_tokens_t = torch.from_numpy(val_tokens.astype(np.int64))

    try:
        for bi in range(n_batches):
            current_batch["i"] = bi
            starts = rng.integers(0, len(val_tokens_t) - block_size - 2, size=batch_size)
            x = torch.stack([val_tokens_t[s:s + block_size] for s in starts]).to(device)
            with force_math_sdpa():
                _ = model(x)
    finally:
        for h in hooks:
            h.remove()

    return captured


def jvp_fn(R, X, v):
    """Returns J v where J = dR/dX at X."""
    _, Jv = torch.autograd.functional.jvp(R, X, v=v, create_graph=False, strict=False)
    return Jv


def vjp_fn(R, X, u):
    """Returns J^T u."""
    _, vjp_result = torch.autograd.functional.vjp(R, X, v=u, create_graph=False, strict=False)
    # vjp returns a tuple matching the inputs; we passed a single tensor.
    return vjp_result


def power_iter_sigma_max(R, X, n_iter: int = 20, tol: float = 1e-3):
    v = torch.randn_like(X)
    v = v / (v.norm() + 1e-12)
    sigma_old = 0.0
    for it in range(n_iter):
        with force_math_sdpa():
            Jv = jvp_fn(R, X, v)
            JTJv = vjp_fn(R, X, Jv)
        v_norm = JTJv.norm()
        sigma_sq = (v * JTJv).sum().item()
        sigma = math.sqrt(max(sigma_sq, 0.0))
        v = JTJv / (v_norm + 1e-12)
        if abs(sigma - sigma_old) / (abs(sigma) + 1e-8) < tol and it >= 5:
            break
        sigma_old = sigma
    return sigma


def randomized_top_k_svd(R, X, k: int = 64, oversample: int = 10):
    """Randomized SVD: top-(k+oversample) singular values of J = dR/dX.

    Y = J Omega   (forward sketch via JVP)
    Q = qr(Y).Q   (range basis of J)
    B = Q^T J     (small matrix via VJP of Q's columns)
    sv = svd(B).S

    All in matrix-free fashion.  Returns numpy array of top-k singular values
    (descending).
    """
    p = k + oversample
    d_flat = X.numel()
    if p > d_flat:
        p = d_flat
        k = max(1, p - oversample)

    # ── Y = J Omega ──────────────────────────────────────────────────────
    Y_cols = []
    for _ in range(p):
        omega = torch.randn_like(X)
        with force_math_sdpa():
            Jw = jvp_fn(R, X, omega)
        Y_cols.append(Jw.reshape(-1))
    Y = torch.stack(Y_cols, dim=1)            # (d_flat, p)

    # ── Q = qr(Y) ────────────────────────────────────────────────────────
    Q, _ = torch.linalg.qr(Y, mode="reduced") # (d_flat, p)

    # ── B = Q^T J via p VJPs ─────────────────────────────────────────────
    B_rows = []
    for j in range(p):
        u = Q[:, j].reshape(X.shape).contiguous()
        with force_math_sdpa():
            JTu = vjp_fn(R, X, u)             # (..., shape of X)
        B_rows.append(JTu.reshape(-1))
    B = torch.stack(B_rows, dim=0)            # (p, d_flat); rows are J^T q_j

    # SVD of small matrix B^T (we want SVs of J ~= Q B' where B' = Q^T J)
    # Since B = Q^T J, the SVs of B equal the top-p SVs of J (within the range
    # captured by Q).  Take top-k.
    sv = torch.linalg.svdvals(B)              # (p,)
    sv = sv.detach().cpu().numpy()[::-1].copy()  # descending
    return sv[:k]


def hutchinson_frobenius_sq(R, X, n_probes: int = 8):
    """||J||_F^2 ≈ E_z ||J z||^2 with z Rademacher."""
    vals = []
    for _ in range(n_probes):
        z = (torch.randint(0, 2, X.shape, device=X.device, dtype=X.dtype) * 2 - 1)
        with force_math_sdpa():
            Jz = jvp_fn(R, X, z)
        vals.append((Jz * Jz).sum().item())
    return float(np.mean(vals)), float(np.std(vals))


def analyze_one_layer(R, X, k: int, n_power_iter: int, n_probes: int):
    """Returns dict of metrics for one layer × one batch."""
    sigma_max = power_iter_sigma_max(R, X, n_iter=n_power_iter)
    sv = randomized_top_k_svd(R, X, k=k)
    sigma_min_eff = float(sv[-1])
    sigma_top = float(sv[0])
    # Use the max of (power-iter, randomized-top-1) as canonical sigma_max
    # (power iter is usually more accurate for the top mode).
    sigma_max_canonical = max(sigma_max, sigma_top)
    frob_sq, frob_sq_std = hutchinson_frobenius_sq(R, X, n_probes=n_probes)
    kappa_eff = sigma_max_canonical / max(sigma_min_eff, 1e-12)
    stable_rank = frob_sq / max(sigma_max_canonical ** 2, 1e-12)
    return {
        "sigma_max": sigma_max_canonical,
        "sigma_min_eff": sigma_min_eff,
        "kappa_eff": kappa_eff,
        "top_k_sv": sv.tolist(),
        "frob_sq_mean": frob_sq,
        "frob_sq_std": frob_sq_std,
        "stable_rank": stable_rank,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--power_iters", type=int, default=20)
    ap.add_argument("--hutch_probes", type=int, default=8)
    ap.add_argument("--out_dir", default="spectrum_results")
    ap.add_argument("--layers", default="all",
                    help="'all' or comma-separated indices, e.g. '0,5,11'")
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(True)  # we need autograd for JVP
    model, lookahead_kind = load_model(args.variant, device)

    # All OWT variants: use OWT validation tokens.
    from data.openwebtext import load_owt_tokens
    val_tokens = load_owt_tokens("val")
    print(f"  val tokens: {len(val_tokens):,}", flush=True)

    t_capture = time.time()
    captured = capture_inputs(
        model, val_tokens, lookahead_kind,
        n_batches=args.n_batches, batch_size=args.batch_size,
        block_size=args.block_size, device=device, seed=0,
    )
    print(f"  capture done in {time.time() - t_capture:.0f}s, n_layers={len(captured)}, "
          f"n_batches={args.n_batches}, X shape {captured[0][0].shape}", flush=True)

    if args.layers == "all":
        layer_ids = list(range(len(model.layers)))
    else:
        layer_ids = [int(x) for x in args.layers.split(",")]

    per_layer_results = []
    t0 = time.time()
    for li in layer_ids:
        layer = model.layers[li]
        R = make_R_fn(layer)
        per_batch = []
        for bi, X in enumerate(captured[li]):
            tb = time.time()
            metrics = analyze_one_layer(
                R, X, k=args.top_k,
                n_power_iter=args.power_iters,
                n_probes=args.hutch_probes,
            )
            metrics["batch"] = bi
            per_batch.append(metrics)
            print(f"  L{li:02d} b{bi}: sigma_max={metrics['sigma_max']:.3f}, "
                  f"sigma_min={metrics['sigma_min_eff']:.3f}, "
                  f"kappa={metrics['kappa_eff']:.2f}, "
                  f"||J||_F^2={metrics['frob_sq_mean']:.1f}, "
                  f"srank={metrics['stable_rank']:.1f}  [{time.time() - tb:.0f}s]",
                  flush=True)
        per_layer_results.append({"layer": li, "per_batch": per_batch})
        elapsed = time.time() - t0
        n_done = layer_ids.index(li) + 1
        n_total = len(layer_ids)
        eta = elapsed / n_done * (n_total - n_done)
        print(f"  layer {li} mean kappa = "
              f"{np.mean([b['kappa_eff'] for b in per_batch]):.2f}  "
              f"[total {elapsed:.0f}s, ETA {eta:.0f}s]", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.json"
    payload = {
        "variant": args.variant,
        "lookahead_kind": lookahead_kind,
        "config": {
            "n_batches": args.n_batches, "batch_size": args.batch_size,
            "block_size": args.block_size, "top_k": args.top_k,
            "power_iters": args.power_iters, "hutch_probes": args.hutch_probes,
        },
        "per_layer": per_layer_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
