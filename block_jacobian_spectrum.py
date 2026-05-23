"""Sandbox B: per-layer FULL block Jacobian spectrum.

Companion to local_jacobian_spectrum.py (Sandbox A).  Whereas Sandbox A measures
the spectrum of the *canonical* residual oracle

    R_l(X) = A_l(LN(X)) + M_l(LN(X + A_l(LN(X)))),

Sandbox B measures the spectrum of the actual block transition operator

    F_l(X) = x_{l+1},

i.e. the layer's own forward, with the auxiliary state (velocity v for
Yurii/TMM, moments (m, s) for Adam/AdamW) held fixed at its captured value.
For Vanilla, F_l(x) = x + R_l(x), so its block-Jacobian = I + (Sandbox A's J);
for Yurii/TMM/Adam(W) it includes LN_v / gamma / nu / sqrt(s) wrappers and
therefore the contraction operator that the doc's rho_mom analysis is
*actually* about.

For each layer l of each OWT-pretrained variant, on the OWT validation set,
we capture (x_l) and any auxiliary stream into the layer, then estimate

    sigma_max(d F_l / dx)        power iter on J^T J (20 iters)
    top-64 SVs                   randomized SVD (p = 74 JVPs + 74 VJPs)
    ||J||_F^2                    Hutchinson (8 Rademacher probes)
    kappa_eff = sigma_max / sigma_min^eff
    stable rank = ||J||_F^2 / sigma_max^2

All estimators are matrix-free; the Jacobian is never materialized.

The contraction predicted by the doc's theory is rho_mom < rho_vanilla
where rho is roughly ||I - (something)|| of the block-transition spectrum.
The relevant cross-variant rank is on sigma_max(d F_l / dx) and on the
distance from 1, not on the raw oracle norms.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────────────────
# Variant registry.  aux_kind tags how the layer expects to be called:
#   "x"   — layer(x) -> x_next                    (Vanilla)
#   "xv"  — layer(x, v) -> (x_next, v_next)       (Yurii, TMM)
#   "xms" — layer(x, m, s) -> (x_next, m_next, s) (Adam, AdamW)
# ────────────────────────────────────────────────────────────────────────────
VARIANTS = {
    "vanilla":   ("vanilla_model", "VanillaTransformer", "checkpoints_vanilla_owt",   "x"),
    "adam":      ("adam_model",    "AdamFormer",         "checkpoints_adam_owt",      "xms"),
    "adamw":     ("adamw_model",   "AdamWFormer",        "checkpoints_adamw_owt",     "xms"),
    "yurii":     ("model",         "YuriiFormer",        "checkpoints_yurii_owt",     "xv"),
    "tmm":       ("tmm_model",     "TMMFormer",          "checkpoints_tmm_owt",       "xv"),
    "yurii-sam": ("model",         "YuriiFormer",        "checkpoints_yurii_sam_owt", "xv"),
    "yurii-wsd": ("model",         "YuriiFormer",        "checkpoints_yurii_wsd_owt", "xv"),
    "tmm-sam":   ("tmm_model",     "TMMFormer",          "checkpoints_tmm_sam_owt",   "xv"),
    "tmm-wsd":   ("tmm_model",     "TMMFormer",          "checkpoints_tmm_wsd_owt",   "xv"),
}


def force_math_sdpa():
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    )


def load_model(variant: str, device: str):
    mod_name, cls_name, ckpt_subdir, aux_kind = VARIANTS[variant]
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
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  step={ckpt.get('step')}, val_loss={ckpt.get('val_loss')}, "
          f"aux_kind={aux_kind}", flush=True)
    return model, aux_kind


def make_block_fn(layer, aux_kind: str, aux_state):
    """Return F(x) = x_{l+1} given the layer's actual forward signature."""
    if aux_kind == "x":
        def F(x): return layer(x)
        return F
    if aux_kind == "xv":
        v = aux_state[0]
        def F(x): return layer(x, v)[0]
        return F
    if aux_kind == "xms":
        m, s = aux_state
        def F(x): return layer(x, m, s)[0]
        return F
    raise ValueError(f"unknown aux_kind {aux_kind}")


@torch.no_grad()
def capture_inputs(model, val_tokens: np.ndarray, aux_kind: str,
                   n_batches: int, batch_size: int, block_size: int,
                   device: str, seed: int = 0):
    """Pre-forward hook on each layer.  For every batch we record the *full*
    tuple of layer inputs (x, [v] or [m, s]) so the block function can hold
    auxiliary streams fixed."""
    n_layers = len(model.layers)
    captured = [[None] * n_batches for _ in range(n_layers)]

    hooks = []
    current = {"batch": 0}

    def make_pre_hook(layer_idx):
        def hook(module, inputs):
            captured[layer_idx][current["batch"]] = tuple(
                t.detach().clone() for t in inputs
            )
        return hook

    for li, layer in enumerate(model.layers):
        h = layer.register_forward_pre_hook(make_pre_hook(li))
        hooks.append(h)

    rng = np.random.default_rng(seed)
    val_tokens_t = torch.from_numpy(val_tokens.astype(np.int64))
    try:
        for bi in range(n_batches):
            current["batch"] = bi
            starts = rng.integers(0, len(val_tokens_t) - block_size - 2, size=batch_size)
            x = torch.stack([val_tokens_t[s:s + block_size] for s in starts]).to(device)
            with force_math_sdpa():
                _ = model(x)
    finally:
        for h in hooks:
            h.remove()

    # Sanity-check the captured tuple length matches aux_kind.
    expect = {"x": 1, "xv": 2, "xms": 3}[aux_kind]
    got = len(captured[0][0])
    if got != expect:
        raise RuntimeError(f"aux_kind={aux_kind} expects {expect} inputs, "
                           f"got {got}")
    return captured


def jvp_fn(F, x, v):
    _, Jv = torch.autograd.functional.jvp(F, x, v=v, create_graph=False, strict=False)
    return Jv


def vjp_fn(F, x, u):
    _, JTu = torch.autograd.functional.vjp(F, x, v=u, create_graph=False, strict=False)
    return JTu


def power_iter_sigma_max(F, x, n_iter: int = 20, tol: float = 1e-3):
    v = torch.randn_like(x); v = v / (v.norm() + 1e-12)
    sigma_old = 0.0
    for it in range(n_iter):
        with force_math_sdpa():
            Jv = jvp_fn(F, x, v)
            JTJv = vjp_fn(F, x, Jv)
        sigma_sq = (v * JTJv).sum().item()
        sigma = math.sqrt(max(sigma_sq, 0.0))
        v = JTJv / (JTJv.norm() + 1e-12)
        if abs(sigma - sigma_old) / (abs(sigma) + 1e-8) < tol and it >= 5:
            break
        sigma_old = sigma
    return sigma


def randomized_top_k_svd(F, x, k: int = 64, oversample: int = 10):
    p = k + oversample
    d_flat = x.numel()
    if p > d_flat:
        p = d_flat; k = max(1, p - oversample)

    Y_cols = []
    for _ in range(p):
        omega = torch.randn_like(x)
        with force_math_sdpa():
            Jw = jvp_fn(F, x, omega)
        Y_cols.append(Jw.reshape(-1))
    Y = torch.stack(Y_cols, dim=1)

    Q, _ = torch.linalg.qr(Y, mode="reduced")

    B_rows = []
    for j in range(p):
        u = Q[:, j].reshape(x.shape).contiguous()
        with force_math_sdpa():
            JTu = vjp_fn(F, x, u)
        B_rows.append(JTu.reshape(-1))
    B = torch.stack(B_rows, dim=0)

    sv = torch.linalg.svdvals(B)
    sv = sv.detach().cpu().numpy()[::-1].copy()
    return sv[:k]


def hutchinson_frobenius_sq(F, x, n_probes: int = 8):
    vals = []
    for _ in range(n_probes):
        z = (torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1)
        with force_math_sdpa():
            Jz = jvp_fn(F, x, z)
        vals.append((Jz * Jz).sum().item())
    return float(np.mean(vals)), float(np.std(vals))


def analyze_one_layer(F, x, k: int, n_power_iter: int, n_probes: int):
    sigma_max_pi = power_iter_sigma_max(F, x, n_iter=n_power_iter)
    sv = randomized_top_k_svd(F, x, k=k)
    sigma_min_eff = float(sv[-1])
    sigma_max = max(sigma_max_pi, float(sv[0]))
    frob_sq, frob_sq_std = hutchinson_frobenius_sq(F, x, n_probes=n_probes)
    return {
        "sigma_max": sigma_max,
        "sigma_min_eff": sigma_min_eff,
        "kappa_eff": sigma_max / max(sigma_min_eff, 1e-12),
        "top_k_sv": sv.tolist(),
        "frob_sq_mean": frob_sq,
        "frob_sq_std": frob_sq_std,
        "stable_rank": frob_sq / max(sigma_max ** 2, 1e-12),
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
    ap.add_argument("--out_dir", default="block_spectrum_results")
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(True)
    model, aux_kind = load_model(args.variant, device)

    from data_owt import load_owt_tokens
    val_tokens = load_owt_tokens("val")
    print(f"  val tokens: {len(val_tokens):,}", flush=True)

    t_capture = time.time()
    captured = capture_inputs(
        model, val_tokens, aux_kind,
        n_batches=args.n_batches, batch_size=args.batch_size,
        block_size=args.block_size, device=device, seed=0,
    )
    x_shape = captured[0][0][0].shape
    print(f"  capture done in {time.time() - t_capture:.0f}s, "
          f"n_layers={len(captured)}, n_batches={args.n_batches}, "
          f"x shape {tuple(x_shape)}", flush=True)

    layer_ids = list(range(len(model.layers)))
    per_layer_results = []
    t0 = time.time()
    for li in layer_ids:
        layer = model.layers[li]
        per_batch = []
        for bi, inputs in enumerate(captured[li]):
            x = inputs[0]
            aux_state = inputs[1:]
            F = make_block_fn(layer, aux_kind, aux_state)
            tb = time.time()
            metrics = analyze_one_layer(
                F, x, k=args.top_k,
                n_power_iter=args.power_iters,
                n_probes=args.hutch_probes,
            )
            metrics["batch"] = bi
            per_batch.append(metrics)
            print(f"  L{li:02d} b{bi}: sigma_max={metrics['sigma_max']:.3f}, "
                  f"sigma_min={metrics['sigma_min_eff']:.3f}, "
                  f"kappa={metrics['kappa_eff']:.2f}, "
                  f"||J||_F^2={metrics['frob_sq_mean']:.2e}, "
                  f"srank={metrics['stable_rank']:.1f}  [{time.time() - tb:.0f}s]",
                  flush=True)
        per_layer_results.append({"layer": li, "per_batch": per_batch})
        elapsed = time.time() - t0
        n_done = layer_ids.index(li) + 1
        eta = elapsed / n_done * (len(layer_ids) - n_done)
        print(f"  layer {li} mean kappa = "
              f"{np.mean([b['kappa_eff'] for b in per_batch]):.2f}  "
              f"[total {elapsed:.0f}s, ETA {eta:.0f}s]", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.json"
    payload = {
        "variant": args.variant,
        "aux_kind": aux_kind,
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
