"""Loss-landscape sharpness for each transformer variant.

Computes three sharpness proxies on the OWT validation set:
  1) Top Hessian eigenvalue lambda_max via power iteration on HVP.
  2) Hessian trace via Hutchinson's estimator (Rademacher probes).
  3) 1D filter-normalized loss perturbation curve along a random
     direction (Li et al. 2018), giving a cheap "flatness shape".

For a model with parameters theta and loss L(theta):
  - HVP: H v = grad(<grad L, v>) computed by double backward.
  - Power iter: v <- H v / ||H v|| converges to top eigvec.
  - Hutchinson: tr(H) ~ E[v^T H v], v Rademacher (+/-1).
"""
import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = {
    # OWT variants (data_owt.load_owt_tokens, $CACHE/checkpoints_<variant>_owt/best.pt)
    "vanilla":  ("vanilla_model",  "VanillaTransformer", "checkpoints_vanilla_owt"),
    "vanilla_adamw": ("vanilla_model", "VanillaTransformer", "checkpoints_vanilla_adamw_owt"),
    "yurii":    ("model",          "YuriiFormer",        "checkpoints_yurii_owt"),
    "tmm":      ("tmm_model",      "TMMFormer",          "checkpoints_tmm_owt"),
    "adam":     ("adam_model",     "AdamFormer",         "checkpoints_adam_owt"),
    "adamw":    ("adamw_model",    "AdamWFormer",        "checkpoints_adamw_owt"),
    "wsd":      ("model",          "YuriiFormer",        "checkpoints_yurii_wsd_owt"),
    "sam":      ("model",          "YuriiFormer",        "checkpoints_yurii_sam_owt"),
    "sawd":     ("model",          "YuriiFormer",        "checkpoints_yurii_sawd_owt"),
    # TinyStories variants (data.load_tokens, $CACHE/checkpoints_ts_archive/<variant>_ts/best.pt)
    "ts-vanilla":  ("vanilla_model",  "VanillaTransformer", "checkpoints_ts_archive/vanilla_ts"),
    "ts-yurii":    ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_ts"),
    "ts-tmm":      ("tmm_model",      "TMMFormer",          "checkpoints_ts_archive/tmm_ts"),
    "ts-adam":     ("adam_model",     "AdamFormer",         "checkpoints_ts_archive/adam_ts"),
    "ts-adamw":    ("adamw_model",    "AdamWFormer",        "checkpoints_ts_archive/adamw_ts"),
    "ts-wsd":      ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_wsd_ts"),
    "ts-sam":      ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_sam_ts"),
    "ts-sawd":     ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_sawd_ts"),
    "ts-muon":     ("muon_model",     "MuonFormer",         "checkpoints_ts_archive/muon_ts"),
    # TMM + schedule variants (TS)
    "ts-tmm-wsd":  ("tmm_model",      "TMMFormer",          "checkpoints_tmm_wsd_ts"),
    "ts-tmm-sam":  ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sam_ts"),
    "ts-tmm-sawd": ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sawd_ts"),
    # TMM + schedule variants (OWT)
    "tmm-wsd":     ("tmm_model",      "TMMFormer",          "checkpoints_tmm_wsd_owt"),
    "tmm-sam":     ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sam_owt"),
    "tmm-sawd":    ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sawd_owt"),
}


def get_loss(model, x, y):
    # Force math SDPA: efficient/flash backends do not support double backward.
    with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                        enable_mem_efficient=False,
                                        enable_math=True):
        logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


def hvp(loss, params, vec):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_g = torch.cat([g.reshape(-1) for g in grads])
    gv = (flat_g * vec).sum()
    Hv = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in Hv]).detach()


def power_iter_top_eig(model, batches, params, n_iter=15, tol=1e-3):
    n = sum(p.numel() for p in params)
    v = torch.randn(n, device="cuda")
    v /= v.norm()
    eig_old = 0.0
    for it in range(n_iter):
        Hv_total = torch.zeros_like(v)
        for x, y in batches:
            loss = get_loss(model, x, y)
            Hv_total += hvp(loss, params, v)
        Hv = Hv_total / len(batches)
        eig = (v * Hv).sum().item()
        v = Hv / (Hv.norm() + 1e-12)
        print(f"  power iter {it+1}: lambda_max ~ {eig:.4f}")
        if abs(eig - eig_old) / (abs(eig) + 1e-8) < tol:
            break
        eig_old = eig
    return eig


def hutchinson_trace(model, batches, params, n_probe=10):
    n = sum(p.numel() for p in params)
    traces = []
    for k in range(n_probe):
        v = torch.randint(0, 2, (n,), device="cuda", dtype=torch.float32) * 2 - 1
        Hv_total = torch.zeros_like(v)
        for x, y in batches:
            loss = get_loss(model, x, y)
            Hv_total += hvp(loss, params, v)
        Hv = Hv_total / len(batches)
        tr = (v * Hv).sum().item()
        traces.append(tr)
        print(f"  hutchinson probe {k+1}: tr ~ {tr:.2f}  (running mean {np.mean(traces):.2f})")
    return float(np.mean(traces)), float(np.std(traces))


def loss_curve(model, batches, params, alphas):
    """1D filter-normalized perturbation: theta + alpha * d, d filter-normalized."""
    # Build filter-normalized direction (per-parameter, scaled to its norm).
    d = []
    for p in params:
        di = torch.randn_like(p)
        di = di * (p.norm() / (di.norm() + 1e-12))
        d.append(di)
    base = [p.detach().clone() for p in params]
    losses = []
    with torch.no_grad():
        for a in alphas:
            for p, b, di in zip(params, base, d):
                p.copy_(b + a * di)
            ls = []
            for x, y in batches:
                ls.append(get_loss(model, x, y).item())
            losses.append(float(np.mean(ls)))
            print(f"  alpha={a:+.3f}  loss={losses[-1]:.4f}")
        for p, b in zip(params, base):
            p.copy_(b)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--power_iters", type=int, default=15)
    ap.add_argument("--hutch_probes", type=int, default=10)
    ap.add_argument("--curve_pts", type=int, default=11)
    ap.add_argument("--curve_max", type=float, default=0.5)
    args = ap.parse_args()

    cache = os.environ.get("CACHE", "./checkpoints_cache")
    mod_name, cls_name, ckpt_subdir = VARIANTS[args.variant]
    ckpt_path = os.path.join(cache, ckpt_subdir, "best.pt")
    print(f"[{args.variant}] loading {ckpt_path}")

    module_obj = __import__(mod_name)
    Model = getattr(module_obj, cls_name)
    model = Model().cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"  step={ckpt.get('step')}, val_loss={ckpt.get('val_loss')}")

    if args.variant.startswith("ts-"):
        from data import load_tokens
        val_tokens = load_tokens("val")
    else:
        from data_owt import load_owt_tokens
        val_tokens = load_owt_tokens("val")
    val_tokens = torch.from_numpy(val_tokens.astype(np.int64))
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(args.n_batches):
        starts = rng.integers(0, len(val_tokens) - args.block_size - 2, size=args.batch_size)
        x = torch.stack([val_tokens[s:s + args.block_size] for s in starts]).cuda()
        y = torch.stack([val_tokens[s + 1:s + 1 + args.block_size] for s in starts]).cuda()
        batches.append((x, y))

    # Reference loss
    with torch.no_grad():
        ref = float(np.mean([get_loss(model, x, y).item() for x, y in batches]))
    print(f"  reference loss = {ref:.4f}")

    params = [p for p in model.parameters() if p.requires_grad]

    print("\n--- power iteration (top Hessian eigenvalue) ---")
    lam = power_iter_top_eig(model, batches, params, n_iter=args.power_iters)

    print("\n--- Hutchinson trace ---")
    tr_mean, tr_std = hutchinson_trace(model, batches, params, n_probe=args.hutch_probes)

    print("\n--- 1D filter-normalized loss curve ---")
    alphas = np.linspace(-args.curve_max, args.curve_max, args.curve_pts).tolist()
    curve = loss_curve(model, batches, params, alphas)

    n = sum(p.numel() for p in params)
    print(f"\n=== {args.variant} sharpness summary ===")
    print(f"  ref loss            = {ref:.4f}")
    print(f"  lambda_max          = {lam:.4f}")
    print(f"  trace(H) (Hutch)    = {tr_mean:.2f}  +/- {tr_std:.2f}")
    print(f"  trace / n_params    = {tr_mean / n:.3e}")
    print(f"  curve loss range    = [{min(curve):.4f}, {max(curve):.4f}]  (delta {max(curve)-min(curve):.4f})")

    out_dir = "loss_sharpness_results"
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "variant": args.variant, "step": ckpt.get("step"), "val_loss": ckpt.get("val_loss"),
        "ref_loss": ref, "lambda_max": lam, "trace_mean": tr_mean, "trace_std": tr_std,
        "n_params": n, "alphas": alphas, "curve_losses": curve,
    }
    torch.save(out, os.path.join(out_dir, f"{args.variant}.pt"))
    print(f"saved → {out_dir}/{args.variant}.pt")


if __name__ == "__main__":
    main()
