"""Measure per-layer attention entropy for each transformer variant.

Monkey-patches CausalSelfAttention.forward to compute attention weights
manually (instead of fused SDPA) and accumulates Shannon entropy of the
softmax distribution, averaged over heads, queries (causal-valid only),
and tokens in the batch.
"""
import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = {
    # OWT variants
    "vanilla":  ("vanilla_model",  "VanillaTransformer", "checkpoints_vanilla_owt"),
    "yurii":    ("model",          "YuriiFormer",        "checkpoints_yurii_owt"),
    "tmm":      ("tmm_model",      "TMMFormer",          "checkpoints_tmm_owt"),
    "adam":     ("adam_model",     "AdamFormer",         "checkpoints_adam_owt"),
    "adamw":    ("adamw_model",    "AdamWFormer",        "checkpoints_adamw_owt"),
    "wsd":      ("model",          "YuriiFormer",        "checkpoints_yurii_wsd_owt"),
    "sam":      ("model",          "YuriiFormer",        "checkpoints_yurii_sam_owt"),
    "sawd":     ("model",          "YuriiFormer",        "checkpoints_yurii_sawd_owt"),
    # TinyStories variants
    "ts-vanilla":  ("vanilla_model",  "VanillaTransformer", "checkpoints_ts_archive/vanilla_ts"),
    "ts-yurii":    ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_ts"),
    "ts-tmm":      ("tmm_model",      "TMMFormer",          "checkpoints_ts_archive/tmm_ts"),
    "ts-adam":     ("adam_model",     "AdamFormer",         "checkpoints_ts_archive/adam_ts"),
    "ts-adamw":    ("adamw_model",    "AdamWFormer",        "checkpoints_ts_archive/adamw_ts"),
    "ts-wsd":      ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_wsd_ts"),
    "ts-sam":      ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_sam_ts"),
    "ts-sawd":     ("model",          "YuriiFormer",        "checkpoints_ts_archive/yurii_sawd_ts"),
    # TMM + schedule variants (TS)
    "ts-tmm-wsd":  ("tmm_model",      "TMMFormer",          "checkpoints_tmm_wsd_ts"),
    "ts-tmm-sam":  ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sam_ts"),
    "ts-tmm-sawd": ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sawd_ts"),
    # TMM + schedule variants (OWT)
    "tmm-wsd":     ("tmm_model",      "TMMFormer",          "checkpoints_tmm_wsd_owt"),
    "tmm-sam":     ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sam_owt"),
    "tmm-sawd":    ("tmm_model",      "TMMFormer",          "checkpoints_tmm_sawd_owt"),
}


def patch_attention(module_obj):
    """Replace CausalSelfAttention.forward with a version that records entropy."""
    Cls = module_obj.CausalSelfAttention
    Cls._entropy_records = []  # list of per-call (layer_idx_unknown, mean_entropy_scalar)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (B,H,T,T)

        # Per-(B,H,query) entropy in nats. Skip query 0 (degenerate, single key).
        with torch.no_grad():
            p = attn.clamp_min(1e-12)
            ent = -(p * p.log()).sum(dim=-1)  # (B,H,T)
            ent = ent[:, :, 1:]  # drop trivial first query
            # mean over batch + valid queries → (H,)
            per_head = ent.mean(dim=(0, 2)).detach().cpu()
            Cls._entropy_records.append(per_head)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.out_proj(out)

    Cls.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=1024)
    args = ap.parse_args()

    cache = os.environ.get("CACHE", "./checkpoints_cache")
    mod_name, cls_name, ckpt_subdir = VARIANTS[args.variant]
    ckpt_path = os.path.join(cache, ckpt_subdir, "best.pt")
    print(f"[{args.variant}] loading {ckpt_path}")

    module_obj = __import__(mod_name)
    patch_attention(module_obj)
    Model = getattr(module_obj, cls_name)
    model = Model().cuda().eval()

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    print(f"  step={ckpt.get('step')}, val_loss={ckpt.get('val_loss')}")

    # Load val tokens (TS or OWT depending on variant prefix)
    if args.variant.startswith("ts-"):
        from data import load_tokens
        val_tokens = load_tokens("val")
    else:
        from data_owt import load_owt_tokens
        val_tokens = load_owt_tokens("val")
    val_tokens = torch.from_numpy(val_tokens.astype(np.int64))

    n_layers = len(model.blocks) if hasattr(model, "blocks") else None
    Cls = module_obj.CausalSelfAttention

    rng = np.random.default_rng(0)
    all_layer_entropies = []  # list[(n_layers, n_heads)]

    with torch.no_grad():
        for b in range(args.n_batches):
            Cls._entropy_records = []
            starts = rng.integers(0, len(val_tokens) - args.block_size - 1, size=args.batch_size)
            x = torch.stack([val_tokens[s:s + args.block_size] for s in starts]).cuda()
            _ = model(x)
            # records is list of length n_layers, each (H,)
            stack = torch.stack(Cls._entropy_records, dim=0)  # (L, H)
            all_layer_entropies.append(stack)
            print(f"  batch {b+1}/{args.n_batches} done")

    avg = torch.stack(all_layer_entropies, dim=0).mean(dim=0)  # (L, H)
    n_layers, n_heads = avg.shape
    uniform_max = math.log(args.block_size)  # max possible entropy

    print(f"\n=== {args.variant} attention entropy (nats) ===")
    print(f"max possible (uniform over {args.block_size}) = {uniform_max:.3f}")
    print(f"{'layer':>5} {'mean':>8} {'min_h':>8} {'max_h':>8}")
    for L in range(n_layers):
        row = avg[L]
        print(f"{L:5d} {row.mean():8.4f} {row.min():8.4f} {row.max():8.4f}")
    print(f"\noverall mean = {avg.mean():.4f}")

    out_dir = "attention_entropy_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.variant}.pt")
    torch.save({"variant": args.variant, "per_layer_per_head": avg,
                "step": ckpt.get("step"), "val_loss": ckpt.get("val_loss")}, out_path)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
