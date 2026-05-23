"""Fine-tune a pretrained checkpoint on a different dataset to measure
plasticity (adaptation) and forgetting (degradation on pretrain dist).

Two directions:
  --direction owt2ts  : OWT-pretrained → fine-tune on TS  → eval both
  --direction ts2owt  : TS-pretrained  → fine-tune on OWT → eval both

Uses fixed AdamW for the fine-tune optimizer regardless of pretrain optimizer
(isolates "landscape the variant found" from "variant's update rule").
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Variant → (model_module, model_class, owt_ckpt, ts_ckpt)
VARIANT_MAP = {
    "vanilla":    ("vanilla_model", "VanillaTransformer",
                   "checkpoints_vanilla_owt/best.pt",
                   "checkpoints_ts_archive/vanilla_ts/best.pt"),
    "adam":       ("adam_model",    "AdamFormer",
                   "checkpoints_adam_owt/best.pt",
                   "checkpoints_ts_archive/adam_ts/best.pt"),
    "adamw":      ("adamw_model",   "AdamWFormer",
                   "checkpoints_adamw_owt/best.pt",
                   "checkpoints_ts_archive/adamw_ts/best.pt"),
    "vanilla_adamw": ("vanilla_model", "VanillaTransformer",
                   "checkpoints_vanilla_adamw_owt/best.pt",
                   None),
    "yurii":      ("model",         "YuriiFormer",
                   "checkpoints_yurii_owt/best.pt",
                   "checkpoints_ts_archive/yurii_ts/best.pt"),
    "tmm":        ("tmm_model",     "TMMFormer",
                   "checkpoints_tmm_owt/best.pt",
                   "checkpoints_ts_archive/tmm_ts/best.pt"),
    "muon":       ("muon_model",    "MuonFormer",
                   None,
                   "checkpoints_ts_archive/muon_ts/best.pt"),
    "yurii_sam":  ("model",         "YuriiFormer",
                   "checkpoints_yurii_sam_owt/best.pt",
                   "checkpoints_ts_archive/yurii_sam_ts/best.pt"),
    "yurii_wsd":  ("model",         "YuriiFormer",
                   "checkpoints_yurii_wsd_owt/best.pt",
                   "checkpoints_ts_archive/yurii_wsd_ts/best.pt"),
    "yurii_sawd": ("model",         "YuriiFormer",
                   "checkpoints_yurii_sawd_owt/best.pt",
                   "checkpoints_ts_archive/yurii_sawd_ts/best.pt"),
    "tmm_sam":    ("tmm_model",     "TMMFormer",
                   "checkpoints_tmm_sam_owt/best.pt",
                   "checkpoints_tmm_sam_ts/best.pt"),
    "tmm_wsd":    ("tmm_model",     "TMMFormer",
                   "checkpoints_tmm_wsd_owt/best.pt",
                   "checkpoints_tmm_wsd_ts/best.pt"),
    "tmm_sawd":   ("tmm_model",     "TMMFormer",
                   "checkpoints_tmm_sawd_owt/best.pt",
                   "checkpoints_tmm_sawd_ts/best.pt"),
}

BLOCK_SIZE = 1024
BATCH_SIZE = 8
TOTAL_GRAD_ACCUM = 60   # effective batch 480, matches pretrain
WARMUP_STEPS = 50
GRAD_CLIP = 1.0

# Fine-tune AdamW hyperparams (uniform across variants for fair comparison)
FT_LR = 1e-4
FT_WD = 0.01
FT_BETAS = (0.9, 0.95)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    return rank, local_rank, world_size, device


def load_data(direction: str, rank: int, device: str):
    """Returns (source_dataset, target_dataset, source_val_tokens, target_val_tokens).
    source = pretrain dist; target = fine-tune dist.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_tokens, TinyStoriesDataset, ValidationDataset
    from data_owt import load_owt_tokens, OWTDataset, OWTValidationDataset

    if direction == "owt2ts":
        # source = OWT, target = TS
        owt_train = load_owt_tokens("train")
        owt_val = load_owt_tokens("val")
        ts_train = load_tokens("train")
        ts_val = load_tokens("val")
        target_train_ds = TinyStoriesDataset(ts_train, BLOCK_SIZE, seed=42 + rank, device=device)
        return target_train_ds, owt_val, ts_val
    elif direction == "ts2owt":
        owt_train = load_owt_tokens("train")
        owt_val = load_owt_tokens("val")
        ts_train = load_tokens("train")
        ts_val = load_tokens("val")
        target_train_ds = OWTDataset(owt_train, BLOCK_SIZE, seed=42 + rank, device=device)
        return target_train_ds, ts_val, owt_val
    else:
        raise ValueError(f"unknown direction {direction}")


def load_model(variant: str, direction: str, device: str):
    if variant not in VARIANT_MAP:
        raise ValueError(f"unknown variant {variant}")
    mod_name, cls_name, owt_ckpt, ts_ckpt = VARIANT_MAP[variant]
    if direction == "owt2ts":
        ckpt_subpath = owt_ckpt
    elif direction == "ts2owt":
        if ts_ckpt is None:
            raise ValueError(f"variant {variant} has no TS checkpoint")
        ckpt_subpath = ts_ckpt
    else:
        raise ValueError

    mod = __import__(mod_name)
    cls = getattr(mod, cls_name)
    model = cls()

    cache = Path(os.environ["CACHE"])
    ckpt_path = cache / ckpt_subpath
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    return model, ckpt.get("val_loss", None)


@torch.no_grad()
def eval_loss(model, val_tokens: np.ndarray, n_batches: int, device: str, seed: int = 0) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        starts = rng.integers(0, len(val_tokens) - BLOCK_SIZE - 2, size=BATCH_SIZE)
        x = torch.tensor(np.stack([val_tokens[s:s + BLOCK_SIZE] for s in starts]),
                         dtype=torch.long, device=device)
        y = torch.tensor(np.stack([val_tokens[s + 1:s + 1 + BLOCK_SIZE] for s in starts]),
                         dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def lr_mult(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_MAP))
    parser.add_argument("--direction", required=True, choices=["owt2ts", "ts2owt"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_ddp()
    is_main = (rank == 0)
    grad_accum_per_gpu = TOTAL_GRAD_ACCUM // world_size

    if is_main:
        print(f"=== Fine-tune forgetting: {args.variant}  direction={args.direction} ===", flush=True)
        print(f"steps={args.steps}, lr={FT_LR}, world_size={world_size}, ga/gpu={grad_accum_per_gpu}", flush=True)

    # Load data
    if is_main: print(f"\n[1/4] Loading data ...", flush=True)
    target_train_ds, source_val_tokens, target_val_tokens = load_data(args.direction, rank, device)

    # Load model
    if is_main: print(f"[2/4] Loading model {args.variant} (direction={args.direction}) ...", flush=True)
    model, recorded_val_loss = load_model(args.variant, args.direction, device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  loaded {n_params:,} params, recorded val_loss={recorded_val_loss}", flush=True)

    # Eval T0 (before fine-tune)
    if is_main: print(f"\n[3/4] Eval T0 (before fine-tune) ...", flush=True)
    src_t0 = eval_loss(model, source_val_tokens, args.eval_batches, device, seed=0)
    tgt_t0 = eval_loss(model, target_val_tokens, args.eval_batches, device, seed=1)
    src_name = "owt" if args.direction == "owt2ts" else "ts"
    tgt_name = "ts"  if args.direction == "owt2ts" else "owt"
    if is_main:
        print(f"  T0: source ({src_name}) loss = {src_t0:.4f}", flush=True)
        print(f"  T0: target ({tgt_name}) loss = {tgt_t0:.4f}", flush=True)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # Optimizer (uniform AdamW for fair comparison)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WD, betas=FT_BETAS)

    # Fine-tune
    if is_main: print(f"\n[4/4] Fine-tune for {args.steps} steps ...", flush=True)
    history = []
    t0 = time.time()
    model.train()

    for step in range(args.steps):
        # LR update
        mul = lr_mult(step, args.steps)
        for pg in optimizer.param_groups:
            pg["lr"] = FT_LR * mul

        optimizer.zero_grad()
        total_loss = 0.0
        for micro in range(grad_accum_per_gpu):
            x, y = target_train_ds.get_batch(BATCH_SIZE)
            ctx = model.no_sync() if micro < grad_accum_per_gpu - 1 else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    scaled = loss / TOTAL_GRAD_ACCUM
                scaled.backward()
            total_loss += loss.item()

        loss_t = torch.tensor(total_loss / grad_accum_per_gpu, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        train_loss = loss_t.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Eval periodically
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            src_loss = eval_loss(model.module, source_val_tokens, args.eval_batches, device, seed=0)
            tgt_loss = eval_loss(model.module, target_val_tokens, args.eval_batches, device, seed=1)
            if is_main:
                elapsed = time.time() - t0
                print(f"  step {step+1:>4d} | train {train_loss:.4f} | "
                      f"src {src_loss:.4f} (Δ{src_loss-src_t0:+.4f}) | "
                      f"tgt {tgt_loss:.4f} (Δ{tgt_loss-tgt_t0:+.4f}) | "
                      f"lr_mul {mul:.3f} | {elapsed:.0f}s", flush=True)
            history.append({"step": step + 1, "train_loss": train_loss,
                            "source_loss": src_loss, "target_loss": tgt_loss,
                            "lr_mul": mul})

    # Final eval
    src_t1 = eval_loss(model.module, source_val_tokens, args.eval_batches, device, seed=0)
    tgt_t1 = eval_loss(model.module, target_val_tokens, args.eval_batches, device, seed=1)

    if is_main:
        results = {
            "variant": args.variant,
            "direction": args.direction,
            "steps": args.steps,
            "ft_lr": FT_LR,
            "n_params": sum(p.numel() for p in model.parameters()),
            "recorded_val_loss": recorded_val_loss,
            "T0": {"source_loss": src_t0, "target_loss": tgt_t0},
            "T1": {"source_loss": src_t1, "target_loss": tgt_t1},
            "forgetting": src_t1 - src_t0,
            "plasticity": tgt_t0 - tgt_t1,
            "history": history,
            "wall_seconds": time.time() - t0,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n=== DONE ({args.variant}, {args.direction}) ===")
        print(f"Forgetting (source loss Δ): {results['forgetting']:+.4f}")
        print(f"Plasticity (target loss drop): {results['plasticity']:+.4f}")
        print(f"Wall: {results['wall_seconds']:.0f}s")
        print(f"Saved to: {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
