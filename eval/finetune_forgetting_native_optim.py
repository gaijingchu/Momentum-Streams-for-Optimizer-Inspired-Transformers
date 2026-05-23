"""Fine-tune for forgetting/plasticity using the SAME optimizer as pretrain.

Companion to finetune_forgetting.py, which uses uniform AdamW for all variants
(isolating the landscape from the update rule). This script does the opposite:
it uses each variant's *native* pretrain optimizer at FT scale, to ask whether
the matched-optimizer FT changes the forgetting/plasticity numbers.

Currently supports: --variant vanilla (Muon-on-2D + AdamW-on-embed/LN hybrid).
Other variants would need their native optimizer wired in below.

Hyperparameter rationale
========================
Pretrain (`vanilla_train_owt.py`):
    Muon : lr=4e-3, momentum=0.95, nesterov=True, wd=0
    AdamW: lr=6e-4, betas=(0.9, 0.95), wd=0.1 on embed, wd=0 on LN
    Cosine schedule with 3k warmup over 30k steps; min ratio 0.1.

Canonical FT (`finetune_forgetting.py`):
    AdamW: lr=1e-4, wd=0.01, betas=(0.9, 0.95)
    50-step linear warmup → cosine to 0.1×. 1000 steps.
    The 1e-4 / 6e-4 = 1/6 ratio scales pretrain → FT for the AdamW group.

Native Muon FT (this script):
    Apply the same 1/6 scale to BOTH groups, mirror pretrain wd policy:
        Muon : lr = 4e-3 / 6 ≈ 6.67e-4, momentum=0.95, nesterov=True, wd=0
        AdamW(embed): lr=1e-4, wd=0.1
        AdamW(LN)   : lr=1e-4, wd=0
    50-step linear warmup → cosine to 0.1×. 1000 steps.

Refs:
- Liu, Su et al. (Moonlight, arXiv 2502.16982): "when the SFT optimizer differs
  from the pretraining optimizer, SFT with Muon does not show a significant
  advantage over AdamW" → matched-optimizer FT is what gives Muon's pretrain
  benefit a chance to carry through.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

# Subset of VARIANT_MAP: only variants whose native optimizer is wired here
VARIANT_MAP = {
    "vanilla": {
        "module": "vanilla_model",
        "class":  "VanillaTransformer",
        "owt_ckpt": "checkpoints_vanilla_owt/best.pt",
        "ts_ckpt":  "checkpoints_ts_archive/vanilla_ts/best.pt",
        "optim_kind": "muon_adamw_hybrid",
    },
}

BLOCK_SIZE = 1024
BATCH_SIZE = 8
TOTAL_GRAD_ACCUM = 60
WARMUP_STEPS = 50
GRAD_CLIP = 1.0

# Native FT hyperparams (1/6 of pretrain LRs; pretrain wd policy)
FT_MUON_LR = 4e-3 / 6.0           # ≈ 6.67e-4
FT_ADAMW_LR = 6e-4 / 6.0          # = 1e-4 (matches canonical AdamW FT)
FT_MUON_MOMENTUM = 0.95
FT_ADAMW_BETAS = (0.9, 0.95)
FT_WD_2D = 0.0                     # Muon wd
FT_WD_EMBED = 0.1                  # AdamW embed wd (mirror pretrain)
FT_WD_LN = 0.0                     # AdamW LN wd (mirror pretrain)
MIN_LR_RATIO = 0.1


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    return rank, local_rank, world_size, device


def load_data(direction: str, rank: int, device: str):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.tinystories import load_tokens, TinyStoriesDataset, ValidationDataset
    from data.openwebtext import load_owt_tokens, OWTDataset, OWTValidationDataset

    if direction == "owt2ts":
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
        raise ValueError(f"variant {variant} not in native-optim map")
    info = VARIANT_MAP[variant]
    if direction == "owt2ts":
        ckpt_subpath = info["owt_ckpt"]
    elif direction == "ts2owt":
        ckpt_subpath = info["ts_ckpt"]
        if ckpt_subpath is None:
            raise ValueError(f"variant {variant} has no TS checkpoint")
    else:
        raise ValueError

    mod = __import__(info["module"])
    cls = getattr(mod, info["class"])
    model = cls()

    cache = Path(os.environ["CACHE"])
    ckpt_path = cache / ckpt_subpath
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    return model, ckpt.get("val_loss", None), info["optim_kind"]


def configure_native_optimizer(raw_model, kind: str):
    """Build the matched-pretrain optimizer at FT scale.

    Returns (optimizers_dict, group_log_strs).
    """
    if kind == "muon_adamw_hybrid":
        muon_params, embed_params, ln_params = [], [], []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "ln" in name:
                ln_params.append(param)
            elif "emb" in name:
                embed_params.append(param)
            elif param.ndim == 2:
                muon_params.append(param)
            else:
                ln_params.append(param)

        opt_muon = torch.optim.Muon(
            muon_params, lr=FT_MUON_LR, momentum=FT_MUON_MOMENTUM,
            nesterov=True, weight_decay=FT_WD_2D,
        )
        opt_adamw = torch.optim.AdamW([
            {"params": embed_params, "lr": FT_ADAMW_LR, "weight_decay": FT_WD_EMBED},
            {"params": ln_params,    "lr": FT_ADAMW_LR, "weight_decay": FT_WD_LN},
        ], betas=FT_ADAMW_BETAS)

        log = [
            f"  Muon (2D weights):  {len(muon_params)} params, lr={FT_MUON_LR:.5g}, mom={FT_MUON_MOMENTUM}, wd={FT_WD_2D}",
            f"  AdamW (embed):      {len(embed_params)} params, lr={FT_ADAMW_LR:.5g}, wd={FT_WD_EMBED}",
            f"  AdamW (LN/1D):      {len(ln_params)} params, lr={FT_ADAMW_LR:.5g}, wd={FT_WD_LN}",
        ]
        return {"muon": opt_muon, "adamw": opt_adamw}, log

    raise ValueError(f"unknown optim_kind {kind}")


def update_lrs(opts, mul: float, kind: str):
    if kind == "muon_adamw_hybrid":
        for pg in opts["muon"].param_groups:
            pg["lr"] = FT_MUON_LR * mul
        for pg in opts["adamw"].param_groups:
            pg["lr"] = FT_ADAMW_LR * mul


@torch.no_grad()
def eval_loss(model, val_tokens, n_batches, device, seed=0):
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


def lr_mult(step, total_steps):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


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
        print(f"=== Native-optim fine-tune: {args.variant}  direction={args.direction} ===", flush=True)
        print(f"steps={args.steps}, world_size={world_size}, ga/gpu={grad_accum_per_gpu}", flush=True)

    if is_main: print("\n[1/4] Loading data ...", flush=True)
    target_train_ds, source_val_tokens, target_val_tokens = load_data(args.direction, rank, device)

    if is_main: print(f"[2/4] Loading model {args.variant} (direction={args.direction}) ...", flush=True)
    model, recorded_val_loss, optim_kind = load_model(args.variant, args.direction, device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  loaded {n_params:,} params, recorded val_loss={recorded_val_loss}", flush=True)

    if is_main: print("\n[3/4] Eval T0 (before fine-tune) ...", flush=True)
    src_t0 = eval_loss(model, source_val_tokens, args.eval_batches, device, seed=0)
    tgt_t0 = eval_loss(model, target_val_tokens, args.eval_batches, device, seed=1)
    src_name = "owt" if args.direction == "owt2ts" else "ts"
    tgt_name = "ts"  if args.direction == "owt2ts" else "owt"
    if is_main:
        print(f"  T0: source ({src_name}) loss = {src_t0:.4f}", flush=True)
        print(f"  T0: target ({tgt_name}) loss = {tgt_t0:.4f}", flush=True)

    model = DDP(model, device_ids=[local_rank])
    raw_model = model.module
    opts, optim_log = configure_native_optimizer(raw_model, optim_kind)
    if is_main:
        print(f"\n[ optimizer ({optim_kind}) ]")
        for line in optim_log:
            print(line, flush=True)

    if is_main: print(f"\n[4/4] Fine-tune for {args.steps} steps ...", flush=True)
    history = []
    t0 = time.time()
    model.train()

    for step in range(args.steps):
        mul = lr_mult(step, args.steps)
        update_lrs(opts, mul, optim_kind)

        for o in opts.values():
            o.zero_grad()

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
        for o in opts.values():
            o.step()

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

    src_t1 = eval_loss(model.module, source_val_tokens, args.eval_batches, device, seed=0)
    tgt_t1 = eval_loss(model.module, target_val_tokens, args.eval_batches, device, seed=1)

    if is_main:
        results = {
            "variant": args.variant,
            "direction": args.direction,
            "steps": args.steps,
            "ft_optimizer": optim_kind,
            "ft_muon_lr": FT_MUON_LR,
            "ft_adamw_lr": FT_ADAMW_LR,
            "ft_wd_2d": FT_WD_2D,
            "ft_wd_embed": FT_WD_EMBED,
            "ft_wd_ln": FT_WD_LN,
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
        print(f"\n=== DONE ({args.variant}, {args.direction}, native optim) ===")
        print(f"Forgetting (source loss Δ): {results['forgetting']:+.4f}")
        print(f"Plasticity (target loss drop): {results['plasticity']:+.4f}")
        print(f"Wall: {results['wall_seconds']:.0f}s")
        print(f"Saved to: {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
