"""DDP training script for VanillaTransformer trained with **AdamW only** on OpenWebText.

Same architecture as vanilla_train_owt.py, but Muon (for 2D weights) is replaced
with AdamW.  Serves as the pure-AdamW baseline in the factorial-ablation table.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from data.openwebtext import load_owt_tokens, OWTDataset, OWTValidationDataset
from formers.vanilla.model import VanillaTransformer

# ── Hyperparameters (OWT) ────────────────────────────────────────────────────
TOTAL_STEPS = 30_000
WARMUP_STEPS = 3_000
BATCH_SIZE = 8
TOTAL_GRAD_ACCUM = 60
BLOCK_SIZE = 1024
GRAD_CLIP = 1.0

# Single learning rate for all AdamW groups (nanoGPT / GPT-2 standard for 125M).
LR = float(os.environ.get("LR", "6e-4"))
MIN_LR_RATIO = 0.1

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
WD_2D = 0.1     # weight decay on 2D matrix weights (attn, MLP)
WD_EMBED = 0.1  # weight decay on embeddings
WD_LN = 0.0     # no weight decay on LayerNorm / 1D params

VAL_INTERVAL = 100
VAL_BATCHES = 160
LOG_INTERVAL = 10

CACHE = os.environ.get("CACHE", "./checkpoints_cache")
LR_SUFFIX = os.environ.get("LR_SUFFIX", "")
CKPT_DIR = os.path.join(CACHE, f"checkpoints_vanilla_adamw_owt{LR_SUFFIX}")

SEED = 42


def get_lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizer(model):
    weight_2d_params, embed_params, ln_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln" in name:
            ln_params.append(param)
        elif "emb" in name:
            embed_params.append(param)
        elif param.ndim == 2:
            weight_2d_params.append(param)
        else:
            ln_params.append(param)

    print("Optimizer groups (single AdamW):")
    print(f"  2D weights:    {len(weight_2d_params):>3} params, lr={LR}, wd={WD_2D}")
    print(f"  Embeddings:    {len(embed_params):>3} params, lr={LR}, wd={WD_EMBED}")
    print(f"  LN / 1D:       {len(ln_params):>3} params, lr={LR}, wd={WD_LN}")

    optimizer = torch.optim.AdamW(
        [
            {"params": weight_2d_params, "weight_decay": WD_2D},
            {"params": embed_params,     "weight_decay": WD_EMBED},
            {"params": ln_params,        "weight_decay": WD_LN},
        ],
        lr=LR, betas=(ADAM_BETA1, ADAM_BETA2),
    )
    return optimizer


def update_lr(step, optimizer):
    mult = get_lr_multiplier(step)
    for pg in optimizer.param_groups:
        pg["lr"] = LR * mult


@torch.no_grad()
def evaluate(model, val_dataset, n_batches=VAL_BATCHES, batch_size=BATCH_SIZE):
    model.eval()
    val_dataset.reset()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = val_dataset.get_batch(batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def save_checkpoint(model, optimizer, step, val_loss, train_loss, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    torch.save({
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
    }, path)
    print(f"  Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    assert TOTAL_GRAD_ACCUM % world_size == 0
    grad_accum_per_gpu = TOTAL_GRAD_ACCUM // world_size

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED + rank)

    if rank == 0:
        print(f"DDP: {world_size} GPUs, {grad_accum_per_gpu} accum/GPU")
        print(f"Effective batch: {BATCH_SIZE * TOTAL_GRAD_ACCUM}")

    if rank == 0:
        print("\n=== Loading data ===")
    train_tokens = load_owt_tokens("train")
    val_tokens = load_owt_tokens("val")
    if rank == 0:
        print(f"Train: {len(train_tokens):,} tokens; Val: {len(val_tokens):,}")

    train_dataset = OWTDataset(train_tokens, BLOCK_SIZE, seed=SEED + rank, device=device)
    val_dataset = OWTValidationDataset(val_tokens, BLOCK_SIZE, device=device)

    if rank == 0:
        print("\n=== Building VanillaTransformer ===")
    model = VanillaTransformer().to(device)
    if rank == 0:
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print("\n=== Configuring optimizer (pure AdamW) ===")
    raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
    optimizer = configure_optimizer(raw_model)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        raw_model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["val_loss"]
        if rank == 0:
            print(f"  Resumed at step {start_step}, best val_loss: {best_val_loss:.4f}")
        skip_batches = start_step * TOTAL_GRAD_ACCUM
        for _ in range(skip_batches // world_size):
            train_dataset.get_batch(BATCH_SIZE)

    if rank == 0:
        config = {
            "model": "VanillaTransformer-small-OWT",
            "variant": "vanilla-adamw (pure AdamW baseline)",
            "dataset": "OpenWebText",
            "n_layers": 12, "n_heads": 12, "d_model": 768,
            "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS,
            "batch_size": BATCH_SIZE, "grad_accum": TOTAL_GRAD_ACCUM,
            "world_size": world_size, "block_size": BLOCK_SIZE,
            "lr": LR, "min_lr_ratio": MIN_LR_RATIO,
            "betas": (ADAM_BETA1, ADAM_BETA2),
            "wd_2d": WD_2D, "wd_embed": WD_EMBED, "wd_ln": WD_LN,
            "grad_clip": GRAD_CLIP, "resumed_from_step": start_step,
        }
        wandb.init(project="optformer-factorial-ablation-owt",
                   name=f"vanilla-adamw-owt{LR_SUFFIX}", config=config, resume="allow")

    if rank == 0:
        print(f"\n=== Training (steps {start_step} -> {TOTAL_STEPS}) ===")
    model.train()
    final_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, TOTAL_STEPS):
        update_lr(step, optimizer)
        optimizer.zero_grad()

        total_loss = 0.0
        for micro in range(grad_accum_per_gpu):
            x, y = train_dataset.get_batch(BATCH_SIZE)
            ctx = model.no_sync() if micro < grad_accum_per_gpu - 1 else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    scaled_loss = loss / TOTAL_GRAD_ACCUM
                scaled_loss.backward()
            total_loss += loss.item()

        loss_tensor = torch.tensor(total_loss / grad_accum_per_gpu, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        train_loss = loss_tensor.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if rank == 0:
            lr_mult = get_lr_multiplier(step)
            log_dict = {"train/loss": train_loss, "train/lr": LR * lr_mult,
                        "train/epoch": train_dataset.epoch}

            if step % LOG_INTERVAL == 0:
                dt = time.time() - t0
                steps_done = step - start_step + 1
                sec_per_step = dt / steps_done if steps_done > 0 else 0
                eta_h = (TOTAL_STEPS - step - 1) * sec_per_step / 3600
                print(f"step {step:>5d} | loss {train_loss:.4f} | "
                      f"lr {LR * lr_mult:.5f} | epoch {train_dataset.epoch} | "
                      f"time {dt:.0f}s | {sec_per_step:.1f}s/step | ETA {eta_h:.1f}h")

            if step % VAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
                val_loss = evaluate(raw_model, val_dataset)
                log_dict["val/loss"] = val_loss
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss, train_loss, "best")
                log_dict["val/best_loss"] = best_val_loss
                final_val_loss = val_loss

            wandb.log(log_dict, step=step)

        dist.barrier()

    if rank == 0:
        save_checkpoint(model, optimizer, TOTAL_STEPS - 1,
                        final_val_loss, train_loss, "final")
        total_time = time.time() - t0
        print(f"\n=== Training complete ===")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Best val loss:  {best_val_loss:.4f}")
        print(f"Final val loss: {final_val_loss:.4f}")
        print(f"Train loss @30k: {train_loss:.4f}")
        wandb.log({"final/best_val_loss": best_val_loss,
                   "final/final_val_loss": final_val_loss,
                   "final/train_loss_30k": train_loss,
                   "final/total_time_hours": total_time / 3600})
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
