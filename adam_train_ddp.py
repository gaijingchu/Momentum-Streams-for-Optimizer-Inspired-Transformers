"""DDP training script for Adam+Lie-Trotter AdamFormer on TinyStories (multi-GPU).

Supports checkpoint resume via --resume flag.
Uses torch.compile for faster training.
"""

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

from data import load_tokens, TinyStoriesDataset, ValidationDataset
from adam_model import AdamFormer

# ── Hyperparameters ──────────────────────────────────────────────────────────
TOTAL_STEPS = 10_000
WARMUP_STEPS = 1_000
BATCH_SIZE = 8
TOTAL_GRAD_ACCUM = 60  # total micro-steps across all GPUs
BLOCK_SIZE = 1024
GRAD_CLIP = 1.0

# Optimizer LRs
MUON_LR = 0.02
ADAMW_LR = 6e-4
SCALAR_LR = 3e-3

# AdamW betas
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95

# Muon
MUON_MOMENTUM = 0.95

# Schedule
MIN_LR_RATIO = 0.1

# Validation
VAL_INTERVAL = 100
VAL_BATCHES = 160

# Logging
LOG_INTERVAL = 10

# Checkpointing
CKPT_DIR = "checkpoints_adam"

# Seed
SEED = 42


def get_lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizers(model):
    muon_params = []
    embed_params = []
    ln_params = []
    scalar_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "_raw" in name:
            scalar_params.append(param)
        elif "ln" in name:
            ln_params.append(param)
        elif "emb" in name:
            embed_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            ln_params.append(param)

    n_muon = sum(p.numel() for p in muon_params)
    n_embed = sum(p.numel() for p in embed_params)
    n_ln = sum(p.numel() for p in ln_params)
    n_scalar = sum(p.numel() for p in scalar_params)
    print(f"Optimizer groups:")
    print(f"  Muon (2D weights):  {len(muon_params):>3} params, {n_muon:>12,} elements, lr={MUON_LR}")
    print(f"  AdamW (embeddings): {len(embed_params):>3} params, {n_embed:>12,} elements, lr={ADAMW_LR}, wd=0.1")
    print(f"  AdamW (LayerNorm):  {len(ln_params):>3} params, {n_ln:>12,} elements, lr={ADAMW_LR}, wd=0")
    print(f"  AdamW (scalars):    {len(scalar_params):>3} params, {n_scalar:>12,} elements, lr={SCALAR_LR}, wd=0")

    optimizer_muon = torch.optim.Muon(
        muon_params, lr=MUON_LR, momentum=MUON_MOMENTUM, nesterov=True, weight_decay=0.0,
    )
    optimizer_adamw = torch.optim.AdamW([
        {"params": embed_params, "lr": ADAMW_LR, "weight_decay": 0.1},
        {"params": ln_params, "lr": ADAMW_LR, "weight_decay": 0.0},
        {"params": scalar_params, "lr": SCALAR_LR, "weight_decay": 0.0},
    ], betas=(ADAM_BETA1, ADAM_BETA2))

    return optimizer_muon, optimizer_adamw


def update_lr(step, optimizer_muon, optimizer_adamw):
    mult = get_lr_multiplier(step)
    for pg in optimizer_muon.param_groups:
        pg["lr"] = MUON_LR * mult
    for pg in optimizer_adamw.param_groups:
        if pg["weight_decay"] > 0:
            pg["lr"] = ADAMW_LR * mult
        elif len(pg["params"]) > 0 and pg["params"][0].numel() == 1:
            pg["lr"] = SCALAR_LR * mult
        else:
            pg["lr"] = ADAMW_LR * mult


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


def save_checkpoint(model, optimizer_muon, optimizer_adamw, step, val_loss, train_loss, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    raw_model = model.module if hasattr(model, "module") else model
    # Unwrap torch.compile wrapper if present
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    torch.save({
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer_muon": optimizer_muon.state_dict(),
        "optimizer_adamw": optimizer_adamw.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
    }, path)
    print(f"  Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── DDP setup ────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    assert TOTAL_GRAD_ACCUM % world_size == 0, \
        f"TOTAL_GRAD_ACCUM ({TOTAL_GRAD_ACCUM}) must be divisible by world_size ({world_size})"
    grad_accum_per_gpu = TOTAL_GRAD_ACCUM // world_size

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED + rank)

    if rank == 0:
        print(f"DDP: {world_size} GPUs, {grad_accum_per_gpu} accum steps/GPU")
        print(f"Effective batch size: {BATCH_SIZE * TOTAL_GRAD_ACCUM}")
        print(f"Device: {device}, GPU: {torch.cuda.get_device_name(local_rank)}")

    # ── Data ─────────────────────────────────────────────────────────────
    if rank == 0:
        print("\n=== Loading data ===")
    train_tokens = load_tokens("train")
    val_tokens = load_tokens("val")
    if rank == 0:
        print(f"Train: {len(train_tokens):,} tokens")
        print(f"Val:   {len(val_tokens):,} tokens")

    train_dataset = TinyStoriesDataset(train_tokens, BLOCK_SIZE, seed=SEED + rank, device=device)
    val_dataset = ValidationDataset(val_tokens, BLOCK_SIZE, device=device)

    tokens_per_step = BATCH_SIZE * TOTAL_GRAD_ACCUM * BLOCK_SIZE

    # ── Model ────────────────────────────────────────────────────────────
    if rank == 0:
        print("\n=== Building model ===")
    model = AdamFormer().to(device)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # torch.compile for speed
    model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank])

    # ── Optimizer ────────────────────────────────────────────────────────
    if rank == 0:
        print("\n=== Configuring optimizers ===")
    raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
    optimizer_muon, optimizer_adamw = configure_optimizers(raw_model)

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Strip _orig_mod. prefix from keys (saved from torch.compile'd model)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        raw_model.load_state_dict(state_dict)
        optimizer_muon.load_state_dict(ckpt["optimizer_muon"])
        optimizer_adamw.load_state_dict(ckpt["optimizer_adamw"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["val_loss"]
        if rank == 0:
            print(f"  Resumed at step {start_step}, best val_loss: {best_val_loss:.4f}")
        # Fast-forward the data iterator to approximate position
        skip_batches = start_step * TOTAL_GRAD_ACCUM
        batches_per_rank = skip_batches // world_size
        for _ in range(batches_per_rank):
            train_dataset.get_batch(BATCH_SIZE)
        if rank == 0:
            print(f"  Skipped ~{skip_batches} batches to sync data position")

    # ── Wandb (rank 0 only) ──────────────────────────────────────────────
    if rank == 0:
        config = {
            "model": "AdamFormer-small",
            "variant": "Adam+Lie-Trotter",
            "n_layers": 12, "n_heads": 12, "d_model": 768,
            "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS,
            "batch_size": BATCH_SIZE, "grad_accum": TOTAL_GRAD_ACCUM,
            "world_size": world_size,
            "block_size": BLOCK_SIZE,
            "muon_lr": MUON_LR, "adamw_lr": ADAMW_LR, "scalar_lr": SCALAR_LR,
            "muon_momentum": MUON_MOMENTUM,
            "adam_betas": (ADAM_BETA1, ADAM_BETA2),
            "grad_clip": GRAD_CLIP,
            "resumed_from_step": start_step,
        }
        wandb.init(project="adamformer", config=config, resume="allow")

    # ── Training ─────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n=== Training (steps {start_step} -> {TOTAL_STEPS}) ===")
    model.train()
    final_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, TOTAL_STEPS):
        update_lr(step, optimizer_muon, optimizer_adamw)

        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()

        total_loss = 0.0
        for micro in range(grad_accum_per_gpu):
            x, y = train_dataset.get_batch(BATCH_SIZE)
            if micro < grad_accum_per_gpu - 1:
                ctx = model.no_sync()
            else:
                ctx = torch.enable_grad()
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

        optimizer_muon.step()
        optimizer_adamw.step()

        if rank == 0:
            lr_mult = get_lr_multiplier(step)
            log_dict = {
                "train/loss": train_loss,
                "train/lr_muon": MUON_LR * lr_mult,
                "train/lr_adamw": ADAMW_LR * lr_mult,
                "train/lr_scalar": SCALAR_LR * lr_mult,
                "train/epoch": train_dataset.epoch,
            }

            if step % LOG_INTERVAL == 0:
                dt = time.time() - t0
                tokens_seen = (step + 1) * tokens_per_step
                steps_done = step - start_step + 1
                steps_left = TOTAL_STEPS - step - 1
                sec_per_step = dt / steps_done if steps_done > 0 else 0
                eta_h = steps_left * sec_per_step / 3600
                print(f"step {step:>5d} | loss {train_loss:.4f} | "
                      f"lr_muon {MUON_LR * lr_mult:.5f} | "
                      f"epoch {train_dataset.epoch} | "
                      f"time {dt:.0f}s | "
                      f"{sec_per_step:.1f}s/step | "
                      f"ETA {eta_h:.1f}h")

            if step % VAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
                val_loss = evaluate(raw_model, val_dataset)
                log_dict["val/loss"] = val_loss
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer_muon, optimizer_adamw,
                                  step, val_loss, train_loss, "best")
                log_dict["val/best_loss"] = best_val_loss
                final_val_loss = val_loss

                for i, layer in enumerate(raw_model.layers):
                    log_dict[f"scalars/layer{i}_beta1_attn"] = torch.sigmoid(layer.beta1_attn_raw).item()
                    log_dict[f"scalars/layer{i}_beta2_attn"] = torch.sigmoid(layer.beta2_attn_raw).item()
                    log_dict[f"scalars/layer{i}_gamma_attn"] = F.softplus(layer.gamma_attn_raw).item()
                    log_dict[f"scalars/layer{i}_beta1_mlp"] = torch.sigmoid(layer.beta1_mlp_raw).item()
                    log_dict[f"scalars/layer{i}_beta2_mlp"] = torch.sigmoid(layer.beta2_mlp_raw).item()
                    log_dict[f"scalars/layer{i}_gamma_mlp"] = F.softplus(layer.gamma_mlp_raw).item()

            wandb.log(log_dict, step=step)

        dist.barrier()

    # ── Final ────────────────────────────────────────────────────────────
    if rank == 0:
        save_checkpoint(model, optimizer_muon, optimizer_adamw,
                      TOTAL_STEPS - 1, final_val_loss, train_loss, "final")

        total_time = time.time() - t0
        print(f"\n=== Training complete ===")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Best val loss:  {best_val_loss:.4f}")
        print(f"Final val loss: {final_val_loss:.4f}")
        print(f"Train loss @10k: {train_loss:.4f}")

        wandb.log({
            "final/best_val_loss": best_val_loss,
            "final/final_val_loss": final_val_loss,
            "final/train_loss_10k": train_loss,
            "final/total_time_hours": total_time / 3600,
        })
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
