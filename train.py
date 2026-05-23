"""Training script for Nesterov+Lie-Trotter YuriiFormer on TinyStories."""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from data import load_tokens, TinyStoriesDataset, ValidationDataset
from model import YuriiFormer

# ── Hyperparameters ──────────────────────────────────────────────────────────
TOTAL_STEPS = 10_000
WARMUP_STEPS = 1_000
BATCH_SIZE = 30
GRAD_ACCUM_STEPS = 16
BLOCK_SIZE = 1024
GRAD_CLIP = 1.0

# Optimizer LRs
MUON_LR = 0.02
ADAMW_LR = 6e-4
SCALAR_LR = 3e-3  # 5x AdamW LR

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
CKPT_DIR = "checkpoints"

# Seed
SEED = 42


def get_lr_multiplier(step: int) -> float:
    """Cosine schedule with linear warmup. Returns multiplier in [MIN_LR_RATIO, 1.0]."""
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizers(model: YuriiFormer):
    """Set up Muon for 2D weight matrices and AdamW for everything else."""
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
            # Fallback — shouldn't happen with this architecture
            ln_params.append(param)

    # Print parameter group assignments
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
        muon_params,
        lr=MUON_LR,
        momentum=MUON_MOMENTUM,
        nesterov=True,
        weight_decay=0.0,
    )

    optimizer_adamw = torch.optim.AdamW([
        {"params": embed_params, "lr": ADAMW_LR, "weight_decay": 0.1},
        {"params": ln_params, "lr": ADAMW_LR, "weight_decay": 0.0},
        {"params": scalar_params, "lr": SCALAR_LR, "weight_decay": 0.0},
    ], betas=(ADAM_BETA1, ADAM_BETA2))

    return optimizer_muon, optimizer_adamw


def update_lr(step: int, optimizer_muon, optimizer_adamw):
    """Update learning rates for all optimizer groups."""
    mult = get_lr_multiplier(step)
    for pg in optimizer_muon.param_groups:
        pg["lr"] = MUON_LR * mult
    for pg in optimizer_adamw.param_groups:
        if pg["weight_decay"] > 0:
            pg["lr"] = ADAMW_LR * mult
        elif len(pg["params"]) > 0 and pg["params"][0].numel() == 1:
            # Scalar group (each param has numel=1)
            pg["lr"] = SCALAR_LR * mult
        else:
            pg["lr"] = ADAMW_LR * mult


@torch.no_grad()
def evaluate(model, val_dataset, n_batches=VAL_BATCHES, batch_size=BATCH_SIZE):
    """Evaluate on validation set."""
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
    """Save a training checkpoint."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer_muon": optimizer_muon.state_dict(),
        "optimizer_adamw": optimizer_adamw.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
    }, path)
    print(f"  Saved checkpoint to {path}")


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    train_tokens = load_tokens("train")
    val_tokens = load_tokens("val")
    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Val:   {len(val_tokens):,} tokens")

    train_dataset = TinyStoriesDataset(train_tokens, BLOCK_SIZE, seed=SEED, device=device)
    val_dataset = ValidationDataset(val_tokens, BLOCK_SIZE, device=device)

    steps_per_epoch = len(train_dataset.block_starts) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total epochs: {TOTAL_STEPS / steps_per_epoch:.1f}")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n=== Building model ===")
    model = YuriiFormer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────────
    print("\n=== Configuring optimizers ===")
    optimizer_muon, optimizer_adamw = configure_optimizers(model)

    # ── Wandb ────────────────────────────────────────────────────────────
    config = {
        "model": "YuriiFormer-small",
        "variant": "Nesterov+Lie-Trotter",
        "n_layers": 12, "n_heads": 12, "d_model": 768,
        "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS,
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM_STEPS,
        "block_size": BLOCK_SIZE,
        "muon_lr": MUON_LR, "adamw_lr": ADAMW_LR, "scalar_lr": SCALAR_LR,
        "muon_momentum": MUON_MOMENTUM,
        "adam_betas": (ADAM_BETA1, ADAM_BETA2),
        "grad_clip": GRAD_CLIP,
        "train_tokens": len(train_tokens), "val_tokens": len(val_tokens),
    }
    wandb.init(project="yuriiformer", config=config)

    # ── Training ─────────────────────────────────────────────────────────
    print("\n=== Training ===")
    model.train()
    best_val_loss = float("inf")
    final_val_loss = float("inf")
    t0 = time.time()

    for step in range(TOTAL_STEPS):
        # Update learning rates
        update_lr(step, optimizer_muon, optimizer_adamw)

        # Zero gradients
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()

        # Gradient accumulation
        total_loss = 0.0
        for micro in range(GRAD_ACCUM_STEPS):
            x, y = train_dataset.get_batch(BATCH_SIZE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            total_loss += loss.item()

        train_loss = total_loss / GRAD_ACCUM_STEPS

        # Global gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Optimizer step
        optimizer_muon.step()
        optimizer_adamw.step()

        # Wandb logging (every step for loss curve)
        lr_mult = get_lr_multiplier(step)
        log_dict = {
            "train/loss": train_loss,
            "train/lr_muon": MUON_LR * lr_mult,
            "train/lr_adamw": ADAMW_LR * lr_mult,
            "train/lr_scalar": SCALAR_LR * lr_mult,
            "train/epoch": train_dataset.epoch,
        }

        # Console logging
        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            tokens_seen = (step + 1) * tokens_per_step
            print(f"step {step:>5d} | loss {train_loss:.4f} | "
                  f"lr_muon {MUON_LR * lr_mult:.5f} | "
                  f"epoch {train_dataset.epoch} | "
                  f"time {dt:.0f}s | "
                  f"tokens {tokens_seen/1e6:.1f}M")

        # Validation
        if step % VAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
            val_loss = evaluate(model, val_dataset)
            log_dict["val/loss"] = val_loss
            print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer_muon, optimizer_adamw,
                              step, val_loss, train_loss, "best")
            log_dict["val/best_loss"] = best_val_loss
            final_val_loss = val_loss

            # Log learned scalars
            for i, layer in enumerate(model.layers):
                log_dict[f"scalars/layer{i}_mu_attn"] = torch.sigmoid(layer.mu_attn_raw).item()
                log_dict[f"scalars/layer{i}_beta_attn"] = torch.sigmoid(layer.beta_attn_raw).item()
                log_dict[f"scalars/layer{i}_gamma_attn"] = F.softplus(layer.gamma_attn_raw).item()
                log_dict[f"scalars/layer{i}_mu_mlp"] = torch.sigmoid(layer.mu_mlp_raw).item()
                log_dict[f"scalars/layer{i}_beta_mlp"] = torch.sigmoid(layer.beta_mlp_raw).item()
                log_dict[f"scalars/layer{i}_gamma_mlp"] = F.softplus(layer.gamma_mlp_raw).item()

        wandb.log(log_dict, step=step)

    # ── Final ────────────────────────────────────────────────────────────
    # Save final checkpoint
    save_checkpoint(model, optimizer_muon, optimizer_adamw,
                  TOTAL_STEPS - 1, final_val_loss, train_loss, "final")

    total_time = time.time() - t0
    print(f"\n=== Training complete ===")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Best val loss:  {best_val_loss:.4f} (target: 1.078)")
    print(f"Final val loss: {final_val_loss:.4f} (target: 1.090)")
    print(f"Train loss @10k: {train_loss:.4f} (target: 0.896)")

    wandb.log({
        "final/best_val_loss": best_val_loss,
        "final/final_val_loss": final_val_loss,
        "final/train_loss_10k": train_loss,
        "final/total_time_hours": total_time / 3600,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
