"""DDP training of YuriiFormer on TinyStories with sharpness-aware interventions.

Modes (selected via --mode):
  cosine : baseline (cosine schedule, no SAM)             ~1x compute
  sam    : cosine schedule + Sharpness-Aware Minimization ~2x compute
  wsd    : warmup-stable-decay schedule, no SAM           ~1x compute
  sawd   : WSD schedule + SAM only during decay phase     ~1.17x compute

SAM (Foret et al. 2021):
    epsilon* = rho * g / ||g||_2
    1) forward+backward at theta to get g
    2) ascend: theta <- theta + epsilon*
    3) forward+backward at theta+epsilon* to get g_pert (using SAME minibatches)
    4) descend: theta <- theta - eta * g_pert; theta <- theta - epsilon* (undo)
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
from model import YuriiFormer

# ── Hyperparameters ─────────────────────────────────────────────────────────
TOTAL_STEPS = 10_000
WARMUP_STEPS = 1_000
DECAY_START_STEP = 8_300  # WSD: stable until here, then linear decay (83% like OWT)
BATCH_SIZE = 30
TOTAL_GRAD_ACCUM = 16
BLOCK_SIZE = 1024
GRAD_CLIP = 1.0

MUON_LR = 0.02
ADAMW_LR = 6e-4
SCALAR_LR = 3e-3

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
MUON_MOMENTUM = 0.95
MIN_LR_RATIO = 0.1

SAM_RHO = 0.05

VAL_INTERVAL = 100
VAL_BATCHES = 160
LOG_INTERVAL = 10

CKPT_DIR = "checkpoints"

SEED = 42


# ── LR schedules ────────────────────────────────────────────────────────────
def lr_mult_cosine(step: int) -> float:
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


def lr_mult_wsd(step: int) -> float:
    """Warmup -> Stable -> linear Decay to MIN_LR_RATIO."""
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    if step < DECAY_START_STEP:
        return 1.0
    progress = (step - DECAY_START_STEP) / max(1, TOTAL_STEPS - DECAY_START_STEP)
    return 1.0 - (1.0 - MIN_LR_RATIO) * progress


def in_decay_phase(step: int) -> bool:
    return step >= DECAY_START_STEP


def configure_optimizers(model):
    muon_params, embed_params, ln_params, scalar_params = [], [], [], []
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

    optimizer_muon = torch.optim.Muon(
        muon_params, lr=MUON_LR, momentum=MUON_MOMENTUM, nesterov=True, weight_decay=0.0,
    )
    optimizer_adamw = torch.optim.AdamW([
        {"params": embed_params, "lr": ADAMW_LR, "weight_decay": 0.1},
        {"params": ln_params, "lr": ADAMW_LR, "weight_decay": 0.0},
        {"params": scalar_params, "lr": SCALAR_LR, "weight_decay": 0.0},
    ], betas=(ADAM_BETA1, ADAM_BETA2))
    return optimizer_muon, optimizer_adamw


def update_lr(step, optimizer_muon, optimizer_adamw, lr_mult_fn):
    mult = lr_mult_fn(step)
    for pg in optimizer_muon.param_groups:
        pg["lr"] = MUON_LR * mult
    for pg in optimizer_adamw.param_groups:
        if pg["weight_decay"] > 0:
            pg["lr"] = ADAMW_LR * mult
        elif len(pg["params"]) > 0 and pg["params"][0].numel() == 1:
            pg["lr"] = SCALAR_LR * mult
        else:
            pg["lr"] = ADAMW_LR * mult
    return mult


# ── SAM helpers ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sam_ascent(params, rho):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return [None] * len(params)
    device = grads[0].device
    grad_norm = torch.norm(
        torch.stack([g.detach().norm(2).to(device) for g in grads]), p=2
    )
    scale = rho / (grad_norm + 1e-12)
    e_ws = []
    for p in params:
        if p.grad is None:
            e_ws.append(None)
            continue
        e_w = p.grad.detach() * scale
        p.add_(e_w)
        e_ws.append(e_w)
    return e_ws


@torch.no_grad()
def sam_undo(params, e_ws):
    for p, e_w in zip(params, e_ws):
        if e_w is not None:
            p.sub_(e_w)


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


def save_checkpoint(model, optimizer_muon, optimizer_adamw, step, val_loss, train_loss, name, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{name}.pt")
    raw_model = model.module if hasattr(model, "module") else model
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
    parser.add_argument("--mode", required=True, choices=["cosine", "sam", "wsd", "sawd"])
    parser.add_argument("--rho", type=float, default=SAM_RHO)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.mode in ("cosine", "sam"):
        lr_mult_fn = lr_mult_cosine
    else:
        lr_mult_fn = lr_mult_wsd

    def use_sam_at(step):
        if args.mode == "sam":
            return True
        if args.mode == "sawd":
            return in_decay_phase(step)
        return False

    ckpt_dir = os.path.join(CKPT_DIR, f"yurii_{args.mode}")

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
        print(f"=== Mode: {args.mode}  rho={args.rho if args.mode in ('sam','sawd') else 'n/a'} ===")
        print(f"DDP: {world_size} GPUs, {grad_accum_per_gpu} accum/GPU")
        print(f"Effective batch: {BATCH_SIZE * TOTAL_GRAD_ACCUM}")
        print(f"Schedule: warmup={WARMUP_STEPS}  "
              f"{'decay_start='+str(DECAY_START_STEP) if args.mode in ('wsd','sawd') else 'cosine'}")
        print(f"Checkpoint dir: {ckpt_dir}")

    # ── Data ─────────────────────────────────────────────────────────────
    if rank == 0:
        print("\n=== Loading data ===")
    train_tokens = load_tokens("train")
    val_tokens = load_tokens("val")

    train_dataset = TinyStoriesDataset(train_tokens, BLOCK_SIZE, seed=SEED + rank, device=device)
    val_dataset = ValidationDataset(val_tokens, BLOCK_SIZE, device=device)

    if rank == 0:
        print(f"Train: {len(train_tokens):,} tokens")
        print(f"Val:   {len(val_tokens):,} tokens")
        tokens_per_step = BATCH_SIZE * TOTAL_GRAD_ACCUM * BLOCK_SIZE
        print(f"Tokens per step: {tokens_per_step:,}")

    # ── Model ────────────────────────────────────────────────────────────
    if rank == 0:
        print("\n=== Building YuriiFormer ===")
    model = YuriiFormer().to(device)
    if rank == 0:
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    raw_model = model.module._orig_mod if hasattr(model.module, "_orig_mod") else model.module
    optimizer_muon, optimizer_adamw = configure_optimizers(raw_model)

    all_params = [p for p in model.parameters() if p.requires_grad]

    start_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        raw_model.load_state_dict(state_dict)
        optimizer_muon.load_state_dict(ckpt["optimizer_muon"])
        optimizer_adamw.load_state_dict(ckpt["optimizer_adamw"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["val_loss"]
        if rank == 0:
            print(f"  Resumed at step {start_step}, best val_loss: {best_val_loss:.4f}")
        skip_batches = start_step * TOTAL_GRAD_ACCUM
        for _ in range(skip_batches // world_size):
            train_dataset.get_batch(BATCH_SIZE)

    if rank == 0:
        config = {
            "model": "YuriiFormer-small-TinyStories",
            "mode": args.mode,
            "rho": args.rho,
            "schedule": "cosine" if args.mode in ("cosine", "sam") else "wsd",
            "decay_start_step": DECAY_START_STEP if args.mode in ("wsd", "sawd") else None,
            "n_layers": 12, "n_heads": 12, "d_model": 768,
            "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS,
            "batch_size": BATCH_SIZE, "grad_accum": TOTAL_GRAD_ACCUM,
            "world_size": world_size, "block_size": BLOCK_SIZE,
            "muon_lr": MUON_LR, "adamw_lr": ADAMW_LR, "scalar_lr": SCALAR_LR,
            "muon_momentum": MUON_MOMENTUM,
            "grad_clip": GRAD_CLIP, "resumed_from_step": start_step,
        }
        wandb.init(project="yuriiformer-ts-flat",
                   name=f"yurii-{args.mode}",
                   config=config, resume="allow")

    # ── Training ─────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n=== Training (steps {start_step} -> {TOTAL_STEPS}) ===")
    model.train()
    final_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, TOTAL_STEPS):
        lr_mult = update_lr(step, optimizer_muon, optimizer_adamw, lr_mult_fn)
        sam_active = use_sam_at(step)

        # ── Pass 1: clean forward/backward, cache batches for possible reuse ──
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        cached = []
        total_loss = 0.0
        for micro in range(grad_accum_per_gpu):
            x, y = train_dataset.get_batch(BATCH_SIZE)
            cached.append((x, y))
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

        # ── SAM ascent + pass 2 (replay cached batches) ───────────────────────
        if sam_active:
            e_ws = sam_ascent(all_params, args.rho)
            optimizer_muon.zero_grad()
            optimizer_adamw.zero_grad()
            for micro, (x, y) in enumerate(cached):
                ctx = model.no_sync() if micro < grad_accum_per_gpu - 1 else torch.enable_grad()
                with ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        scaled_loss = loss / TOTAL_GRAD_ACCUM
                    scaled_loss.backward()
            sam_undo(all_params, e_ws)
            del e_ws

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer_muon.step()
        optimizer_adamw.step()

        if rank == 0:
            log_dict = {"train/loss": train_loss,
                        "train/lr_muon": MUON_LR * lr_mult,
                        "train/lr_mult": lr_mult,
                        "train/sam_active": int(sam_active),
                        "train/epoch": train_dataset.epoch}

            if step % LOG_INTERVAL == 0:
                dt = time.time() - t0
                steps_done = step - start_step + 1
                sec_per_step = dt / steps_done if steps_done > 0 else 0
                eta_h = (TOTAL_STEPS - step - 1) * sec_per_step / 3600
                print(f"step {step:>5d} | loss {train_loss:.4f} | "
                      f"lr_mult {lr_mult:.3f} | sam={int(sam_active)} | "
                      f"epoch {train_dataset.epoch} | "
                      f"{sec_per_step:.1f}s/step | ETA {eta_h:.1f}h")

            if step % VAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
                val_loss = evaluate(raw_model, val_dataset)
                log_dict["val/loss"] = val_loss
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer_muon, optimizer_adamw,
                                    step, val_loss, train_loss, "best", ckpt_dir)
                log_dict["val/best_loss"] = best_val_loss
                final_val_loss = val_loss

            wandb.log(log_dict, step=step)

        dist.barrier()

    if rank == 0:
        save_checkpoint(model, optimizer_muon, optimizer_adamw,
                        TOTAL_STEPS - 1, final_val_loss, train_loss, "final", ckpt_dir)
        total_time = time.time() - t0
        print(f"\n=== Training complete ({args.mode}) ===")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Best val loss:  {best_val_loss:.4f}")
        print(f"Final val loss: {final_val_loss:.4f}")
        wandb.log({"final/best_val_loss": best_val_loss,
                   "final/final_val_loss": final_val_loss,
                   "final/train_loss_10k": train_loss,
                   "final/total_time_hours": total_time / 3600})
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
