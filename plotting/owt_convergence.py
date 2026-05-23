"""Plot VanillaTransformer convergence on OWT under two optimizer setups.

Same architecture (VanillaTransformer 12L/12H/d=768, 124M params), same data,
same DDP / batch / schedule. The only thing that differs is the optimizer:

- vanilla + AdamW   (`vanilla_adamw_train_owt.py`, job 7890344):
    pure AdamW for everything (2D weights, embeddings, LN/biases).
- vanilla + Muon+AdamW hybrid (`vanilla_train_owt.py`, job 6989892):
    Muon (lr=4e-3, momentum=0.95) for 2D matrix weights,
    AdamW (lr=6e-4) for embeddings / LN / biases.

This is the "optimizer A vs optimizer B on identical architecture" comparison,
NOT a comparison against MuonFormer (which is a different architecture).

Outputs:
  analysis/plots/owt_convergence.png
  analysis/owt_convergence.json
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([\d.]+)")
VAL_RE = re.compile(r"val_loss:\s+([\d.]+)")


def parse_log(path: Path):
    """Return dict with 'train' (list of (step, loss)) and 'val' (list of (step, loss))."""
    train, val = [], []
    last_step = None
    last_train = None
    with path.open() as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                train.append((step, loss))
                last_step = step
                last_train = loss
                continue
            m = VAL_RE.search(line)
            if m and last_step is not None:
                val.append((last_step, float(m.group(1))))
    return {"train": train, "val": val}


def stitch(logs):
    """Concatenate multiple log dicts. Resume runs overlap on first step → de-dupe by max step seen."""
    train_all, val_all = [], []
    seen_train, seen_val = set(), set()
    for d in logs:
        for s, v in d["train"]:
            if s not in seen_train:
                train_all.append((s, v))
                seen_train.add(s)
        for s, v in d["val"]:
            if s not in seen_val:
                val_all.append((s, v))
                seen_val.add(s)
    train_all.sort(key=lambda x: x[0])
    val_all.sort(key=lambda x: x[0])
    return {"train": train_all, "val": val_all}


def smooth(xy, win=50):
    """EMA over loss series for nicer plot lines."""
    if not xy:
        return xy
    out = []
    s = xy[0][1]
    alpha = 1.0 / max(1, win)
    for x, y in xy:
        s = (1 - alpha) * s + alpha * y
        out.append((x, s))
    return out


def main():
    base = Path(".")
    logs_dir = base / "logs"

    adamw = parse_log(logs_dir / "vanilla_adamw_owt_7890344.out")
    muon  = parse_log(logs_dir / "vanilla_owt_6989892.out")

    out = {
        "vanilla_adamw": {
            "optimizer": "pure AdamW (lr=6e-4)",
            "train_log": "vanilla_adamw_owt_7890344.out",
            "train_step_loss": adamw["train"],
            "val_step_loss": adamw["val"],
            "final_train_step": adamw["train"][-1][0] if adamw["train"] else None,
            "best_val": min((v for _, v in adamw["val"]), default=None),
        },
        "vanilla_muon_adamw_hybrid": {
            "optimizer": "Muon (lr=4e-3, mom=0.95) on 2D + AdamW (lr=6e-4) on embed/LN",
            "train_log": "vanilla_owt_6989892.out",
            "train_step_loss": muon["train"],
            "val_step_loss": muon["val"],
            "final_train_step": muon["train"][-1][0] if muon["train"] else None,
            "best_val": min((v for _, v in muon["val"]), default=None),
        },
    }
    Path("analysis").mkdir(exist_ok=True)
    with open("analysis/owt_convergence.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved analysis/owt_convergence.json")
    print(f"  vanilla-AdamW: {len(adamw['train'])} train pts, {len(adamw['val'])} val pts, "
          f"last step={out['vanilla_adamw']['final_train_step']}, "
          f"best val={out['vanilla_adamw']['best_val']:.4f}")
    print(f"  vanilla-Muon+AdamW hybrid : {len(muon['train'])} train pts, {len(muon['val'])} val pts, "
          f"last step={out['vanilla_muon_adamw_hybrid']['final_train_step']}, "
          f"best val={out['vanilla_muon_adamw_hybrid']['best_val']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def draw(ax, key_train, key_val):
        for name, color, marker in [
            ("vanilla_adamw", "#1f77b4", "o"),
            ("vanilla_muon_adamw_hybrid",  "#d62728", "s"),
        ]:
            label = "Vanilla + AdamW" if name == "vanilla_adamw" else "Vanilla + Muon (on 2D) + AdamW (on embed/LN)"
            if key_train:
                pts = out[name][key_train]
                if pts:
                    sm = smooth(pts, win=40)
                    xs, ys = zip(*sm)
                    ax.plot(xs, ys, color=color, lw=1.2, alpha=0.9, label=label)
            if key_val:
                pts = out[name][key_val]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=color, lw=1.4, marker=marker, ms=4, label=label)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    # Row 0: full range, log-y train, val
    draw(axes[0, 0], "train_step_loss", None)
    axes[0, 0].set_ylabel("train loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Train loss — full range (log y, EMA smoothed)")

    draw(axes[0, 1], None, "val_step_loss")
    axes[0, 1].set_ylabel("val loss")
    axes[0, 1].set_title("Val loss — full range")

    # Row 1: zoomed to step >= 2000 to see late-stage divergence
    draw(axes[1, 0], "train_step_loss", None)
    axes[1, 0].set_ylabel("train loss")
    axes[1, 0].set_xlim(2000, None)
    axes[1, 0].set_ylim(3.0, 4.0)
    axes[1, 0].set_title("Train loss — zoom (step ≥ 2000)")

    draw(axes[1, 1], None, "val_step_loss")
    axes[1, 1].set_ylabel("val loss")
    axes[1, 1].set_xlim(2000, None)
    axes[1, 1].set_ylim(3.0, 4.0)
    axes[1, 1].set_title("Val loss — zoom (step ≥ 2000)")

    fig.suptitle("OWT pretraining on VanillaTransformer (124M): pure AdamW vs Muon+AdamW hybrid",
                 fontsize=13)
    fig.tight_layout()

    out_png = Path("analysis/plots/owt_convergence.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
