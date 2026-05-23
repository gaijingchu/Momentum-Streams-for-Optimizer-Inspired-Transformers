"""Plot TS pretraining loss curves across optimizer variants (2x4 grid).

Mirror of plot_owt_all_optimizers.py for TinyStories. Same factorial design
(momentum x preconditioning) from docs/EMNLP_extension_plan.md §2.1.

Note: no MuonFormer-TS training log is preserved locally (the ckpt was
downloaded from HF), so Muon is annotated by its final val only and not
drawn as a curve.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([\d.]+)")
VAL_RE = re.compile(r"val_loss:\s+([\d.]+)")


def parse_log(path: Path):
    train, val = [], []
    last_step = None
    with path.open() as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                train.append((step, loss))
                last_step = step
                continue
            m = VAL_RE.search(line)
            if m and last_step is not None:
                val.append((last_step, float(m.group(1))))
    return {"train": train, "val": val}


def stitch(paths):
    train_all, val_all = [], []
    seen_t, seen_v = set(), set()
    for p in paths:
        if not p.exists():
            continue
        d = parse_log(p)
        for s, v in d["train"]:
            if s not in seen_t:
                train_all.append((s, v)); seen_t.add(s)
        for s, v in d["val"]:
            if s not in seen_v:
                val_all.append((s, v)); seen_v.add(s)
    train_all.sort(); val_all.sort()
    return {"train": train_all, "val": val_all}


def smooth(xy, win=40):
    if not xy:
        return xy
    out, s = [], xy[0][1]
    alpha = 1.0 / max(1, win)
    for x, y in xy:
        s = (1 - alpha) * s + alpha * y
        out.append((x, s))
    return out


LOG = Path("logs")

OPTS = {
    # Canonical TS runs (no-mom row + Polyak Adam family)
    "vanilla":  [LOG / "vanilla_debug_6989890.out"],
    "adam":     [LOG / "adamformer_4gpu_6953304.out"],
    "adamw":    [LOG / "adamwformer_debug_6962025.out"],
    "yurii":    [LOG / "yuriiformer_debug_7009665.out"],
    "tmm":      [LOG / "tmmformer_debug_6989889.out"],
    # No-momentum row: factorial ablation runs
    "HB":       [LOG / "hbformer_ts_7823549.out"],
    "RMSProp":  [LOG / "rmspropformer_ts_7836445.out"],
    "Ortho":    [LOG / "orthoformer_ts_7836446.out",
                 LOG / "orthoformer_ts_7850113.out"],
    "Shampoo":  [LOG / "shampooformer_ts_7840591.out",
                 LOG / "shampooformer_ts_7865859.out",
                 LOG / "shampooformer_ts_7866963.out"],
}

data = {name: stitch(paths) for name, paths in OPTS.items()}

print(f"{'optimizer':<10} {'n_train':>7} {'n_val':>5} {'last_step':>9} {'best_val':>8}")
summary = {}
for name, d in data.items():
    if not d["train"]:
        continue
    last_step = d["train"][-1][0]
    best_val = min((v for _, v in d["val"]), default=None)
    summary[name] = {"last_step": last_step, "best_val": best_val,
                     "n_train": len(d["train"]), "n_val": len(d["val"])}
    line = f"{name:<10} {len(d['train']):>7} {len(d['val']):>5} {last_step:>9}"
    if best_val is not None:
        line += f" {best_val:>8.4f}"
    print(line)

# Muon TS not trainable from logs; annotate from HF ckpt evaluation
summary["Muon (from HF ckpt)"] = {"best_val": 1.1503, "last_step": 9600,
                                  "note": "no local training log"}

Path("analysis").mkdir(exist_ok=True)
with open("analysis/ts_optimizer_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

COLORS = {
    "vanilla":  "#1f77b4",
    "adam":     "#9467bd",
    "adamw":    "#17becf",
    "yurii":    "#bcbd22",
    "tmm":      "#7f7f7f",
    "HB":       "#ff7f0e",
    "RMSProp":  "#8c564b",
    "Ortho":    "#e377c2",
    "Shampoo":  "#2ca02c",
}

GROUPS = [
    ("Adam family",                 ["vanilla", "adam", "adamw"]),
    ("Pure-momentum (none-precond)",["vanilla", "HB", "yurii", "tmm"]),
    ("No-momentum × precond",       ["vanilla", "RMSProp", "Ortho", "Shampoo"]),
    ("All optimizers",              ["vanilla", "adam", "adamw", "yurii", "tmm",
                                     "HB", "RMSProp", "Ortho", "Shampoo"]),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False)

for col, (title, names) in enumerate(GROUPS):
    ax_t = axes[0, col]
    ax_v = axes[1, col]
    for name in names:
        if name not in data or not data[name]["train"]:
            continue
        c = COLORS[name]
        sm = smooth(data[name]["train"], win=40)
        xs, ys = zip(*sm)
        ax_t.plot(xs, ys, color=c, lw=1.4, alpha=0.9, label=name)
        if data[name]["val"]:
            xs, ys = zip(*data[name]["val"])
            ax_v.plot(xs, ys, color=c, lw=1.2, marker="o", ms=3, alpha=0.85, label=name)
    ax_t.set_title(f"{title} — train")
    ax_v.set_title(f"{title} — val")
    ax_t.set_xlabel("step"); ax_t.set_ylabel("train loss")
    ax_v.set_xlabel("step"); ax_v.set_ylabel("val loss")
    ax_t.set_yscale("log")
    ax_t.grid(alpha=0.3); ax_v.grid(alpha=0.3)
    ax_t.legend(fontsize=8, loc="upper right")
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.set_ylim(1.05, 2.5)

fig.suptitle("TinyStories pretraining loss across optimizers (VanillaTransformer 60M backbone)",
             fontsize=14)
fig.tight_layout()

out = Path("analysis/plots/ts_all_optimizers.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved {out}")
