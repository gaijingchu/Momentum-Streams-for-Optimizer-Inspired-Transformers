"""Plot OWT pretraining loss curves across all available optimizers (2x4 grid).

Layout: 2 rows x 4 cols.
  Row 0 = train loss (EMA-smoothed, log-y);  Row 1 = val loss.
  Col 0 = Adam family    : vanilla-AdamW  vs  Adam
  Col 1 = Muon family    : vanilla-AdamW  vs  Muon (vanilla arch + Muon optim)
  Col 2 = 2nd-order      : vanilla-AdamW  vs  Shampoo  vs  HB (heavy-ball)
  Col 3 = All overlay    : every optimizer on a single panel

For optimizers with multiple consecutive logs (Muon: 3 logs; HB: 2 logs;
Ortho: 2 logs; AdamW: 1 log), we de-dupe overlapping steps and stitch.
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


LOG_DIR = Path("logs")

# Build per-optimizer stitched trajectories
OPTS = {
    "vanilla-AdamW": [LOG_DIR / "vanilla_adamw_owt_7890344.out"],
    "Adam":          [LOG_DIR / "adam_owt_6963486.out"],
    "AdamW":         [LOG_DIR / "adamw_owt_6962638.out"],
    "Muon":          [LOG_DIR / "muonformer_owt_7880493.out",
                      LOG_DIR / "muonformer_owt_7884940.out",
                      LOG_DIR / "muonformer_owt_7890332.out"],
    "Shampoo":       [LOG_DIR / "shampooformer_owt_7840592.out"],
    "HB":            [LOG_DIR / "hbformer_owt_7865858.out",
                      LOG_DIR / "hbformer_owt_7866962.out",
                      LOG_DIR / "hbformer_owt_7873959.out"],
    "RMSProp":       [LOG_DIR / "rmspropformer_owt_7840304.out"],
    "Ortho":         [LOG_DIR / "orthoformer_owt_7840305.out",
                      LOG_DIR / "orthoformer_owt_7880494.out",
                      LOG_DIR / "orthoformer_owt_7880496.out"],
}

data = {name: stitch(paths) for name, paths in OPTS.items()}

# Summary print
print(f"{'optimizer':<14} {'n_train':>7} {'n_val':>5} {'last_step':>9} {'best_val':>8}")
summary = {}
for name, d in data.items():
    if not d["train"]:
        continue
    last_step = d["train"][-1][0]
    best_val = min((v for _, v in d["val"]), default=None)
    summary[name] = {"last_step": last_step, "best_val": best_val,
                     "n_train": len(d["train"]), "n_val": len(d["val"])}
    print(f"{name:<14} {len(d['train']):>7} {len(d['val']):>5} "
          f"{last_step:>9} {best_val:>8.4f}" if best_val else "")

Path("analysis").mkdir(exist_ok=True)
with open("analysis/owt_optimizer_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Color palette
COLORS = {
    "vanilla-AdamW": "#1f77b4",   # blue (baseline)
    "Adam":          "#9467bd",   # purple
    "AdamW":         "#17becf",   # cyan
    "Muon":          "#d62728",   # red
    "Shampoo":       "#2ca02c",   # green
    "HB":            "#ff7f0e",   # orange
    "RMSProp":       "#8c564b",   # brown
    "Ortho":         "#e377c2",   # pink
}

GROUPS = [
    ("Adam family",       ["vanilla-AdamW", "Adam", "AdamW"]),
    ("Orthogonalized momentum",   ["vanilla-AdamW", "Muon"]),
    ("2nd-order / heavy-ball",    ["vanilla-AdamW", "Shampoo", "HB"]),
    ("All optimizers",            ["vanilla-AdamW", "Adam", "AdamW", "Muon",
                                   "Shampoo", "HB", "RMSProp", "Ortho"]),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False)

for col, (title, names) in enumerate(GROUPS):
    ax_t = axes[0, col]
    ax_v = axes[1, col]
    for name in names:
        if name not in data or not data[name]["train"]:
            continue
        c = COLORS[name]
        # train (EMA smoothed) on row 0
        sm = smooth(data[name]["train"], win=40)
        xs, ys = zip(*sm)
        ax_t.plot(xs, ys, color=c, lw=1.4, alpha=0.9, label=name)
        # val (raw, with markers) on row 1
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
    ax_v.set_ylim(2.9, 6.5)

fig.suptitle("OWT pretraining loss across optimizers (VanillaTransformer 124M backbone)",
             fontsize=14)
fig.tight_layout()

out = Path("analysis/plots/owt_all_optimizers.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved {out}")
