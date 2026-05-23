"""Parse SLURM .out files to reconstruct val-loss curves per variant and plot
them as grouped figures (one panel per dataset: TinyStories and OpenWebText).

Train loss is noisy (per-step minibatch loss); we plot the evaluated validation
loss only. Each variant's curve is reconstructed by:
  1. Finding all .out files matching the variant glob
  2. Sorting by job id (last suffix)
  3. Parsing `step NNNN | loss X.XXXX` to track the most recent training step
  4. Parsing `val_loss: X.XXXX (best: ...)` and assigning it that step
  5. Later runs at the same step overwrite earlier runs (resumes)
"""
import glob
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt

LOGS = "logs"

OWT_VARIANTS = OrderedDict([
    ("Vanilla",         "vanilla_owt_*.out"),
    ("YuriiFormer",     "yurii_owt_*.out"),
    ("TMMFormer",       "tmm_owt_*.out"),
    ("AdamFormer",      "adam_owt_*.out"),
    ("AdamWFormer",     "adamw_owt_*.out"),
    ("YuriiFormer+WSD", "yurii_wsd_owt_*.out"),
    ("YuriiFormer+SAM", "yurii_sam_owt_*.out"),
    ("YuriiFormer+SAWD","yurii_sawd_owt_*.out"),
])

# TS baselines predate the SAM/WSD/SAWD naming scheme.
TS_VARIANTS = OrderedDict([
    ("Vanilla",         ["vanilla_debug_*.out"]),
    ("YuriiFormer",     ["yuriiformer_debug_*.out"]),
    ("TMMFormer",       ["tmmformer_debug_*.out"]),
    ("AdamFormer",      ["adamformer_4gpu_*.out", "adamformer_train_*.out",
                         "adamformer_2gpu_*.out"]),
    ("AdamWFormer",     ["adamwformer_general_*.out", "adamwformer_preempt_*.out",
                         "adamwformer_debug_*.out"]),
    ("YuriiFormer+WSD", ["yurii_wsd_ts_*.out"]),
    ("YuriiFormer+SAM", ["yurii_sam_ts_*.out"]),
    ("YuriiFormer+SAWD","yurii_sawd_ts_*.out"),
])

COLORS = {
    "Vanilla":          "#888888",
    "YuriiFormer":      "#1f77b4",
    "TMMFormer":        "#2ca02c",
    "AdamFormer":       "#d62728",
    "AdamWFormer":      "#ff7f0e",
    "YuriiFormer+WSD":  "#9467bd",
    "YuriiFormer+SAM":  "#17becf",
    "YuriiFormer+SAWD": "#e377c2",
}

STEP_RE = re.compile(r"^step\s+(\d+)\s*\|\s*loss\s+([\d.]+)")
VAL_RE  = re.compile(r"^\s*val_loss:\s+([\d.]+)")
RESUME_RE = re.compile(r"Resumed at step\s+(\d+)")


def job_id(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"_(\d+)\.out$", base)
    return int(m.group(1)) if m else 0


def parse_variant(globs):
    if isinstance(globs, str):
        globs = [globs]
    files = []
    for g in globs:
        files.extend(glob.glob(os.path.join(LOGS, g)))
    files = sorted(set(files), key=job_id)

    # step -> val_loss (later runs overwrite earlier)
    val_at_step = {}
    for f in files:
        cur_step = 0
        with open(f) as fh:
            for line in fh:
                rm = RESUME_RE.search(line)
                if rm:
                    cur_step = int(rm.group(1))
                    continue
                sm = STEP_RE.match(line)
                if sm:
                    cur_step = int(sm.group(1))
                    continue
                vm = VAL_RE.match(line)
                if vm:
                    val_at_step[cur_step] = float(vm.group(1))
    return sorted(val_at_step.items())


def plot_panel(ax, variants, title, xlim, ylim, label_legend=True):
    for name, pat in variants.items():
        pts = parse_variant(pat)
        if not pts:
            continue
        steps, losses = zip(*pts)
        is_new = name.startswith("YuriiFormer+")
        lw = 2.2 if is_new else 1.3
        alpha = 1.0 if is_new else 0.75
        ms = 3.0 if is_new else 2.0
        ax.plot(steps, losses, label=name, color=COLORS[name],
                linewidth=lw, alpha=alpha, marker='o', markersize=ms)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (nats)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    if label_legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)


def main():
    print("=== TinyStories ===")
    for n, p in TS_VARIANTS.items():
        pts = parse_variant(p)
        if pts:
            print(f"  {n:18} {len(pts):4d} pts, "
                  f"final={pts[-1][1]:.4f}, min={min(v for _, v in pts):.4f}")
    print("=== OpenWebText ===")
    for n, p in OWT_VARIANTS.items():
        pts = parse_variant(p)
        if pts:
            print(f"  {n:18} {len(pts):4d} pts, "
                  f"final={pts[-1][1]:.4f}, min={min(v for _, v in pts):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_panel(axes[0, 0], TS_VARIANTS,
               "TinyStories — full run (10k steps)",
               xlim=(0, 10000), ylim=(1.05, 1.60))
    plot_panel(axes[0, 1], OWT_VARIANTS,
               "OpenWebText — full run (30k steps)",
               xlim=(0, 30000), ylim=(2.90, 3.25))
    plot_panel(axes[1, 0], TS_VARIANTS,
               "TinyStories — late training (step 6000–10000)",
               xlim=(6000, 10000), ylim=(1.070, 1.175),
               label_legend=False)
    plot_panel(axes[1, 1], OWT_VARIANTS,
               "OpenWebText — late training (step 20000–30000)",
               xlim=(20000, 30000), ylim=(2.925, 3.020),
               label_legend=False)

    fig.suptitle("Validation loss curves across variants — TinyStories (left) & OpenWebText (right)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    out = "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
