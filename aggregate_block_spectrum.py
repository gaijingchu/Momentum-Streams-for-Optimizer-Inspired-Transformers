"""Aggregate Sandbox B block-Jacobian spectrum results across variants.

Inputs:  block_spectrum_results/<variant>.json
Outputs: block_spectrum_results/_aggregate.json
         analysis/plots/block_spectrum_kappa_vs_depth.png  (per-variant per-layer)
         analysis/plots/sandbox_a_vs_b.png                  (side-by-side comparison)

Pulls val_loss labels from spectrum_results/_aggregate.json (Sandbox A) so
identical val numbers appear in both tables.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ["vanilla", "adam", "adamw", "yurii", "tmm",
            "yurii-sam", "yurii-wsd", "tmm-sam", "tmm-wsd"]


def load_val_loss():
    a = json.load(open("spectrum_results/_aggregate.json"))
    return {v["variant"]: v["val_loss"] for v in a["per_variant"]}


def summarize(path: Path, val_loss: float):
    d = json.load(path.open())
    per_layer = d["per_layer"]
    layer_mean_kappa = []
    layer_mean_sigma_max = []
    layer_mean_sigma_min = []
    layer_mean_srank = []
    for L in per_layer:
        kappa = np.mean([b["kappa_eff"] for b in L["per_batch"]])
        smax = np.mean([b["sigma_max"] for b in L["per_batch"]])
        smin = np.mean([b["sigma_min_eff"] for b in L["per_batch"]])
        srank = np.mean([b["stable_rank"] for b in L["per_batch"]])
        layer_mean_kappa.append(float(kappa))
        layer_mean_sigma_max.append(float(smax))
        layer_mean_sigma_min.append(float(smin))
        layer_mean_srank.append(float(srank))

    return {
        "variant": d["variant"],
        "aux_kind": d["aux_kind"],
        "val_loss": val_loss,
        "mean_sigma_max": float(np.mean(layer_mean_sigma_max)),
        "mean_sigma_min": float(np.mean(layer_mean_sigma_min)),
        "mean_kappa": float(np.mean(layer_mean_kappa)),
        "max_kappa": float(np.max(layer_mean_kappa)),
        "mean_stable_rank": float(np.mean(layer_mean_srank)),
        "prod_sigma_min": float(np.prod(layer_mean_sigma_min)),
        "log_prod_sigma_min": float(np.sum(np.log(layer_mean_sigma_min))),
        "per_layer_kappa": layer_mean_kappa,
        "per_layer_sigma_max": layer_mean_sigma_max,
        "per_layer_sigma_min": layer_mean_sigma_min,
    }


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    d2 = np.sum((rx - ry) ** 2)
    return float(1.0 - 6 * d2 / (n * (n * n - 1)))


def main():
    val_map = load_val_loss()
    per_variant = []
    for v in VARIANTS:
        p = Path(f"block_spectrum_results/{v}.json")
        if not p.exists():
            print(f"  MISSING {p}")
            continue
        per_variant.append(summarize(p, val_map[v]))
        s = per_variant[-1]
        print(f"  {v:>10}  meanκ={s['mean_kappa']:6.3f}  maxκ={s['max_kappa']:6.3f}  "
              f"srank={s['mean_stable_rank']:8.1f}  log∏σmin={s['log_prod_sigma_min']:7.3f}  val={s['val_loss']:.3f}")

    # Correlations across variants
    vals = [v["val_loss"] for v in per_variant]
    correlations = {
        "spearman(mean_kappa, val_loss)":     spearman([v["mean_kappa"] for v in per_variant], vals),
        "spearman(max_kappa, val_loss)":      spearman([v["max_kappa"] for v in per_variant], vals),
        "spearman(stable_rank, val_loss)":    spearman([v["mean_stable_rank"] for v in per_variant], vals),
        "spearman(log_prod_sigma_min, val)":  spearman([v["log_prod_sigma_min"] for v in per_variant], vals),
    }
    print("\nCorrelations across variants:")
    for k, v in correlations.items():
        print(f"  {k}: {v:+.3f}" if v is not None else f"  {k}: n/a")

    out = {"per_variant": per_variant, "correlations": correlations}
    Path("block_spectrum_results/_aggregate.json").write_text(json.dumps(out, indent=2))
    print("\nSaved block_spectrum_results/_aggregate.json")

    # Render per-variant per-layer kappa plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    groups = [
        ("Sgd / Adam family", ["vanilla", "adam", "adamw"]),
        ("Yurii family",      ["yurii", "yurii-sam", "yurii-wsd"]),
        ("TMM family",        ["tmm", "tmm-sam", "tmm-wsd"]),
    ]
    colors = {
        "vanilla": "#444444", "adam": "#888888", "adamw": "#bbbbbb",
        "yurii": "#1f77b4", "yurii-sam": "#aec7e8", "yurii-wsd": "#7fa9d9",
        "tmm": "#d62728", "tmm-sam": "#ff9896", "tmm-wsd": "#e15a5b",
    }
    by_name = {v["variant"]: v for v in per_variant}
    for ax, (title, names) in zip(axes, groups):
        for n in names:
            if n not in by_name:
                continue
            ys = by_name[n]["per_layer_kappa"]
            ax.plot(range(len(ys)), ys, marker="o", lw=1.4, color=colors[n], label=n)
        ax.set_title(title)
        ax.set_xlabel("layer ℓ")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("mean κ_eff (block Jacobian, Sandbox B)")
    fig.suptitle("Sandbox B: block-Jacobian per-layer condition number κ_eff(F_l)",
                 fontsize=12)
    fig.tight_layout()
    out_png = Path("analysis/plots/block_spectrum_kappa_vs_depth.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved {out_png}")

    # Side-by-side Sandbox A vs B comparison
    a_agg = json.load(open("spectrum_results/_aggregate.json"))
    a_by_name = {v["variant"]: v for v in a_agg["per_variant"]}

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    names = [v["variant"] for v in per_variant]
    xs = np.arange(len(names))
    a_mean = [a_by_name[n]["mean_kappa"] for n in names]
    b_mean = [by_name[n]["mean_kappa"] for n in names]
    a_max = [a_by_name[n]["max_kappa"] for n in names]
    b_max = [by_name[n]["max_kappa"] for n in names]

    w = 0.4
    axes2[0].bar(xs - w/2, a_mean, w, label="Sandbox A (canonical R_l)",
                 color="#1f77b4", alpha=0.85)
    axes2[0].bar(xs + w/2, b_mean, w, label="Sandbox B (full F_l incl. wrapper)",
                 color="#d62728", alpha=0.85)
    axes2[0].set_xticks(xs)
    axes2[0].set_xticklabels(names, rotation=30, ha="right")
    axes2[0].set_ylabel("mean κ (avg over 12 layers)")
    axes2[0].set_title("mean κ per variant — Sandbox A vs B")
    axes2[0].legend(fontsize=9)
    axes2[0].grid(alpha=0.3, axis="y")

    axes2[1].bar(xs - w/2, a_max, w, label="Sandbox A", color="#1f77b4", alpha=0.85)
    axes2[1].bar(xs + w/2, b_max, w, label="Sandbox B", color="#d62728", alpha=0.85)
    axes2[1].set_xticks(xs)
    axes2[1].set_xticklabels(names, rotation=30, ha="right")
    axes2[1].set_ylabel("max κ across layers")
    axes2[1].set_title("max κ per variant — Sandbox A vs B")
    axes2[1].legend(fontsize=9)
    axes2[1].grid(alpha=0.3, axis="y")

    fig2.suptitle("Local Jacobian spectra: canonical residual vs full block (with wrapper)",
                  fontsize=12)
    fig2.tight_layout()
    out2 = Path("analysis/plots/sandbox_a_vs_b.png")
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
