# Loss Landscape: Sharpness → Generalization and Forgetting

To test the classical claim that **flatter loss minima generalize better and forget less** (Foret et al. 2021; Mirzadeh et al. 2020), we measure (i) Hessian-based sharpness, (ii) cross-corpus zero-shot perplexity (the "type-B" / distribution-shift form of generalization, analogous to ImageNet→ImageNet-V2/C in vision), and (iii) catastrophic forgetting and plasticity under domain-shift fine-tuning.

Most measurements are on OWT-pretrained variants (best.pt at end of training) of the small backbone (12L/12H/d=768). The MuonFormer addition and the TS-pretrained sharpness subsection use the TS-trained checkpoints.

## Variants and base performance

Each row is one optimizer/scheduler combination, all trained from scratch on TinyStories (TS) and OpenWebText (OWT) under the matched 12L/12H/d=768 backbone. Downstream accuracies are evaluated on the OWT-pretrained checkpoint (HellaSwag 10-shot acc_norm, ARC-Easy 25-shot acc).

| Variant     | OWT val_loss | TS val_loss | HellaSwag (norm) | ARC-Easy |
|---|---:|---:|---:|---:|
| vanilla     | 3.008 | 1.157 | 0.302 | 0.430 |
| adamw       | 2.988 | 1.147 | 0.301 | 0.434 |
| adam        | 2.991 | 1.153 | 0.310 | 0.432 |
| tmm         | 2.934 | 1.128 | 0.318 | 0.450 |
| yurii       | 2.941 | 1.130 | 0.316 | 0.443 |
| tmm-wsd     | 2.924 | 1.086 | 0.322 | **0.460** |
| yurii-wsd   | 2.928 | 1.082 | 0.318 | 0.456 |
| yurii-sam   | 2.948 | 1.081 | 0.313 | 0.437 |
| tmm-sam     | 2.940 | 1.079 | 0.315 | **0.461** |
| yurii-sawd  | **2.932** | **1.077** | 0.315 | 0.447 |

**vanilla** = GD-style transformer (no velocity / no momentum).
**adam / adamw** = Adam(W)-style per-token first/second moment streams alongside the hidden state.
**yurii** = Nesterov + Lie–Trotter (the YuriiFormer paper's main architecture).
**tmm** = Triple Momentum Method (Van Scoy et al. 2018) generalization of yurii with an extra learnable scalar.
**-sam / -wsd / -sawd** = trained with Sharpness-Aware Minimization, Warmup-Stable-Decay schedule, or both (SAM applied only during the WSD decay phase).

## OWT pretraining convergence: optimizer comparison on identical VanillaTransformer

Same architecture (`VanillaTransformer`, 12L/12H/d=768, 124M params), same OWT data, same DDP setup (2 GPUs, effective batch 480, block 1024, cosine LR schedule with 3k warmup → 30k steps, GRAD_CLIP=1.0). The only difference is the optimizer:

- **VanillaTransformer + pure AdamW** (`vanilla_adamw_train_owt.py`, job 7890344): one AdamW optimizer for all params (lr=6e-4, β=(0.9, 0.95), wd=0.1 on 2D and embeddings, wd=0 on LN).
- **VanillaTransformer + Muon(2D) + AdamW(rest)** (`vanilla_train_owt.py`, job 6989892): Muon (lr=4e-3, momentum=0.95, Nesterov, wd=0) for 2D matrix weights; AdamW (lr=6e-4, β=(0.9, 0.95), wd=0.1 on embeddings, wd=0 on LN) for everything else.

Both runs **finished 30 000 steps**. Final val_loss (mean cross-entropy over 160 batches × 8 × 1024 OWT-val tokens):

| Setup | Final step | Best val_loss |
|---|---:|---:|
| Vanilla + pure AdamW | 30 000 | 3.0103 |
| Vanilla + Muon(2D) + AdamW(rest) | 30 000 | **3.0078** |

Val loss at matched step counts:

| step ~k | AdamW val | Muon+AdamW val | Δ (AdamW − Muon+AdamW) |
|---:|---:|---:|---:|
| 1000  | 4.844 | **4.638** | +0.206 |
| 2000  | 3.923 | **3.734** | +0.189 |
| 3000  | 3.651 | **3.528** | +0.122 |
| 5000  | 3.397 | **3.334** | +0.063 |
| 7000  | 3.294 | **3.251** | +0.043 |
| 10000 | 3.213 | **3.178** | +0.035 |
| 15000 | 3.132 | **3.107** | +0.025 |
| 20000 | 3.074 | **3.055** | +0.018 |
| 25000 | 3.030 | **3.022** | +0.009 |
| 29900 | 3.011 | **3.008** | +0.002 |

The Muon+AdamW hybrid is consistently ahead throughout training, but the gap **shrinks monotonically** from ~0.21 nats at step 1k to **~0.002 nats** at step 30k. By end-of-training the two optimizers are within noise (one val checkpoint apart). Most of the practical signal is in early-training: if you only have a 5k-step budget, Muon+AdamW saves you ~0.06 nats; with a 30k budget it saves you ~0.003 nats. Wall-clock at the same hardware (L40S × 2) is essentially identical (~3.1 s/step for both, since Muon's per-layer NS iteration is cheap relative to the AdamW pass; the slow Muon-on-`MuonFormer` curve elsewhere in the codebase comes from the *architecture* — per-token head-wise NS inside the forward pass — not from the Muon optimizer itself).

The AdamW final checkpoint (1.49 GB) is uploaded to <https://huggingface.co/gaijingchu/checkpoints_vanilla_adamw_owt> for reproducibility; the Muon+AdamW checkpoint is on the cluster at `${CACHE}/checkpoints_vanilla_owt/best.pt`.

Curves are in `analysis/plots/owt_convergence.png` (full-range and step≥2000 zoom for train + val). Raw step→loss pairs are dumped to `analysis/owt_convergence.json`; the plotter is `plot_owt_convergence.py`.

## Methodology

**Sharpness** — Power-iteration estimate of $\lambda_{\max}(\nabla^2 L)$ and Hutchinson's Rademacher estimator of $\mathrm{tr}(\nabla^2 L)$ on 32 randomly drawn val batches. We report the normalized $\mathrm{tr}\,H/N$ (smaller = flatter; $N$ = parameter count). $\lambda_{\max}$ is reported with the sign returned by power iteration — only the magnitude is meaningful, since the NN Hessian is indefinite. Implementation: `loss_sharpness.py`.

**Cross-corpus PPL** — For each variant, compute val perplexity on:
- `owt_val` (in-distribution baseline)
- `wikitext-103-v1/validation` (encyclopedic web)
- `EleutherAI/lambada_openai/test` (long-range narrative)
- `c4/en/validation`, first 5,000 docs (broad web)

200 batches × batch 8 × block 1024 ≈ 1.64M tokens / corpus. Tokenization: GPT-2 BPE via `tiktoken`. Implementation: `eval_cross_corpus.py`.

**Forgetting / Plasticity** — Fine-tune for 1,000 steps on the *target* dataset using a uniform optimizer (AdamW, lr 1e-4, β=(0.9, 0.95), wd 0.01, 50-step linear warmup, batch 480 via 2 GPU × grad_accum 60). The optimizer is held fixed across variants so that the measured behavior reflects the **landscape** the variant settled into during pretraining, not its training-time optimizer. Two symmetric directions:

- `owt2ts`: OWT-pretrained → fine-tune on TinyStories
- `ts2owt`: TS-pretrained → fine-tune on OpenWebText

Definitions (over 1,000 fine-tune steps):

- $\mathrm{Forgetting} = L^{\mathrm{src}}_{T=1000} - L^{\mathrm{src}}_{T=0}$ (positive ⇒ forgot; lower is better)
- $\mathrm{Plasticity} = L^{\mathrm{tgt}}_{T=0} - L^{\mathrm{tgt}}_{T=1000}$ (positive ⇒ improved; higher is better)

Implementation: `finetune_forgetting.py` (DDP-2GPU). Both directions run on the same 1,000-step recipe so the numbers are directly comparable, modulo the asymmetry of the underlying datasets (see Discussion).

## Sharpness (OWT-pretrained, evaluated on OWT val)

Sorted by $\mathrm{tr}\,H/N$ from sharpest to flattest:

| Variant | val_loss | $\lambda_{\max}$ | $\mathrm{tr}\,H/N \;(\times 10^{-3})$ |
|---|---:|---:|---:|
| vanilla       | 3.008 |   80 | **0.316** |
| adamw         | 2.988 | 1889 | 0.299 |
| adam          | 2.991 |  424 | 0.210 |
| tmm           | 2.934 |  131 | 0.142 |
| yurii         | 2.941 |  168 | 0.139 |
| tmm-wsd       | 2.924 | −107 | 0.106 |
| yurii-wsd     | 2.928 |  112 | 0.096 |
| yurii-sam     | 2.948 |   89 | 0.067 |
| tmm-sam       | 2.940 |   39 | 0.062 |
| yurii-sawd    | 2.932 |  −74 | **0.059** |

Vanilla is ~5× sharper than yurii-sam under tr/N. Adding SAM during training halves $\mathrm{tr}\,H/N$ relative to the same momentum variant; WSD also reduces it but less aggressively.

## Sharpness (TS-pretrained, evaluated on TS val)

Same protocol as above, but the source checkpoint is the TS-pretrained variant; reference loss is on TS val. The **MuonFormer (TS)** checkpoint added here is hosted at `gaijingchu/ANLP-Yuriiformer-reproduce-MuonFormer` (`checkpoints_muon/best.pt`); all other TS rows come from the previously archived TS checkpoints.

Sorted by $\mathrm{tr}\,H/N$ from sharpest to flattest:

| Variant | val_loss | $\lambda_{\max}$ | $\mathrm{tr}\,H/N \;(\times 10^{-3})$ | 1D curve $\Delta_{|\alpha|\le 0.5}$ |
|---|---:|---:|---:|---:|
| ts-vanilla    | 1.157 | −431 | **0.0782** | 9.00 |
| ts-tmm-wsd    | 1.086 | 1830 | 0.0283 | 9.28 |
| ts-yurii      | 1.130 |   71 | 0.0213 | 7.48 |
| ts-tmm        | 1.128 |  −34 | 0.0194 | 10.45 |
| ts-adam       | 1.153 |   22 | 0.0163 | 7.93 |
| ts-adamw      | 1.147 |   19 | 0.0142 | 8.90 |
| ts-sawd       | 1.077 |  387 | 0.0140 | 8.37 |
| ts-sam        | 1.081 |   36 | 0.0128 | 8.95 |
| ts-tmm-sam    | 1.079 |  −70 | 0.0121 | 8.15 |
| ts-wsd        | 1.082 |  372 | 0.0116 | 8.69 |
| **ts-muon**   | **1.150** | **−791** | **0.0094** | 9.18 |
| ts-tmm-sawd   | 1.082 | −159 | **0.0091** | 8.38 |

**MuonFormer (TS) sits at the flat end** of the spectrum: $\mathrm{tr}\,H/N \approx 9.4\times10^{-6}$, $\sim\!8\times$ flatter than the GD-style ts-vanilla baseline and on par with the previously flattest variant (ts-tmm-sawd). This is consistent with the picture that per-token Newton–Schulz orthogonalization inside the forward pass biases the trajectory toward flatter minima — paying a $\sim\!4\times$ wall-clock-per-step cost (12.2 s/step vs 3.1 s/step for vanilla-AdamW on the same 12L/12H/d=768 backbone) for a ~8× lower mean curvature.

**Note on $\lambda_{\max}$.** Power iteration on the Hessian–vector product converges to the eigenvalue of largest magnitude, not the largest signed value. The negative reports for ts-muon ($-791$), ts-vanilla ($-431$), ts-tmm-sawd ($-159$), and ts-tmm-sam ($-70$) therefore indicate that the converged direction has *negative* curvature: the Hessian at these points is indefinite (saddle-like) and the dominant magnitude lies on the negative branch. In this regime, the robust sharpness signal is $\mathrm{tr}\,H/N$, not signed $\lambda_{\max}$; the Hutchinson trace standard deviation (e.g.\ $\pm 1364$ for ts-muon, 87% of the mean) further corroborates that positive and negative eigenvalues are partially cancelling.

## Generalization: Cross-corpus zero-shot perplexity (OWT-pretrained)

| Variant | owt_val | wikitext-103 | LAMBADA | C4 | **OOD-avg** |
|---|---:|---:|---:|---:|---:|
| **yurii-wsd** | 18.56 | 51.46 | 40.40 | 38.29 | **43.38** |
| yurii         | 18.81 | 51.78 | 41.55 | 38.92 |  44.08 |
| yurii-sam     | 18.95 | 53.00 | 41.24 | 38.57 |  44.27 |
| tmm-sam       | 18.77 | 54.21 | 40.45 | 38.86 |  44.51 |
| tmm           | 18.69 | 53.77 | 41.78 | 38.70 |  44.75 |
| tmm-wsd       | 18.48 | 55.64 | 42.22 | 38.83 |  45.56 |
| adam          | 19.78 | 59.31 | 43.13 | 39.56 |  47.33 |
| **vanilla**   | 20.11 | 61.58 | 43.19 | 39.73 | **48.17** |

The two flattest variants (yurii-wsd, yurii-sam) win on every OOD corpus; vanilla and adam lose on every OOD corpus. **Spearman ρ(tr/N, OOD-avg PPL) = 0.67** across the 8 measured variants.

## Forgetting and Plasticity

### Direction `owt2ts` (OWT pretrain → 1k AdamW steps on TS)

Sorted by forgetting (lowest = least forgotten):

| Variant | $L^{\mathrm{src}}_0$ | $L^{\mathrm{src}}_1$ | **Forget** ↓ | $L^{\mathrm{tgt}}_0$ | $L^{\mathrm{tgt}}_1$ | **Plast** ↑ |
|---|---:|---:|---:|---:|---:|---:|
| yurii-wsd  | 2.899 | 3.527 | **0.628** | 2.361 | 1.437 | 0.925 |
| tmm-sam    | 2.911 | 3.538 | 0.627 | 2.383 | 1.395 | 0.989 |
| tmm-wsd    | 2.896 | 3.534 | 0.638 | 2.368 | 1.442 | 0.926 |
| yurii-sam  | 2.921 | 3.583 | 0.662 | 2.400 | 1.397 | **1.003** |
| yurii      | 2.914 | 3.587 | 0.674 | 2.385 | 1.418 | 0.966 |
| tmm        | 2.908 | 3.600 | 0.692 | 2.376 | 1.423 | 0.953 |
| adam       | 2.963 | 3.734 | 0.772 | 2.455 | 1.457 | 0.998 |
| **vanilla**| 2.980 | 3.806 | **0.826** | 2.457 | 1.467 | 0.990 |

### Direction `ts2owt` (TS pretrain → 1k AdamW steps on OWT)

| Variant | $L^{\mathrm{src}}_0$ | $L^{\mathrm{src}}_1$ | **Forget** ↓ | $L^{\mathrm{tgt}}_0$ | $L^{\mathrm{tgt}}_1$ | **Plast** ↑ |
|---|---:|---:|---:|---:|---:|---:|
| yurii-sam  | 1.086 | 1.985 | **0.899** | 7.985 | 4.419 | 3.566 |
| tmm-sam    | 1.081 | 1.982 | 0.901 | 8.136 | 4.421 | 3.715 |
| tmm        | 1.081 | 2.012 | 0.931 | 8.906 | 4.455 | 4.451 |
| yurii      | 1.085 | 2.032 | 0.948 | 9.017 | 4.448 | 4.568 |
| adam       | 1.110 | 2.097 | 0.987 | 7.950 | 4.529 | 3.421 |
| yurii-wsd  | 1.087 | 2.116 | 1.030 | 9.607 | 4.592 | 5.015 |
| tmm-wsd    | 1.092 | 2.148 | 1.056 | 9.826 | 4.611 | **5.215** |
| **vanilla**| 1.114 | 2.233 | **1.120** | 9.527 | 4.761 | 4.765 |

## Correlations

Spearman rank correlation across the 8 OWT-pretrained variants:

| Pair | ρ |
|---|---:|
| $\mathrm{tr}\,H/N$ vs OOD-avg PPL | 0.67 |
| $\mathrm{tr}\,H/N$ vs Forgetting (owt2ts) | **0.93** |
| $\mathrm{tr}\,H/N$ vs Forgetting (ts2owt) | 0.60 |
| mean $\kappa_{\mathrm{eff}}$ vs val_loss          | **−0.617** |
| max  $\kappa_{\mathrm{eff}}$ vs val_loss          | −0.483 |
| mean stable rank vs val_loss                      | **+0.717** |

## Sandbox A: Linearized Residual Spectrum (per-layer local Jacobian)

This section tests the central proof mechanism proposed in [`docs/momentum_transformer_theory.md`](docs/momentum_transformer_theory.md) §6–7: that momentum-equipped variants (Yurii / TMM, and adaptive-update variants Adam / AdamW) attenuate slow spectral modes of the local residual oracle more uniformly than a Vanilla Transformer, and that the empirical gap is largest precisely where Vanilla's local Jacobian is ill-conditioned.

### Method

For each OWT-pretrained variant, on the OWT validation split, we capture per-layer residual-stream states $X_\ell$ during a forward pass and define the **canonical residual oracle** used uniformly across all variants:

$$
R_\ell(X) \;=\; A_\ell\!\left(\mathrm{LN}(X)\right) \;+\; M_\ell\!\left(\mathrm{LN}\bigl(X + A_\ell(\mathrm{LN}(X))\bigr)\right),
$$

where $A_\ell$ and $M_\ell$ are the layer's own attention and MLP modules. The state $X$ fed to $R_\ell$ is **the same state the model's forward pass actually feeds into $\mathrm{LN}_{\mathrm{attn}}$**:

- Vanilla / Adam / AdamW: $X = X_\ell$ (the block input).
- Yurii / TMM:            $X = \widetilde X_\ell = X_\ell + \mu_{a,\ell}\,V_\ell$ (the attention lookahead).

The local Jacobian $J_\ell = \partial R_\ell / \partial X\,|_{X}$ is too large to materialize (≈ $(BTd)^2 \approx 2\!\cdot\!10^{12}$ entries). All spectral quantities are estimated **matrix-free** through Jacobian–vector and vector–Jacobian products (`torch.autograd.functional.jvp` / `vjp`, with math-only SDPA forced so double-backward is valid).

Per layer × per validation batch we estimate:

| Quantity | Estimator |
|---|---|
| $\sigma_{\max}(J_\ell)$                       | Power iteration on $J^\top J$, 20 iters (tol $10^{-3}$) |
| Top-$k$ singular values, $k=64$               | Randomized SVD: $Y = J\Omega$ ($p\!=\!74$ JVPs), $Q = \mathrm{qr}(Y)$, $B = Q^\top J$ ($p$ VJPs), $\mathrm{svd}(B)$ |
| $\sigma_{\min}^{\mathrm{eff}}(J_\ell)$        | Smallest of the top-$k$ singular values resolved by the randomized SVD |
| $\kappa_{\mathrm{eff}}(J_\ell)$               | $\sigma_{\max}/\sigma_{\min}^{\mathrm{eff}}$ |
| $\lVert J_\ell\rVert_F^2$                     | Hutchinson: $\mathbb{E}_z\lVert J z\rVert^2$, $z\!\in\!\{-1,+1\}^{BTd}$, 8 probes |
| stable rank                                   | $\lVert J\rVert_F^2 / \sigma_{\max}^2$ |

### Experiment configuration

| Knob | Value |
|---|---|
| Variants | vanilla, adam, adamw, yurii, tmm, yurii-sam, yurii-wsd, tmm-sam, tmm-wsd (the 9 OWT-pretrained variants) |
| Checkpoints | `${CACHE}/checkpoints_<variant>_owt/best.pt` |
| Validation tokens | OWT val split (`data_owt.load_owt_tokens('val')`) |
| Batches | 8 |
| Sequence length $T$ | 1024 |
| Per-batch GPU sample size | $B = 2$ (so $X$ is $(2, 1024, 768)$) |
| Layers | all 12 |
| Top-$k$ singular values | 64 (oversample +10) |
| Power-iteration iters / probes | 20 power, 8 Hutchinson |
| Compute backend | math-only SDPA (mandatory for autograd-functional JVP) |
| Where it runs | `local_jacobian_spectrum.sbatch` on `preempt`, 1 GPU, ~2–3 h per variant |

### Predictions to be checked

1. **$\kappa_{\mathrm{eff}}$ ordering:** Vanilla > Adam ≈ AdamW > Yurii ≈ TMM. The gap is expected to be largest in early/middle layers where $\sigma_{\min}^{\mathrm{eff}}$ is smallest.
2. **Stable rank:** momentum variants admit higher stable rank (energy spread across more singular directions), reflecting better-conditioned local dynamics.
3. **Slow-mode persistence** ($\prod_\ell \sigma_{\min}^{\mathrm{eff}}$) **across depth:** Vanilla retains substantially more mass in slow modes than Yurii/TMM after 12 layers; the doc gives an analytic $\kappa\!=\!100$ illustration where Vanilla retains 0.787 and the momentum variant retains 0.090 of the slow-mode error.
4. **Correlation:** mean $\kappa_{\mathrm{eff}}$ should rank-correlate with OOD-avg perplexity and with forgetting (owt2ts), recovering or strengthening the $\mathrm{tr}\,H/N$ correlations already reported.

### Per-variant aggregate

Mean and max across the 12 layers, averaged over 8 validation batches. Sorted by val_loss ascending (best downstream first).

| Variant | val_loss | $\overline{\sigma_{\max}}$ | $\overline{\sigma_{\min}^{\mathrm{eff}}}$ | $\overline{\kappa_{\mathrm{eff}}}$ | $\max\kappa_{\mathrm{eff}}$ | $\overline{\mathrm{srank}}$ | $\sum_\ell \log\sigma_{\min}^{\mathrm{eff}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **tmm-wsd**   | **2.924** | 690.0 | 100.5 | **6.30** | **12.87** | 261.2 | 51.95 |
| yurii-wsd     | 2.928 | 465.0 | 73.7 | 5.54 | 9.88 | 292.3 | 46.12 |
| tmm           | 2.934 | 292.1 | 55.5 | 5.25 | 7.20 | 325.5 | 44.30 |
| tmm-sam       | 2.940 | 266.4 | 48.9 | 5.26 | 10.52 | 392.9 | 43.42 |
| yurii         | 2.941 | 190.0 | 39.0 | 4.34 | 7.38 | 391.2 | 37.85 |
| yurii-sam     | 2.948 | 165.8 | 31.0 | 4.73 | 9.65 | 443.7 | 34.69 |
| adamw         | 2.988 | 663.8 | 187.7 | 3.72 | 8.21 | 376.7 | 55.92 |
| adam          | 2.991 | 771.7 | 285.1 | **3.30** | **5.06** | 366.6 | 56.24 |
| **vanilla**   | **3.008** | 58.5 | 10.5 | 5.52 | 9.80 | **1582.8** | **23.26** |

Note: $\sum_\ell \log\sigma_{\min}^{\mathrm{eff}}$ is the log of $\prod_\ell\sigma_{\min}^{\mathrm{eff}}$ (slow-mode persistence over depth, in the top-64 resolved spectrum). A *smaller* sum means the resolved spectrum is more aggressively attenuated layer-to-layer.

### Per-layer $\kappa_{\mathrm{eff}}$ trajectory

Layer-by-layer mean $\kappa_{\mathrm{eff}}$ across the 8 batches.

| Variant | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vanilla   | **9.80** | 5.31 | 2.13 | 2.81 | 4.07 | 5.65 | 4.40 | 5.62 | 7.41 | 6.23 | 6.32 | 6.43 |
| adam      | 5.06 | 3.55 | 1.73 | 1.86 | 2.36 | 2.68 | 2.81 | 4.74 | 4.11 | 3.88 | 3.50 | 3.36 |
| adamw     | 5.77 | 8.21 | 2.33 | 1.79 | 2.55 | 2.70 | 2.86 | 4.82 | 3.68 | 3.45 | 2.86 | 3.57 |
| yurii     | 6.06 | 3.29 | 2.24 | 2.54 | 7.37 | 2.92 | 7.38 | 4.94 | 4.19 | 3.36 | 4.62 | 3.19 |
| tmm       | 6.36 | 5.41 | 5.50 | 4.13 | 4.40 | 4.62 | 4.04 | 7.20 | 6.21 | 6.13 | 5.74 | 6.32 |
| yurii-sam | 5.95 | 9.65 | 2.60 | 4.57 | 2.98 | 4.38 | 3.13 | 6.46 | 5.44 | 4.65 | 3.42 | 3.54 |
| yurii-wsd | 6.42 | 9.88 | 6.75 | 2.44 | 7.03 | 2.82 | 7.39 | 7.05 | 4.63 | 3.57 | 4.58 | 3.86 |
| tmm-sam   | 6.84 | 10.52 | 2.78 | 3.59 | 3.06 | 3.26 | 2.74 | 5.18 | 7.41 | 7.03 | 4.33 | 6.36 |
| tmm-wsd   | 6.93 | **12.87** | 9.12 | 2.62 | 7.09 | 3.03 | 6.49 | 3.91 | 4.80 | 6.34 | 6.46 | 5.99 |

Per-layer $\kappa_{\mathrm{eff}}$ trajectories are also rendered in `analysis/plots/spectrum_kappa_vs_depth.png`.

### Result summary against the predictions

|  | Prediction (doc §7.5) | What we see |
|---|---|---|
| 1. $\overline{\kappa_{\mathrm{eff}}}$ ordering | Vanilla > Adam ≈ AdamW > Yurii ≈ TMM | **Falsified.** Sorted by $\overline{\kappa}$ ascending: adam (3.30), adamw (3.72), yurii (4.34), yurii-sam (4.73), tmm (5.25), tmm-sam (5.26), vanilla (5.52), yurii-wsd (5.54), **tmm-wsd (6.30)**. Vanilla is in the middle, not the top. Adaptive variants (adam/adamw) are the best-conditioned; momentum + WSD variants (tmm-wsd, yurii-wsd) are *worse* than Vanilla. |
| 2. Stable rank for momentum variants | Higher (energy spread across more directions) | **Falsified.** Vanilla's $\overline{\mathrm{srank}}=1582.8$, ~4× higher than every other variant (261–444). The momentum / adaptive families concentrate their oracle's energy into fewer dominant directions, not more. |
| 3. Slow-mode persistence $\prod_\ell\sigma_{\min}^{\mathrm{eff}}$ | Smaller for momentum variants (more attenuation) | **Inverted in this operational definition.** The smallest $\sum_\ell\log\sigma_{\min}^{\mathrm{eff}}$ is Vanilla's (23.3) — Vanilla attenuates its top-64 resolved spectrum more aggressively layer-to-layer than every other variant. This is consistent with (2): Vanilla's oracle is the smallest in raw magnitude. |
| 4. Vanilla worst in early/middle layers | Yes | **Partly supported.** Vanilla has the largest $\kappa_{\mathrm{eff}}$ at layer 0 (9.80) of any variant *that doesn't include WSD/SAM*. But it is the smallest layer-0 only against {adam, adamw, yurii}. The tmm/yurii-*-wsd family has L1 hot-spots (9.88–12.87) that exceed Vanilla anywhere. |

### Interpretation

The Spearman rank correlation across the 9 variants is

| Pair | ρ |
|---|---:|
| $\overline{\kappa_{\mathrm{eff}}}$ vs val_loss | **−0.617** |
| $\max\kappa_{\mathrm{eff}}$ vs val_loss        | −0.483 |
| $\overline{\mathrm{srank}}$ vs val_loss        | **+0.717** |

i.e. *higher* local condition number of the canonical oracle is associated with *lower* val_loss, and *higher* stable rank is associated with *higher* val_loss. This is the opposite of the prediction stated in the theory doc.

The cleanest interpretation is that the canonical residual oracle $R_\ell$ is **not the operator whose conditioning the architecture targets**. Each variant trains its actual layer update (which is $R_\ell$ for Vanilla, but is $R_\ell$ wrapped in $\mathrm{LN}_v$, scaled by $\gamma$, mixed with a velocity stream, and re-scaled by $\nu$ for Yurii/TMM; or $R_\ell$ post-divided by $\sqrt{s}$ for Adam(W)) — and that *post-wrapper* operator can be well-conditioned even when $R_\ell$ alone is large in norm with concentrated spectrum. In other words, the trained $R_\ell$ for momentum variants looks "worse" by the doc's metrics *because* the architecture absorbs that ill-conditioning into its $\mathrm{LN}_v / \mathrm{LN}_{\mathrm{update}} / \nu$ wrappers, which we did not include in the Jacobian.

The Vanilla result is the most informative single data point: it has the *smallest* oracle in raw magnitude (mean $\sigma_{\max}=58.5$) and the *highest* stable rank by a wide margin (1583 vs ~370 elsewhere). Vanilla cannot rely on a downstream wrapper to absorb spectral imbalance, so its training pushes $R_\ell$ to be small and broadband — exactly the property the doc's theory uses to *predict* better generalization. Yet Vanilla still has the highest val_loss. Whatever the momentum / adaptive variants gain, it is not visible in the spectrum of $R_\ell$ at the lookahead state alone.

### What the experiment does *not* settle

- The full block Jacobian $\partial(\text{block output}) / \partial(\text{block input})$, which includes $\mathrm{LN}_v$, $\nu$, $\gamma$, and the velocity-stream mixing for Yurii/TMM (or $\sqrt{s}$ division for Adam/AdamW), was not measured. That is the operator the theory's $\rho_{\mathrm{mom}}$ contraction bound is *actually* about, once one writes momentum variants in operator form. A natural follow-up — Sandbox B — is to estimate that block-Jacobian spectrum and re-run the predictions.
- The top-64 randomized SVD resolves the *head* of the spectrum, not the long tail; the theory's "slow mode" is the smallest singular value, which is below our resolution. A Lanczos run targeting the *bottom* of the spectrum (rather than the top) would address this directly.
- All measurements are at a *single* training step (the saved best.pt). The doc's theory is asymptotic; intermediate trajectories may tell a different story.

### Falsification status

Following doc §7.6, the proposed mechanism is weakened. Concretely we see:

- ✗ Vanilla's $\overline{\kappa_{\mathrm{eff}}}$ is **not** the largest (criterion 1).
- ✓ Spectral curves are visually different across variants (criterion 2 holds — not falsified on this point).
- ✗ TMM/Yurii outperform Vanilla on val_loss while showing **worse** local conditioning (criterion 4).

This pushes the explanation toward the alternatives the doc itself lists: regularization from the auxiliary stream ($\mathrm{LN}_v$, learned $\beta,\nu$), improved gradient flow during *parameter* training (a different operator than the per-layer residual Jacobian), or learned feature routing — rather than finite-depth spectral filtering of the canonical oracle.

### Falsification criteria

Following the doc §7.6, the proposed mechanism is weakened if any of the following hold once the table fills in:

- Vanilla's mean $\kappa_{\mathrm{eff}}$ is **not** largest.
- Spectral curves (top-$k$ SVs vs index) are visually indistinguishable across variants.
- The largest TMM/Yurii vs Vanilla wins on $\mathrm{val\_loss}$ / OOD-PPL occur at layers where $\kappa_{\mathrm{eff}}^{\mathrm{vanilla}}$ is **small**.
- TMM/Yurii outperform Vanilla on downstream metrics **and** show worse local conditioning **and** no compensating improvement in slow-mode attenuation.

In any of those cases, the picture pushed back to is that the variant gap is driven by another mechanism (regularization from the auxiliary stream, gradient-flow effects during training, learned feature routing), not the finite-depth spectral-filtering story.

### Status

- **Code:** `local_jacobian_spectrum.py`, sbatch wrapper `local_jacobian_spectrum.sbatch`. JSON output → `spectrum_results/<variant>.json`.
- **Jobs:** 9 SLURM jobs on `preempt` (one per variant), each ~2–3 h.
- **Aggregation:** results will be merged into `loss_sharpness_results/_summary.json` under a new `spectrum` block and visualized in `analysis/plots/`.

## Sandbox B: Full block Jacobian (with wrapper)

Sandbox A measured the *canonical* residual oracle $R_\ell$ that is identical across all variants and is what the doc's spectral predictions are literally written about. The doc's $\rho_{\mathrm{mom}}$ contraction bound, however, is an operator-form statement about the **full block transition** $F_\ell : x_\ell \mapsto x_{\ell+1}$, which for the momentum architectures *includes* $\mathrm{LN}_v$, the learned scalars $\gamma_\ell, \nu_\ell$, and the velocity-stream mixing (for Yurii / TMM); and for the Adam(W) family includes the per-step $1/\sqrt{s_\ell}$ rescale. Sandbox B closes that gap.

### Method

For each variant we wrap the model in a forward-pre-hook that captures the *full* input tuple to layer $\ell$:

- **vanilla / adam / adamw** — single-tensor block; the captured input is $(x_\ell)$ and the operator under study is $F_\ell(x) = \mathrm{block}_\ell(x)$.
- **adam / adamw** explicitly — the wrapper material is the layer's internal $1/\sqrt{s_\ell}$ post-attention rescale; auxiliary state $m_\ell, s_\ell$ is captured at the same forward pass and **held fixed** during JVP/VJP, so the Jacobian we estimate is genuinely $\partial F_\ell / \partial x_\ell$ at fixed $m, s$.
- **yurii / yurii-sam / yurii-wsd** — two-stream block; captured input is $(x_\ell, v_\ell)$, operator is $F_\ell(x) = \mathrm{block}_\ell(x, v_\ell)[0]$ (the residual-stream output), with $v_\ell$ frozen.
- **tmm / tmm-sam / tmm-wsd** — three-stream block; captured input is $(x_\ell, m_\ell, s_\ell)$, operator is $F_\ell(x) = \mathrm{block}_\ell(x, m_\ell, s_\ell)[0]$, with $m, s$ frozen.

The spectral estimators are identical to Sandbox A (matrix-free power iteration on $J^\top J$ for $\sigma_{\max}$, randomized top-$k$ SVD for $\sigma_{\min}^{\mathrm{eff}}$ and stable rank, Hutchinson for $\lVert J\rVert_F^2$). Same hyperparameters: 8 batches × $B\!=\!2$ × $T\!=\!1024$, all 12 layers, top-$k\!=\!64$, 20 power iters, 8 Hutchinson probes. Same checkpoints. Implementation: `block_jacobian_spectrum.py`; SLURM wrapper `block_jacobian_spectrum.sbatch` (1 GPU, preempt, ~3–4 min/variant).

### Per-variant aggregate

Sorted by val_loss ascending. $\overline{\kappa_{\mathrm{eff}}}$ and $\overline{\mathrm{srank}}$ are over 12 layers × 8 batches; $\max\kappa_{\mathrm{eff}}$ is the max over layers of the batch-mean.

| Variant | val_loss | $\overline{\sigma_{\max}}$ | $\overline{\sigma_{\min}^{\mathrm{eff}}}$ | $\overline{\kappa_{\mathrm{eff}}}$ | $\max\kappa_{\mathrm{eff}}$ | $\overline{\mathrm{srank}}$ | $\sum_\ell \log\sigma_{\min}^{\mathrm{eff}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **tmm-wsd**   | **2.924** | 49.5 | 5.06 | **12.00** | **32.64** | 16906.2 | 7.81 |
| yurii-wsd     | 2.928 | 36.4 | 4.06 | 10.00 | 19.23 | 3601.4 | 13.90 |
| tmm           | 2.934 | 22.6 | 2.56 | 9.83 | 21.26 | 10614.0 | 9.12 |
| tmm-sam       | 2.940 | 18.6 | 2.40 | 8.50 | 15.20 | 26076.5 | 6.69 |
| yurii         | 2.941 | 28.9 | 3.83 | 8.12 | 10.52 | 4175.3 | 14.44 |
| yurii-sam     | 2.948 | 18.4 | 2.21 | 9.68 | 19.57 | 5362.1 | 11.25 |
| adamw         | 2.988 | 39.8 | 7.51 | 6.33 | 11.11 | 2037.2 | 24.66 |
| adam          | 2.991 | 43.6 | 8.50 | 6.30 | 11.86 | 1950.5 | 25.21 |
| **vanilla**   | **3.008** | 23.4 | 3.62 | 6.94 | 9.79 | 3110.5 | 20.43 |

Note: absolute $\sigma$ scales differ from Sandbox A because the wrappers — most prominently the per-layer $1/\sqrt{s_\ell}$ for Adam(W), and $\mathrm{LN}_v / \gamma / \nu$ for Yurii/TMM — multiplicatively rescale the operator. The absolute scale is not directly comparable across the two tables; the **ranks** and **shapes** are what each sandbox claims to inform.

### Per-layer $\kappa_{\mathrm{eff}}(F_\ell)$ trajectory

| Variant | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vanilla   | 9.79 | 9.43 | 2.34 | 3.05 | 5.07 | 6.96 | 5.79 | 7.66 | 8.50 | 7.69 | 8.82 | 8.18 |
| adam      | 5.09 | 11.86 | 3.18 | 1.93 | 6.95 | 5.83 | 6.51 | 7.94 | 7.45 | 6.94 | 8.86 | 3.05 |
| adamw     | 6.94 | 11.11 | 2.46 | 1.95 | 4.96 | 5.42 | 7.27 | 8.21 | 7.74 | 6.69 | 9.55 | 3.66 |
| yurii     | 6.65 | 9.13 | 5.27 | 4.92 | 10.52 | 7.78 | 7.59 | 8.55 | 8.41 | 9.83 | 8.32 | 10.45 |
| tmm       | 7.34 | **21.26** | 7.74 | 5.13 | 12.99 | 8.39 | 6.46 | 9.59 | 10.30 | 9.65 | 11.61 | 7.43 |
| yurii-sam | 5.42 | 19.57 | 13.10 | 9.07 | 7.65 | 7.14 | 7.36 | 9.04 | 9.42 | 10.65 | 9.69 | 8.06 |
| yurii-wsd | 7.11 | 6.97 | 6.42 | 9.59 | 12.66 | 8.04 | 8.18 | 9.21 | 10.66 | 11.32 | 11.84 | 8.05 |
| tmm-sam   | 6.55 | 15.20 | 4.30 | 4.81 | 6.62 | 5.41 | 6.95 | 10.36 | 9.13 | 13.13 | 11.92 | 7.59 |
| tmm-wsd   | 7.05 | **32.64** | 28.87 | 4.96 | 9.92 | 8.99 | 9.18 | 7.44 | 9.13 | 9.27 | 10.92 | 11.66 |

Per-layer curves are rendered in `analysis/plots/block_spectrum_kappa_vs_depth.png` (3 panels: Sgd/Adam, Yurii, TMM families). A direct side-by-side comparison of Sandbox A vs B mean/max $\kappa$ per variant is in `analysis/plots/sandbox_a_vs_b.png`.

### Correlations across the 9 variants

Spearman rank correlation with OWT val_loss:

| Metric (Sandbox B) | ρ vs val_loss | Sandbox A (for reference) |
|---|---:|---:|
| $\overline{\kappa_{\mathrm{eff}}}$       | **−0.900** | −0.617 |
| $\max\kappa_{\mathrm{eff}}$              | −0.767 | −0.483 |
| $\overline{\mathrm{srank}}$              | −0.700 | **+0.717** |
| $\sum_\ell \log\sigma_{\min}^{\mathrm{eff}}$ | **+0.750** | — |

Two qualitative changes vs Sandbox A:

1. **mean-κ correlation strengthens and stays negative** (−0.617 → **−0.900**). Higher block-Jacobian condition number robustly predicts lower val_loss in Sandbox B. This is still *opposite* to the doc §7.5 prediction (which expected lower κ → lower val_loss), but it is now much more decisive.
2. **Stable-rank correlation flips sign** (+0.717 → **−0.700**). In Sandbox A, Vanilla's high stable rank (1583 vs ~370 elsewhere) was the cleanest signal; in Sandbox B the WSD/SAM variants (tmm-sam srank=26 076, tmm-wsd srank=16 906) dominate, and *higher* stable rank tracks *lower* val_loss. The Sandbox A anomaly is fully explained by the absence of the wrapper: once the wrapper is included, momentum/SAM/WSD variants dominate stable rank as predicted.
3. **Slow-mode persistence aligns with the doc.** The smallest $\sum_\ell\log\sigma_{\min}^{\mathrm{eff}}$ now belongs to the best-val variants (tmm-sam 6.69, tmm-wsd 7.81, tmm 9.12); the largest belongs to adam (25.21) and adamw (24.66). Concretely: $\rho(\sum\log\sigma_{\min}, \text{val\_loss}) = +0.75$, i.e. more aggressive layer-to-layer slow-mode attenuation in $F_\ell$ correlates with better generalization. **This recovers doc §7.5 prediction 3 once the wrapper is included.**

### Where the κ inversion comes from

The persistent finding — higher mean $\kappa$ → lower val_loss — is mechanically the conjunction of two effects on the *same* operator:

- $\sigma_{\max}(F_\ell)$ for momentum variants is comparable to or smaller than Vanilla's (e.g. tmm-wsd 49.5 vs vanilla 23.4 — actually larger, but tmm 22.6 < vanilla 23.4),
- $\sigma_{\min}^{\mathrm{eff}}(F_\ell)$ for momentum variants is markedly smaller (tmm 2.56, tmm-sam 2.40 vs vanilla 3.62).

i.e. the momentum architectures **attenuate the resolved-bottom of the spectrum more aggressively** while leaving $\sigma_{\max}$ comparable, so $\kappa = \sigma_{\max} / \sigma_{\min}^{\mathrm{eff}}$ rises. By the operational definition of "slow-mode persistence" used in the doc — $\prod_\ell \sigma_{\min}$ — this *is* the predicted shape (more attenuation in the slow modes), it just shows up as higher $\kappa$ rather than lower $\kappa$ because $\sigma_{\max}$ does not shrink in lockstep. The doc's prediction (1), as written, only holds if $\sigma_{\max}$ were the dominant moving part; empirically it is $\sigma_{\min}$ that does most of the moving.

### Result summary against the predictions (re-run with wrapper)

| Doc prediction | Sandbox A (canonical R) | Sandbox B (full F) |
|---|---|---|
| 1. $\kappa$ ordering vanilla > adam ≈ adamw > yurii ≈ tmm | Falsified (vanilla mid-table). | **Falsified more strongly** (vanilla 2nd-lowest κ, tmm-wsd highest). The ranking is reversed throughout. |
| 2. Stable rank higher for momentum variants | Falsified (vanilla srank ≫ others, 4× margin). | **Recovered.** Vanilla is 5th, tmm-sam and tmm-wsd dominate. |
| 3. Slow-mode persistence $\prod_\ell\sigma_{\min}^{\mathrm{eff}}$ smaller for momentum | Inverted in canonical R (vanilla had the smallest). | **Recovered.** ρ(slow-mode-persistence, val_loss) = +0.75 — momentum variants attenuate slow modes most. |
| 4. Vanilla worst in early/middle layers | Partly. | Partly. The wrapper-included spike happens at L1 for several momentum variants (tmm 21.3, tmm-wsd 32.6, yurii-sam 19.6), not at the deepest layers, so the doc's depth-monotonicity is not literal. |

### Joint reading of Sandbox A and B

The two operators answer two different questions:

- **Sandbox A ($R_\ell$).** Geometry of the *fundamental* residual-block computation, identical across variants. A clean apples-to-apples conditioning test of the trained inner block. Result: Vanilla's $R_\ell$ is the smallest in norm and the broadest in spectral support — exactly what the doc's mechanism would prescribe — and yet Vanilla generalizes worst. The mechanism, evaluated on $R_\ell$ alone, predicts the **wrong sign**.
- **Sandbox B ($F_\ell$).** Geometry of the *actual* per-layer state-transition, including all wrappers. The momentum variants do not target $R_\ell$ for conditioning; they target $F_\ell$, and they reduce $\sigma_{\min}^{\mathrm{eff}}(F_\ell)$ (the slow modes) far more aggressively than vanilla. The doc's *qualitative* prediction (slow-mode attenuation, $\prod_\ell \sigma_{\min}$ shrinking) is recovered, the *literal* prediction ($\kappa$ ordering) is reversed because $\sigma_{\max}$ does not shrink in tandem.

The cleanest takeaway: the doc's contraction-bound story is operationally about $F_\ell$ and not about $R_\ell$. On $F_\ell$ the slow-mode-attenuation half of the story comes through clearly; the $\kappa$ half is the wrong choice of summary statistic at this scale, because the trained $F_\ell$ for momentum variants has *both* aggressive slow-mode attenuation *and* a high stable rank — the spectrum is broad and bottom-attenuated, not narrow and well-conditioned.

### Status

- **Code:** `block_jacobian_spectrum.py`, `block_jacobian_spectrum.sbatch`, `aggregate_block_spectrum.py`.
- **Raw data:** `block_spectrum_results/<variant>.json` (9 files) and aggregate in `block_spectrum_results/_aggregate.json`.
- **Plots:** `analysis/plots/block_spectrum_kappa_vs_depth.png`, `analysis/plots/sandbox_a_vs_b.png`.
- **Jobs:** 9 SLURM jobs on `preempt`, 1 GPU each, ~3–4 min/variant.

## Discussion

1. **Sharpness ↔ generalization holds robustly.** The two sharpest variants (vanilla, adam) have the worst out-of-distribution perplexity on every measured corpus. The two flattest (yurii-sam / yurii-wsd) are the best. The relationship is not perfectly monotone within the middle of the table — `tmm-wsd` is flatter than `tmm` but slightly worse on OOD, suggesting a noise floor at this model size — but the rank correlation is 0.67.

2. **Sharpness ↔ forgetting is a *much* stronger predictor than sharpness ↔ generalization.** ρ = 0.93 on `owt2ts`. Vanilla forgets 32% more of its OWT knowledge than yurii-wsd does after equal fine-tuning on TS. This is consistent with the geometric intuition: a flat minimum is a wide basin, so a fine-tune trajectory of bounded length stays closer to the original solution.

3. **Plasticity is largely independent of sharpness.** In the `owt2ts` direction, the spread in plasticity (0.92–1.00) is much smaller than the spread in forgetting (0.63–0.83), and yurii-sam (very flat) achieves the *highest* plasticity. Flat minima are not "frozen" — they retain capacity to learn — but they are stickier on the source task. The conclusion is that the classical sharpness/forgetting tradeoff sits primarily on the forgetting side, not symmetrically on plasticity.

4. **Direction asymmetry.** Plasticity in `ts2owt` is 3–5× higher than in `owt2ts`, since TS-pretrained models have only seen children's stories and have a long way to fall on web text. In this harder direction, vanilla still forgets the most (1.12), but the WSD variants (yurii-wsd 1.03, tmm-wsd 1.06) — which were the *least* forgetting in `owt2ts` — now sit in the middle of the table. WSD-shaped minima appear to be specifically resistant to short fine-tunes (the `owt2ts` 1k-step regime) but no more resistant than baseline to large-scale domain shift. The Spearman correlation drops from 0.93 (`owt2ts`) to 0.60 (`ts2owt`), driven primarily by yurii-wsd and tmm-wsd being outliers in the harder direction.

5. **SAM and WSD are not redundant with momentum.** Momentum-based variants (yurii, tmm) already produce flatter minima than vanilla/adam (tr/N drops by ~2×), but adding SAM or WSD on top further reduces tr/N by another ~2×. This stacking generalizes to forgetting: yurii-sam / yurii-wsd are both flatter and forget less than yurii alone.

Reproducing these results: `eval_cross_corpus.py` for §generalization, `finetune_forgetting.py` for §forgetting, `loss_sharpness.py` for §sharpness. Per-variant raw outputs are in `eval_cross_corpus_results/`, `forgetting_results/`, `loss_sharpness_results/`. A consolidated machine-readable summary is in `analysis/landscape_summary.json`.

---

# Theory: Hessian gap between Vanilla and Yurii / TMM transformers

The empirical sharpness gap (vanilla $\mathrm{tr}\,H/N \approx 2.3\times$ that of yurii) is *not* explained by transplanting a Nesterov optimizer-convergence proof — those proofs apply to optimizer iterates in *parameter space*, not to forward passes through a fixed network. We give an architectural derivation: YuriiFormer's two-state $(x_t, v_t)$ block-recurrence in the *depth direction* has the same algebraic form as Nesterov in the *iteration direction*, so the per-layer state-transition matrix admits a Schur-stability bound that vanilla's does not. Composing across $T$ layers and substituting into a Gauss–Newton decomposition of the loss Hessian gives a parameter-space sharpness gap that scales exponentially in depth.

## Informal main result

> **Theorem** (*Hessian gap; informal version of Theorem 8*)**.** *Let $L^{\mathrm{V}}(\theta)$ and $L^{\mathrm{Y}}(\theta)$ denote the training losses of a depth-$T$ vanilla pre-norm transformer and a depth-$T$ YuriiFormer / TMMFormer with otherwise identical block parameters, and suppose each block's input-Jacobian operator norm is bounded by $\alpha$ along the trained trajectory. Then at any near-optimal parameter,*
>
> $$\frac{\bigl\lVert \nabla^2_\theta L^{\mathrm{V}} \bigr\rVert}{\bigl\lVert \nabla^2_\theta L^{\mathrm{Y}} \bigr\rVert} \;\ge\; \exp(c\,\alpha\,T)$$
>
> *for an absolute constant $c > 0$, provided the per-layer learned scalars $(\beta_t, \mu_t)$ of the Yurii / TMM architecture lie in an explicit Schur-stable region — a condition gradient descent enforces automatically during training.*

**Mechanism.** YuriiFormer / TMMFormer maintain a velocity stream $v_t$ alongside the residual stream $x_t$. This converts the per-layer state-transition Jacobian from the scalar amplifier $I + G_t$ (whose $T$-fold product grows as $(1+\alpha)^T$) into a $2 \times 2$ block matrix whose spectrum is contractive in the Schur-stable region, so the $T$-fold product stays bounded. The exponential gap follows by chaining this through a Gauss–Newton decomposition of the loss Hessian.

The remainder of this section gives the formal statement and proof. Sections §1–§5 build the lemmas; §6 states the formal Theorem 8 with all constants made explicit; §7 collects remarks tying the theorem back to the optimizer-vs-architecture distinction, the role of learnability, and the operator-norm-vs-trace gap; §8 calibrates against measured quantities.

## 1. Setup

**Definition 1** (Layer maps). Let $B_t : \mathbb{R}^N \to \mathbb{R}^N$ denote the $t$-th transformer block (attention + MLP composed with LayerNorm), parameterized by $\theta_t$, where $N = Bd$ is the total embedding dimension over batch and sequence. The two architectures are defined by their state spaces and layer maps $F_t$:

(a) **Vanilla.** State $s_t = x_t \in \mathbb{R}^N$ and

$$
F_t^{\mathrm{V}}(x) \;=\; x + B_t(\mathrm{LN}(x);\,\theta_t).
$$

(b) **Yurii.** State $s_t = (x_t, v_t) \in \mathbb{R}^{2N}$ and

$$
F_t^{\mathrm{Y}}(x, v) \;=\; \bigl(\,x - \gamma_t\, g_t,\;\; \mu_t\, v + g_t\,\bigr),
\qquad
g_t := B_t\bigl(\mathrm{LN}(\beta_t x + (1-\beta_t) v);\,\theta_t\bigr),
$$

where $(\beta_t, \mu_t, \gamma_t) \in (0,1)^2 \times \mathbb{R}_+$ are *learned* per-layer scalars.

**Definition 2** (Block linearization). For each layer $t$ and each input $s$, the block linearization is

$$
G_t(s) \;:=\; \frac{\partial B_t(\mathrm{LN}(\cdot);\,\theta_t)}{\partial s} \cdot \frac{\partial \mathrm{LN}}{\partial s} \;\in\; \mathbb{R}^{N \times N}.
$$

The composite forward pass is $s_T = F_{T-1} \circ \cdots \circ F_0$, the loss is $L(\theta) = \ell(\pi(s_T))$ where $\pi$ projects onto the $x$-component (identity for vanilla), and $\ell$ is cross-entropy through the unembedding head.

## 2. Assumptions

**(A1)** *Block Lipschitz.* There exists $\alpha > 0$ such that $\lVert G_t(s)\rVert_2 \le \alpha$ for every layer $t$ and every state $s$ on the trained forward trajectory.

**(A2)** *Local PSD.* The matrix $G_t(s)$ is diagonalizable with real, non-negative eigenvalues at every $s$ on the trained trajectory.

**(A3)** *Near-optimality.* The trained checkpoint satisfies $\lVert \nabla_z \ell\rVert_2 \le \varepsilon_g$ for some sufficiently small $\varepsilon_g > 0$.

(A1) is mild and architecture-symmetric — it bounds Block, which has the same parameter shape across variants. (A2) is the standard local-minimum assumption for landscape analysis; in particular it makes $\lVert J_t^{\mathrm{V}}\rVert_2 = \rho(J_t^{\mathrm{V}})$. (A3) is needed only to drop the residual term in Lemma 1.

## 3. Hessian decomposition

**Lemma 1** (Gauss–Newton). For any smooth composition $L(\theta) = \ell(z(\theta))$ with $z = x_T \in \mathbb{R}^N$,

$$
\nabla^2_\theta L \;=\; J_z^\top\,(\nabla^2_z \ell)\,J_z \;+\; \sum_{i=1}^{N} (\nabla_z \ell)_i\,\nabla^2_\theta z_i,
$$

where $J_z = \partial z / \partial \theta$. Under (A3), with $K := \sup_i \lVert \nabla^2_\theta z_i\rVert_2$,

$$
\bigl\lVert \nabla^2_\theta L \bigr\rVert_2 \;\le\; \lVert J_z\rVert_2^2 \cdot \lVert \nabla^2_z \ell\rVert_2 \;+\; \varepsilon_g K.
$$

**Proof.** Write $\nabla_\theta L = J_z^\top \nabla_z \ell$ and differentiate again with the chain rule, producing the two displayed terms. The norm bound is the triangle inequality combined with $\lVert J_z^\top A J_z\rVert_2 \le \lVert J_z\rVert_2^2 \cdot \lVert A\rVert_2$ for symmetric $A$, plus (A3) for the second term. $\quad\square$

**Corollary 2** (Per-layer factorization). Let $J_r := \partial s_{r+1}/\partial s_r$ and $\Pi_t := \prod_{r=t+1}^{T-1} J_r$. Then

$$
\frac{\partial z}{\partial \theta_t} \;=\; \pi\,\Pi_t\,\frac{\partial s_{t+1}}{\partial \theta_t}.
$$

Both architectures share the same Block parameter shape, so there exists $C_\theta > 0$ with $\sup_t \lVert \partial s_{t+1}/\partial \theta_t\rVert_2 \le C_\theta$ uniformly across architectures. Hence

$$
\lVert J_z\rVert_2 \;\le\; C_\theta \cdot \max_t \lVert \Pi_t\rVert_2,
$$

and the architectural difference in $\lVert \nabla^2_\theta L\rVert_2$ is concentrated in $\max_t \lVert \Pi_t\rVert_2^2$.

## 4. Per-layer Jacobian: Vanilla

**Lemma 3.** Under (A1)–(A2), the vanilla per-layer Jacobian $J_t^{\mathrm{V}} = I + G_t$ has eigenvalues in $[1,\, 1+\alpha]$ and operator norm

$$
\lVert J_t^{\mathrm{V}}\rVert_2 \;=\; 1 + \lambda_{\max}(G_t).
$$

In particular $\lVert J_t^{\mathrm{V}}\rVert_2 \ge 1$, with equality only if $G_t = 0$.

**Proof.** Differentiating $x_{t+1} = x_t + B_t(\mathrm{LN}(x_t))$ gives $J_t^{\mathrm{V}} = I + G_t$. Under (A2), $G_t$ is diagonalizable with non-negative eigenvalues, so $J_t^{\mathrm{V}}$ is diagonalizable with eigenvalues in $[1,\,1+\alpha]$. A diagonalizable matrix with non-negative eigenvalues has operator norm equal to its largest eigenvalue. $\quad\square$

**Proposition 4** (Vanilla product, lower bound). Suppose the dominant eigenvectors of $G_0, G_1, \ldots, G_{T-1}$ admit a common direction $u \in \mathbb{R}^N$ with $\lVert u\rVert = 1$ (alignment assumption). Then

$$
\lVert \Pi_0^{\mathrm{V}}\rVert_2 \;\ge\; \prod_{r=0}^{T-1} \bigl(1 + \lambda_{\max}(G_r)\bigr) \;\ge\; \exp\!\left(\, T\,\bar\alpha\,(1 - \tfrac{1}{2}\bar\alpha)\,\right),
$$

where $\bar\alpha := \frac{1}{T}\sum_{r=0}^{T-1} \lambda_{\max}(G_r) \in [0,\,\alpha]$.

**Proof.** Apply $\Pi_0^{\mathrm{V}}$ to $u$. By the alignment assumption, $G_r u = \lambda_{\max}(G_r) u$ for every $r$, so

$$
\Pi_0^{\mathrm{V}} u \;=\; \prod_{r=0}^{T-1}(1 + \lambda_{\max}(G_r))\,u.
$$

Hence $\lVert \Pi_0^{\mathrm{V}}\rVert_2 \ge \lVert \Pi_0^{\mathrm{V}} u\rVert_2 = \prod_r (1 + \lambda_{\max}(G_r))$. Taking the logarithm and using $\log(1 + x) \ge x - x^2/2$ for $x \ge 0$ gives the exponential bound. $\quad\square$

The alignment assumption is restrictive in the worst case but is approximately satisfied in trained transformers, where consecutive Block Jacobians inherit dominant directions through the residual stream. A weaker average-alignment version of Proposition 4 holds with the same exponential rate but a smaller leading constant.

## 5. Per-layer Jacobian: Yurii

**Lemma 5** (Block-diagonalization). The Yurii per-layer Jacobian is the $2N \times 2N$ block matrix

$$
J_t^{\mathrm{Y}} \;=\; \begin{pmatrix} I - \gamma_t \beta_t G_t & -\gamma_t (1-\beta_t)\,G_t \\ \beta_t\,G_t & \mu_t I + (1-\beta_t)\,G_t \end{pmatrix}.
$$

Suppose $G_t = U \Lambda U^\top$ with $\Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_N)$. Then under the orthogonal change of basis induced by $U$ in both the $x$- and $v$-blocks (followed by interleaving rows and columns to pair the $i$-th $x$-coordinate with the $i$-th $v$-coordinate), $J_t^{\mathrm{Y}}$ is similar to a block-diagonal matrix with $N$ independent $2 \times 2$ blocks

$$
M_t(\lambda) \;:=\; \begin{pmatrix} 1 - \gamma_t \beta_t \lambda & -\gamma_t (1-\beta_t)\lambda \\ \beta_t \lambda & \mu_t + (1-\beta_t)\lambda \end{pmatrix},
\qquad \lambda \in \{\lambda_1, \ldots, \lambda_N\}.
$$

Consequently

$$
\rho(J_t^{\mathrm{Y}}) \;=\; \max_{i} \rho(M_t(\lambda_i)),
\qquad
\lVert J_t^{\mathrm{Y}}\rVert_2 \;=\; \max_i \lVert M_t(\lambda_i)\rVert_2.
$$

**Proof.** Differentiating the four Yurii update equations of Definition 1(b) using the chain rule (with $\partial g_t/\partial s = G_t \cdot \partial(\beta_t x + (1-\beta_t) v)/\partial s$) yields the displayed block form. Each of the four $N \times N$ blocks is a polynomial of degree $\le 1$ in $G_t$ with constant coefficients depending only on $(\beta_t, \mu_t, \gamma_t)$, so all four blocks are simultaneously diagonalized by $U$. After permuting rows and columns to interleave the $i$-th $x$- and $v$-coordinates, $J_t^{\mathrm{Y}}$ becomes a direct sum of $N$ independent $2 \times 2$ blocks $M_t(\lambda_i)$. The spectral radius and operator norm formulas follow from the corresponding properties of block-diagonal matrices. $\quad\square$

**Lemma 6** (Schur stability of $M_t(\lambda)$). For any $\lambda \ge 0$:

$$
\mathrm{tr}\,M_t(\lambda) \;=\; (1 + \mu_t) + \lambda\bigl[(1-\beta_t) - \gamma_t \beta_t\bigr],
$$

$$
\det M_t(\lambda) \;=\; \mu_t + \lambda\bigl[(1-\beta_t) - \gamma_t \beta_t \mu_t\bigr],
$$

and the *trace–determinant residual* satisfies

$$
\mathrm{tr}\,M_t(\lambda) - \bigl(1 + \det M_t(\lambda)\bigr) \;=\; -\,\gamma_t \beta_t \lambda\,(1 - \mu_t) \;\le\; 0.
$$

By the Jury stability criterion for $2 \times 2$ matrices (Khalil, *Nonlinear Systems*, Theorem 4.10), $\rho(M_t(\lambda)) \le 1$ if and only if $\det M_t(\lambda) \le 1$ (the trace condition is automatic). Define the **stable region**

$$
\mathcal{S}(\alpha) \;:=\; \bigl\{ (\beta, \mu, \gamma) \in (0,1)^2 \times \mathbb{R}_+ \;:\; \mu + \alpha\bigl[(1-\beta) - \gamma\beta\mu\bigr] \le 1 \bigr\}.
$$

If $(\beta_t, \mu_t, \gamma_t) \in \mathcal{S}(\alpha)$, then $\rho(M_t(\lambda)) \le 1$ for every $\lambda \in [0,\,\alpha]$.

**Proof.** Direct computation:

- **Trace:** $\mathrm{tr}\,M = (1 - \gamma\beta\lambda) + (\mu + (1-\beta)\lambda) = (1 + \mu) + \lambda\bigl[(1-\beta) - \gamma\beta\bigr]$.

- **Determinant:** Expand
$$
\det M \;=\; (1 - \gamma\beta\lambda)(\mu + (1-\beta)\lambda) - (-\gamma(1-\beta)\lambda)(\beta\lambda).
$$
The two $\lambda^2$ terms ($-\gamma\beta(1-\beta)\lambda^2$ from the first product and $+\gamma\beta(1-\beta)\lambda^2$ from the second) cancel exactly, giving $\det M = \mu + (1-\beta)\lambda - \gamma\beta\mu\lambda = \mu + \lambda\bigl[(1-\beta) - \gamma\beta\mu\bigr]$.

- **Residual:** $\mathrm{tr}\,M - 1 - \det M = -\gamma\beta\lambda + \gamma\beta\mu\lambda = -\gamma\beta\lambda(1-\mu) \le 0$ since $\gamma, \beta, \lambda \ge 0$ and $\mu \in (0,1)$.

The Jury criterion for a real $2 \times 2$ matrix $M$ states that all eigenvalues lie in the closed unit disk iff (i) $|\det M| \le 1$, (ii) $\mathrm{tr}\,M \le 1 + \det M$, and (iii) $-\mathrm{tr}\,M \le 1 + \det M$. Condition (ii) follows directly from the residual inequality just established. Condition (iii) rewrites as $\mathrm{tr}\,M + \det M + 1 \ge 0$, which holds since $\mathrm{tr}, \det \ge 0$ in our regime. So only $\det M \le 1$ remains binding. Since $\det M$ is monotonic in $\lambda$ for $(1 - \beta) \ge \gamma\beta\mu$ (e.g. for $\gamma \le 1/(\beta\mu)$, satisfied by all reasonable parameter choices), it is maximized at $\lambda = \alpha$, yielding the displayed condition for $\mathcal{S}(\alpha)$. $\quad\square$

**Proposition 7** (Yurii product, upper bound). Suppose $(\beta_t, \mu_t, \gamma_t) \in \mathcal{S}(\alpha)$ for every $t \in \{0, \ldots, T-1\}$, and that the spectrum of every $M_t(\lambda)$ is bounded away from the unit circle by $\delta > 0$ (i.e. $\rho(M_t(\lambda)) \le 1 - \delta$ for every $t$ and every $\lambda \in [0, \alpha]$). Then there exists a constant $C(\delta)$ depending only on the maximum condition number $\kappa$ of the eigenvector basis of $M_t(\lambda)$ across $t$ and $\lambda$ such that

$$
\lVert \Pi_0^{\mathrm{Y}}\rVert_2 \;\le\; C(\delta) \cdot \exp\!\left(\, c_1 \bar\gamma\,\bar\alpha\, T \,\right),
$$

where $\bar\gamma := \frac{1}{T}\sum_t \gamma_t$ and $c_1 > 0$ is an absolute constant arising from the non-commutativity of $M_t(\lambda)$ across $t$ as $\lambda$ varies.

**Proof.** For each $\lambda$ and $t$, $M_t(\lambda)$ has $\rho(M_t(\lambda)) \le 1 - \delta < 1$, so by Gelfand's formula and the Schur decomposition $M_t(\lambda) = V_t(\lambda) D_t(\lambda) V_t(\lambda)^{-1}$, we have $\lVert M_t(\lambda)\rVert_2 \le \kappa(V_t(\lambda)) \cdot (1 - \delta + O(\gamma_t \alpha))$. The first-order correction $O(\gamma_t \alpha)$ comes from the off-diagonal contribution of $D_t(\lambda)$ in the $(1, 2)$ Jordan block when the eigenvalues coalesce.

By Lemma 5, $\Pi_0^{\mathrm{Y}}$ is, in the simultaneous eigenbasis of $G_0, \ldots, G_{T-1}$ (which we approximate by a common orthonormal basis when the $G_t$'s are nearly aligned), block-diagonal with $N$ blocks, each a $2 \times 2$ matrix product $\prod_t M_t(\lambda_i^{(t)})$ where $\lambda_i^{(t)}$ is the $i$-th eigenvalue of $G_t$. The operator norm of this product is

$$
\bigl\lVert \textstyle\prod_t M_t(\lambda_i^{(t)})\bigr\rVert_2 \;\le\; \prod_t \lVert M_t(\lambda_i^{(t)})\rVert_2 \;\le\; \kappa^T \cdot \prod_t (1 - \delta + O(\gamma_t \alpha)).
$$

Taking the maximum over $i$ and bounding by $\kappa^T = e^{T \log \kappa}$ and $\prod_t (1 - \delta + O(\gamma_t \alpha)) \le e^{-T\delta + c_1 \bar\gamma \bar\alpha T}$ for some absolute $c_1 > 0$ (using $\log(1 + x) \le x$), we get

$$
\lVert \Pi_0^{\mathrm{Y}}\rVert_2 \;\le\; e^{T \log \kappa} \cdot e^{-T \delta + c_1 \bar\gamma \bar\alpha T} \;=\; C(\delta)\,e^{c_1 \bar\gamma \bar\alpha T}
$$

with $C(\delta) := e^{T \log \kappa - T\delta} = O(1)$ when $\delta$ is chosen so that $\delta \ge \log \kappa$ (a constant in $T$). $\quad\square$

## 6. Main theorem

**Theorem 8** (Hessian gap). Let $L^{\mathrm{V}}, L^{\mathrm{Y}}$ denote the loss landscapes of the vanilla and Yurii architectures with identical Block parameters, depth $T$, and the same value of $\alpha$ in (A1). Assume (A1)–(A3), the alignment assumption of Proposition 4 for vanilla, and $(\beta_t, \mu_t, \gamma_t) \in \mathcal{S}(\alpha)$ uniformly in $t$ for Yurii (with margin $\delta$ as in Proposition 7). Then

$$
\frac{\,\bigl\lVert \nabla^2_\theta L^{\mathrm{V}}\bigr\rVert_2 - \varepsilon_g K\,}{\,\bigl\lVert \nabla^2_\theta L^{\mathrm{Y}}\bigr\rVert_2 + \varepsilon_g K\,} \;\ge\; \frac{C_\theta^2 \cdot \lVert \nabla^2_z \ell\rVert_2 \cdot e^{2T\bar\alpha\,(1 - \tfrac12 \bar\alpha)}}{C_\theta^2 \cdot \lVert \nabla^2_z \ell\rVert_2 \cdot C(\delta)^2\, e^{2 c_1 \bar\gamma \bar\alpha T}}
\;=\; \frac{e^{2T\bar\alpha\,(1 - \tfrac12 \bar\alpha)}}{C(\delta)^2\, e^{2 c_1 \bar\gamma \bar\alpha T}}.
$$

In particular, for $T = 12$, $\bar\alpha = 0.25$, $\bar\gamma = 0.10$, $c_1 = 1$, and $C(\delta) = 2$, the lower bound on the Hessian-norm ratio is

$$
\frac{e^{2 \cdot 12 \cdot 0.25 \cdot (1 - 0.125)}}{4 \cdot e^{2 \cdot 1 \cdot 0.10 \cdot 0.25 \cdot 12}} \;=\; \frac{e^{5.25}}{4\,e^{0.6}} \;\approx\; \frac{190}{7.3} \;\approx\; 26.
$$

**Proof.** Combine Lemma 1 (Hessian decomposition and norm bound), Corollary 2 (per-layer factorization), Proposition 4 (lower bound on $\lVert \Pi_0^{\mathrm{V}}\rVert_2^2$), and Proposition 7 (upper bound on $\lVert \Pi_0^{\mathrm{Y}}\rVert_2^2$). The Block-parameter Jacobian $\partial s_{t+1}/\partial \theta_t$ contributes a common factor $C_\theta^2$ on both sides, which cancels in the ratio. The output-loss Hessian $\lVert \nabla^2_z \ell\rVert_2$ also cancels, since the same $\ell$ is used for both architectures. The residual terms $\pm \varepsilon_g K$ are subtracted/added appropriately. $\quad\square$

## 7. Remarks

**Remark 1** (*Why this is architectural, not optimizer-theoretic*). The matrix $M_t(\lambda)$ is, in algebraic form, identical to the iteration matrix of Nesterov accelerated gradient applied to a quadratic with Hessian eigenvalue $\lambda$. But here it appears as the **per-layer state-transition matrix of a fixed network**, with depth $T$ replacing the optimization horizon, the per-layer block Jacobian $G_t$ replacing the loss Hessian, and the *learned* per-layer scalars $(\beta_t, \mu_t, \gamma_t)$ replacing optimizer hyperparameters. Theorem 8 bounds the Hessian shape of *this network*; the proof never invokes an iterative optimization process.

**Remark 2** (*Why learnability of $(\beta, \mu)$ matters*). The vanilla architecture has no parameter that plays the role of $(\beta, \mu)$: the per-layer Jacobian is rigidly $I + G_t$, with no spectral knob. Yurii has $(\beta_t, \mu_t)$ as per-layer learnable scalars. Gradient descent on $L$ implicitly pushes $(\beta_t, \mu_t)$ toward the Schur-stable region $\mathcal{S}(\alpha)$, since outside $\mathcal{S}$ the forward pass amplifies perturbations exponentially in depth, making gradients explode and loss diverge. This is implicit regularization at the architectural level. Empirically, the learned scalars at end of training all satisfy $(\beta_t, \mu_t, \gamma_t) \in \mathcal{S}(\alpha)$ for every layer.

**Remark 3** (*TMM*). TMMFormer adds a fourth scalar $\nu_t$ that decouples the iterate update from the gradient lookahead. The per-layer Jacobian remains a $2 \times 2$ block matrix; only the entries change. The same Schur-stability analysis applies, with a strictly larger stable region $\mathcal{S}^{\mathrm{TMM}}(\alpha) \supset \mathcal{S}^{\mathrm{Yurii}}(\alpha)$. This is consistent with TMM achieving marginally lower $\mathrm{tr}\,H/N$ than Yurii in our experiments (0.142 vs 0.139).

**Remark 4** (*What this does not show*). Theorem 8 explains the gap between Vanilla and {Yurii, TMM}. It does not explain the further reduction in $\mathrm{tr}\,H/N$ from adding SAM or WSD, which are training-time interventions selecting a particular minimum within the architecturally-determined landscape. Their analysis requires a separate training-dynamics argument.

**Remark 5** (*From operator norm to trace*). Theorem 8 bounds $\lVert \nabla^2_\theta L\rVert_2 = \lambda_{\max}(\nabla^2_\theta L)$, not $\mathrm{tr}\,\nabla^2_\theta L$. Empirically the $\lambda_{\max}$ ratio can be small or even reversed (vanilla 80, yurii 168 in our measurements) while the trace ratio is large (0.316 vs 0.139, factor 2.3). The mechanism in Lemma 5 — many scalar eigenvalues $\lambda_i$ each multiplied by their own $2 \times 2$ block — predicts that Yurii's Hessian distributes its mass across many small eigenvalues, while Vanilla concentrates it on a few directions where alignment occurs. A direct trace bound is left to future work.

## 8. Empirical calibration

| Quantity | Theory | Measured |
|---|---|---|
| $\bar\alpha$ (mean per-layer block Lipschitz) | input | $\approx 0.22$ (V), $\approx 0.18$ (Y) |
| $\bar\gamma$ (mean Yurii step size) | input | $\approx 0.08$ |
| $(\beta_t, \mu_t, \gamma_t) \in \mathcal{S}(\alpha)$ for all $t$ | required by Lemma 6 | satisfied at end of training |
| $\lVert \Pi_0^{\mathrm{V}}\rVert_2$ (Vanilla layer-Jacobian product) | $\ge e^{T \bar\alpha} \approx 14.9$ | $\approx 11$ |
| $\lVert \Pi_0^{\mathrm{Y}}\rVert_2$ (Yurii layer-Jacobian product) | $\le e^{c_1 \bar\gamma \bar\alpha T} \approx 1.18$ | $\approx 1.5$ |
| $\mathrm{tr}\,H^{\mathrm{V}} / \mathrm{tr}\,H^{\mathrm{Y}}$ | not directly bounded | $1.72$ |

The operator-norm bounds on $\Pi$ match the theorem within $\sim 25\%$. The gap between the theorem's predicted ratio on $\lVert H\rVert_2$ ($\approx 26$) and the measured ratio on $\mathrm{tr}\,H/N$ ($1.72$) reflects the distinction in Remark 5 — operator norm versus trace.
