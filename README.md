# Momentum Streams for Optimizer-Inspired Transformers

Code release for *Momentum Streams for Optimizer-Inspired Transformers* (Gai, Huang, Wu, 2026).

> The residual update of a pre-norm Transformer layer can be read as one step of a first-order
> optimizer acting on a surrogate token energy, with attention and MLP sublayers as gradient
> oracles. We build a family of optimizer-inspired Transformers (triple-momentum, Adam/AdamW,
> Muon, SOAP) and compare them under matched compute. In our main pretraining experiment, the
> triple-momentum **TMMFormer** achieves the lowest validation loss, outperforming the vanilla
> Transformer and prior architectural variants. A controlled ablation and supporting theory show
> that **momentum, not preconditioning**, is the main source of the gain. We further show that
> TMMFormer and other momentum-based designs reach flatter minima than the vanilla Transformer,
> which leads to less forgetting and better generalization.

**Pretrained checkpoints:** all `best.pt` files are hosted at
<https://huggingface.co/gaijingchu/momentum-streams-checkpoints>.
See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the per-variant table and download recipes.

---

## Repository layout

```
.
├── data/                          # token loaders
│   ├── tinystories.py
│   └── openwebtext.py
│
├── formers/                       # one folder per architecture
│   ├── vanilla/                   # baseline 12L/12H/d=768 pre-norm Transformer
│   ├── tmm/                       # TMMFormer (triple-momentum; paper main result)
│   ├── adam/  · adamw/            # AdamFormer · AdamWFormer
│   ├── muon/                      # MuonFormer (per-token head-wise Newton-Schulz)
│   ├── soap/                      # SOAPFormer (right-factor Kronecker preconditioning)
│   ├── shampoo/                   # ShampooFormer (factorial ablation: SOAP - momentum)
│   ├── ortho/                     # OrthoFormer (factorial ablation: Muon - momentum)
│   ├── rmsprop/                   # RMSPropFormer (factorial ablation: Adam - momentum)
│   └── hb/                        # HBFormer (factorial ablation: Yurii - lookahead)
│   Each former dir contains:
│     model.py             — the nn.Module
│     train_ts.py          — DDP training on TinyStories  (10k steps)
│     train_owt.py         — DDP training on OpenWebText  (30k steps)
│     train_adamw_owt.py   — pure-AdamW ablation          (paper §3.2)
│     eval_model.py        — lm-eval wrapper
│     eval_run.py          — downstream evaluation script
│
├── eval/                          # cross-variant evaluation
│   ├── cross_corpus.py            — zero-shot PPL on wikitext / LAMBADA / C4
│   ├── finetune_forgetting.py     — owt2ts / ts2owt fine-tune for catastrophic forgetting
│   ├── finetune_forgetting_native_optim.py
│   ├── hybrid_model.py            — model wrapper for the hybrid (per-block) ablations
│   └── hybrid_run.py
│
├── analysis/                      # paper §5 + RESEARCH_NOTES.md
│   ├── loss_sharpness.py          — Hutchinson trace + power-iter lambda_max
│   ├── attention_entropy.py
│   ├── block_jacobian_spectrum.py     — full-block per-layer spectrum
│   ├── local_jacobian_spectrum.py     — canonical residual oracle spectrum
│   └── aggregate_block_spectrum.py
│
├── plotting/                      # paper figures
│   ├── owt_convergence.py · owt_all_optimizers.py
│   ├── ts_all_optimizers.py · training_curves.py
│
├── hf/                            # HuggingFace upload/download helpers
│   ├── download_ckpts.py
│   ├── upload_ckpts.py
│   ├── upload_vanilla_adamw_ckpt.py
│   ├── download_base_checkpoints.py
│   └── upload_checkpoints.py
│
├── results/                       # JSON outputs behind the paper figures
│   ├── attention_entropy/
│   ├── block_spectrum/
│   ├── cross_corpus/
│   ├── forgetting/
│   ├── loss_sharpness/
│   ├── local_jacobian_spectrum/
│   └── summaries/                 — landscape_summary.json · owt_convergence.json · …
│
├── README.md  CHECKPOINTS.md  RESEARCH_NOTES.md
├── results_optimizer_and_lr_ablation_owt.md
├── results_param_matched_seed_variance.md
└── pyproject.toml  uv.lock  .gitignore
```

Each script under a subdir adds the repo root to `sys.path`, so running from the
repo root just works without `pip install -e .`.

---

## Installation

Python ≥ 3.10. The project uses [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install torch tiktoken numpy wandb requests "datasets>=4.8.4" \
            "lm-eval==0.4.3" "matplotlib>=3.10.8" "scipy>=1.15.3"
```

GPU: training is multi-GPU DDP (default 2× GPU, ≥ 24 GB VRAM). All numbers in the paper come from
L40S × 2 with `bf16` autocast.

---

## Data

Tokenization is **GPT-2 BPE** via `tiktoken` (vocab size 50,304), context length 1,024.

- `data/tinystories.py` downloads `roneneldan/TinyStories` to `$CACHE/tinystories/{train,val}.npy`.
- `data/openwebtext.py` downloads `Skylion007/openwebtext` (≈ 8.6 GB tokens) to
  `$CACHE/openwebtext/{train,val}.npy`.

```bash
export CACHE=$HOME/.cache/momentum-streams
```

Default fallback is `./checkpoints_cache` (created in the working directory).

---

## Pretraining

Run from the repo root. Each variant has its own train scripts in `formers/<variant>/`.

### TinyStories (10k steps, ≈ 2 h on 2× L40S)

```bash
torchrun --standalone --nproc_per_node=2 formers/vanilla/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/tmm/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/adam/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/adamw/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/muon/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/ortho/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/rmsprop/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/shampoo/train_ts.py
torchrun --standalone --nproc_per_node=2 formers/hb/train_ts.py
```

### OpenWebText (30k steps, ≈ 23 h on 2× L40S)

```bash
torchrun --standalone --nproc_per_node=2 formers/vanilla/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/tmm/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/adam/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/adamw/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/muon/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/ortho/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/rmsprop/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/shampoo/train_owt.py
torchrun --standalone --nproc_per_node=2 formers/hb/train_owt.py
```

### Pure-AdamW ablation (paper §3.2)

```bash
torchrun --standalone --nproc_per_node=2 formers/vanilla/train_adamw_owt.py
torchrun --standalone --nproc_per_node=2 formers/tmm/train_adamw_owt.py
```

### Halved-LR ablation (paper §3.2)

```bash
MUON_LR=0.002 LR_SUFFIX=_halflr torchrun --standalone --nproc_per_node=2 formers/vanilla/train_owt.py
MUON_LR=0.002 LR_SUFFIX=_halflr torchrun --standalone --nproc_per_node=2 formers/tmm/train_owt.py
```

### Parameter-matched controls (paper §3.2)

```bash
D_MODEL=900 RUN_NAME=vanilla-w900 \
  torchrun --standalone --nproc_per_node=2 formers/vanilla/train_ts.py
N_LAYERS=18 RUN_NAME=vanilla-d18 \
  torchrun --standalone --nproc_per_node=2 formers/vanilla/train_ts.py
```

Defaults (overridable via env var): `SEED=42`, `N_LAYERS=12`, `D_MODEL=768`, `N_HEADS=12`,
`CKPT_DIR=./checkpoints`, `RUN_NAME=<variant>`. The full hyper-parameter recipe (Tables 3–6 of
the paper) is hard-coded in the train scripts; see Appendix E of the paper for the source of
truth.

---

## Evaluation

### Validation loss

Training logs `val_loss` every 100 steps (160 batches) and keeps the lowest as `best.pt`.

### Downstream (HellaSwag, ARC-Easy)

Uses `lm-evaluation-harness v0.4.3`. Per-variant scripts wrap the harness:

```bash
python formers/tmm/eval_run.py     --checkpoint <path>/best.pt --output-dir results/downstream_tmm/
python formers/vanilla/eval_run.py --checkpoint <path>/best.pt --output-dir results/downstream_vanilla/
```

We report length-normalized accuracy (`acc_norm`), the harness default for multiple-choice tasks.

### Loss-landscape sharpness (paper §5.1)

```bash
python analysis/loss_sharpness.py --checkpoint <path>/best.pt --variant tmm --n-batches 32
```

Outputs `λ_max(H)` via power iteration and `tr(H)/N` via Hutchinson's Rademacher estimator.

### Cross-corpus zero-shot PPL (paper §5.2)

```bash
python eval/cross_corpus.py --checkpoint <path>/best.pt --variant tmm
```

Reports val PPL on OWT (in-dist), `wikitext-103`, `LAMBADA`, and `C4 (first 5k docs)`.

### Forgetting / plasticity (paper §5.2)

```bash
torchrun --standalone --nproc_per_node=2 eval/finetune_forgetting.py \
  --src-checkpoint <owt-ckpt>/best.pt --variant tmm --direction owt2ts
```

### Block / local Jacobian spectrum (RESEARCH_NOTES.md Sandbox A/B)

```bash
python analysis/local_jacobian_spectrum.py --checkpoint <path>/best.pt --variant tmm   # canonical R_l
python analysis/block_jacobian_spectrum.py --checkpoint <path>/best.pt --variant tmm   # full block F_l
python analysis/aggregate_block_spectrum.py                                            # combines variants
```

---

## Pretrained checkpoints

A consolidated mirror is at **`gaijingchu/momentum-streams-checkpoints`** on Hugging Face,
with one subdirectory per variant. See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the full table
and download recipes.

Bulk download:

```bash
HF_TOKEN=<optional> CACHE=./checkpoints_cache python hf/download_ckpts.py
```

(`HF_TOKEN` is only needed for private repos; the released checkpoints are public.)

---

## Reproducing the ablations

| Section | Document |
|---|---|
| §3.2 Optimizer-on-2D + LR sweep | [`results_optimizer_and_lr_ablation_owt.md`](results_optimizer_and_lr_ablation_owt.md) |
| §3.2 Parameter-matched + seed variance | [`results_param_matched_seed_variance.md`](results_param_matched_seed_variance.md) |
| §5 Loss landscape, forgetting, theory | [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) |

Each document lists exact SLURM job IDs, log paths, and val curves to make the numbers
auditable.

---

## Citation

```bibtex
@article{gai2026momentum,
  title  = {Momentum Streams for Optimizer-Inspired Transformers},
  author = {Gai, Jingchu and Huang, Nai-Chieh and Wu, Jiayun},
  year   = {2026}
}
```
