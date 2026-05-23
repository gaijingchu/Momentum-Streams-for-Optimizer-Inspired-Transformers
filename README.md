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

**Pretrained checkpoints:** all `best.pt` files are hosted on Hugging Face. See [`CHECKPOINTS.md`](CHECKPOINTS.md).

---

## Repository layout

```
.
├── vanilla_model.py / vanilla_train_{ddp,owt}.py / vanilla_adamw_train_owt.py
├── tmm_model.py     / tmm_train_{ddp,owt}.py     / tmm_adamw_train_owt.py
├── model.py         (YuriiFormer)            / yurii_train_{ddp,owt}.py
├── adam_model.py    / adam_train_{ddp,owt}.py
├── adamw_model.py   / adamw_train_{ddp,owt}.py
├── muon_model.py    / muon_train_{ddp,owt}.py
├── ortho_model.py   / ortho_train_{ddp,owt}.py
├── rmsprop_model.py / rmsprop_train_{ddp,owt}.py
├── shampoo_model.py / shampoo_train_{ddp,owt}.py
├── hb_model.py      / hb_train_{ddp,owt}.py
├── soap_model.py    (SOAPFormer; right-factor Kronecker preconditioning)
│
├── data.py          (TinyStories token loader)
├── data_owt.py      (OpenWebText token loader)
│
├── eval_model.py    eval_run.py        eval.sh
├── {vanilla,tmm,adam,adamw}_eval_{model,run}.py
├── hybrid_eval_{model,run}.py
├── eval_cross_corpus.py           (zero-shot PPL on wikitext / LAMBADA / C4)
├── finetune_forgetting{,_native_optim}.py
│
├── loss_sharpness.py              (Hutchinson trace + power-iter λ_max)
├── attention_entropy.py
├── block_jacobian_spectrum.py     (full-block per-layer spectrum)
├── local_jacobian_spectrum.py     (canonical residual oracle spectrum)
├── aggregate_block_spectrum.py
├── dump_tmm_scalars.py
│
├── plot_owt_convergence.py        plot_training_curves.py
├── plot_owt_all_optimizers.py     plot_ts_all_optimizers.py
│
├── download_ckpts.py              upload_ckpts.py
├── hf_download_base_checkpoints.py
├── upload_vanilla_adamw_ckpt.py
│
├── results_optimizer_and_lr_ablation_owt.md
├── results_param_matched_seed_variance.md
├── RESEARCH_NOTES.md              (extended technical journal & theory)
└── CHECKPOINTS.md
```

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

For lm-evaluation-harness downstream tasks, `pip install lm-eval==0.4.3`.

GPU: training is multi-GPU DDP (default 2× GPU, ≥ 24 GB VRAM). All numbers in the paper come from
L40S × 2 (`bf16` autocast).

---

## Data

Tokenization is **GPT-2 BPE** via `tiktoken` (vocab size 50,304), context length 1,024.

### TinyStories

`data.py` downloads and tokenizes `roneneldan/TinyStories` to `$CACHE/tinystories/{train,val}.npy`
on first use.

### OpenWebText

`data_owt.py` downloads `Skylion007/openwebtext` (≈ 8.6 GB tokens) on first use to
`$CACHE/openwebtext/{train,val}.npy`. Adjust `CACHE` via env var:

```bash
export CACHE=$HOME/.cache/momentum-streams
```

The default fallback is `./checkpoints_cache` (created in the working directory).

---

## Pretraining

Each variant has its own train script. The 2-GPU DDP recipe is:

```bash
# TinyStories (10k steps, ≈ 2 h on 2× L40S)
torchrun --standalone --nproc_per_node=2 vanilla_train_ddp.py
torchrun --standalone --nproc_per_node=2 tmm_train_ddp.py
torchrun --standalone --nproc_per_node=2 adam_train_ddp.py
torchrun --standalone --nproc_per_node=2 adamw_train_ddp.py
torchrun --standalone --nproc_per_node=2 muon_train_ddp.py
torchrun --standalone --nproc_per_node=2 ortho_train_ddp.py
torchrun --standalone --nproc_per_node=2 rmsprop_train_ddp.py
torchrun --standalone --nproc_per_node=2 shampoo_train_ddp.py
torchrun --standalone --nproc_per_node=2 hb_train_ddp.py
torchrun --standalone --nproc_per_node=2 yurii_train_ddp.py

# OpenWebText (30k steps, ≈ 23 h on 2× L40S)
torchrun --standalone --nproc_per_node=2 vanilla_train_owt.py
torchrun --standalone --nproc_per_node=2 tmm_train_owt.py
torchrun --standalone --nproc_per_node=2 adam_train_owt.py
torchrun --standalone --nproc_per_node=2 adamw_train_owt.py
torchrun --standalone --nproc_per_node=2 muon_train_owt.py
torchrun --standalone --nproc_per_node=2 ortho_train_owt.py
torchrun --standalone --nproc_per_node=2 rmsprop_train_owt.py
torchrun --standalone --nproc_per_node=2 shampoo_train_owt.py
torchrun --standalone --nproc_per_node=2 hb_train_owt.py
torchrun --standalone --nproc_per_node=2 yurii_train_owt.py
```

**Pure-AdamW ablation** (paper §3.2, Ablation Study, "Parameter-training optimizer"):

```bash
torchrun --standalone --nproc_per_node=2 vanilla_adamw_train_owt.py
torchrun --standalone --nproc_per_node=2 tmm_adamw_train_owt.py
```

**Halved-LR ablation** (paper §3.2, Ablation Study, "Peak learning rate"):

```bash
MUON_LR=0.002 LR_SUFFIX=_halflr torchrun --standalone --nproc_per_node=2 vanilla_train_owt.py
MUON_LR=0.002 LR_SUFFIX=_halflr torchrun --standalone --nproc_per_node=2 tmm_train_owt.py
```

**Parameter-matched controls** (paper §3.2, "Parameter-matched controls"):

```bash
D_MODEL=900 RUN_NAME=vanilla-w900 \
  torchrun --standalone --nproc_per_node=2 vanilla_train_ddp.py
N_LAYERS=18 RUN_NAME=vanilla-d18 \
  torchrun --standalone --nproc_per_node=2 vanilla_train_ddp.py
```

Defaults (overridable by env var): `SEED=42`, `N_LAYERS=12`, `D_MODEL=768`, `N_HEADS=12`,
`CKPT_DIR=./checkpoints`, `RUN_NAME=<variant>`.

The full hyper-parameter recipe (Tables 3–6 of the paper) is hard-coded in the train scripts;
see Appendix E of the paper for the source of truth.

---

## Evaluation

### Validation loss

Training logs `val_loss` every 100 steps (160 batches) and keeps the lowest as `best.pt`. To
re-evaluate a saved checkpoint:

```bash
python eval_run.py --checkpoint <path>/best.pt --output-dir eval_results/
```

### Downstream (HellaSwag, ARC-Easy)

Uses `lm-evaluation-harness v0.4.3`. Per-variant scripts wrap the harness:

```bash
python tmm_eval_run.py     --checkpoint <path>/best.pt --output-dir eval_results_tmm_owt/
python vanilla_eval_run.py --checkpoint <path>/best.pt --output-dir eval_results_vanilla_owt/
```

We report length-normalized accuracy (`acc_norm`), the harness default for multiple-choice tasks.

### Loss-landscape sharpness (paper §5.1)

```bash
python loss_sharpness.py --checkpoint <path>/best.pt --variant tmm --n-batches 32
```

Outputs `λ_max(H)` via power iteration and `tr(H)/N` via Hutchinson's Rademacher estimator.

### Cross-corpus zero-shot PPL (paper §5.2)

```bash
python eval_cross_corpus.py --checkpoint <path>/best.pt --variant tmm
```

Reports val PPL on OWT (in-dist), `wikitext-103`, `LAMBADA`, and `C4 (first 5k docs)`.

### Forgetting / plasticity (paper §5.2)

```bash
torchrun --standalone --nproc_per_node=2 finetune_forgetting.py \
  --src-checkpoint <owt-ckpt>/best.pt --variant tmm --direction owt2ts
```

### Block / local Jacobian spectrum (RESEARCH_NOTES.md Sandbox A/B)

```bash
python local_jacobian_spectrum.py --checkpoint <path>/best.pt --variant tmm   # canonical R_l
python block_jacobian_spectrum.py --checkpoint <path>/best.pt --variant tmm   # full block F_l
python aggregate_block_spectrum.py                                            # combines variants
```

---

## Pretrained checkpoints

A consolidated mirror is at **`gaijingchu/momentum-streams-checkpoints`** on Hugging Face,
with one subdirectory per variant. Per-variant repos (the originals) are also available; see
[`CHECKPOINTS.md`](CHECKPOINTS.md) for the full table and download recipes.

Bulk download via:

```bash
HF_TOKEN=<token> CACHE=./checkpoints_cache python download_ckpts.py
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

Building blocks we draw on:

```bibtex
@article{zimin2026yuriiformer,
  title   = {YuriiFormer: A Suite of Nesterov-Accelerated Transformers},
  author  = {Zimin, Aleksandr and Polyanskiy, Yury and Rigollet, Philippe},
  journal = {arXiv preprint arXiv:2601.23236},
  year    = {2026}
}
```
