# Pretrained checkpoints

All checkpoints are hosted on Hugging Face. The **consolidated mirror** is

> **<https://huggingface.co/gaijingchu/momentum-streams-checkpoints>**

with one `<variant>/best.pt` per architecture. The original per-variant repos (one repo per
variant, as the checkpoints were uploaded during the project) are also kept live as a backup.

Each `best.pt` is the lowest-val-loss model state dict from the canonical SLURM run that
produced the numbers in the paper. Variant ↔ HF repo:

## OpenWebText (30k steps, 12L / 12H / d=768, paper Table 1)

| Variant            | Validation loss | Consolidated path                          | Original repo                                      |
|---|---:|---|---|
| VanillaTransformer | 3.0078 | `vanilla_owt/best.pt`        | [`gaijingchu/checkpoints_vanilla_owt`](https://huggingface.co/gaijingchu/checkpoints_vanilla_owt) |
| Vanilla + pure AdamW | 3.0103 | `vanilla_adamw_owt/best.pt` | [`gaijingchu/checkpoints_vanilla_adamw_owt`](https://huggingface.co/gaijingchu/checkpoints_vanilla_adamw_owt) |
| AdamFormer         | 2.9911 | `adam_owt/best.pt`           | [`gaijingchu/checkpoints_adam_owt`](https://huggingface.co/gaijingchu/checkpoints_adam_owt) |
| AdamWFormer        | 2.9883 | `adamw_owt/best.pt`          | [`gaijingchu/checkpoints_adamw_owt`](https://huggingface.co/gaijingchu/checkpoints_adamw_owt) |
| YuriiFormer        | 2.9413 | `yurii_owt/best.pt`          | [`gaijingchu/checkpoints_yurii_owt`](https://huggingface.co/gaijingchu/checkpoints_yurii_owt) |
| TMMFormer          | **2.9342** | `tmm_owt/best.pt`        | [`gaijingchu/checkpoints_tmm_owt`](https://huggingface.co/gaijingchu/checkpoints_tmm_owt) |
| TMMFormer + pure AdamW | 2.9696 | `tmm_adamw_owt/best.pt`  | (in consolidated repo) |
| HBFormer (heavy-ball) | —    | `hb_owt/best.pt`            | [`gaijingchu/checkpoints_hb_owt`](https://huggingface.co/gaijingchu/checkpoints_hb_owt) |
| RMSPropFormer      | —      | `rmsprop_owt/best.pt`        | [`gaijingchu/checkpoints_rmsprop_owt`](https://huggingface.co/gaijingchu/checkpoints_rmsprop_owt) |

## OpenWebText with WSD / SAM / SAWD recipes (RESEARCH_NOTES.md §loss-landscape)

| Variant           | Val loss | Consolidated path           | Original repo |
|---|---:|---|---|
| TMMFormer + SAM   | 2.940 | `tmm_sam_owt/best.pt`         | [`gaijingchu/checkpoints_tmm_sam_owt`](https://huggingface.co/gaijingchu/checkpoints_tmm_sam_owt) |
| TMMFormer + WSD   | 2.924 | `tmm_wsd_owt/best.pt`         | [`gaijingchu/checkpoints_tmm_wsd_owt`](https://huggingface.co/gaijingchu/checkpoints_tmm_wsd_owt) |
| TMMFormer + SAWD  | —     | `tmm_sawd_owt/best.pt`        | [`gaijingchu/checkpoints_tmm_sawd_owt`](https://huggingface.co/gaijingchu/checkpoints_tmm_sawd_owt) |
| YuriiFormer + SAM | 2.948 | `yurii_sam_owt/best.pt`       | [`gaijingchu/checkpoints_yurii_sam_owt`](https://huggingface.co/gaijingchu/checkpoints_yurii_sam_owt) |
| YuriiFormer + WSD | 2.928 | `yurii_wsd_owt/best.pt`       | [`gaijingchu/checkpoints_yurii_wsd_owt`](https://huggingface.co/gaijingchu/checkpoints_yurii_wsd_owt) |
| YuriiFormer + SAWD| 2.932 | `yurii_sawd_owt/best.pt`      | [`gaijingchu/checkpoints_yurii_sawd_owt`](https://huggingface.co/gaijingchu/checkpoints_yurii_sawd_owt) |

## TinyStories (10k steps)

The TS-pretrained variants are bundled in a single archive repo (one subdir per variant) in
both the consolidated and the original layout.

| Variant            | Val loss | Consolidated path        | Original location |
|---|---:|---|---|
| Vanilla            | 1.1569 | `vanilla_ts/best.pt`       | `gaijingchu/checkpoints_ts_archive` → `vanilla_ts/best.pt` |
| Adam               | 1.153  | `adam_ts/best.pt`          | `…/adam_ts/best.pt` |
| AdamW              | 1.147  | `adamw_ts/best.pt`         | `…/adamw_ts/best.pt` |
| YuriiFormer        | 1.130  | `yurii_ts/best.pt`         | `…/yurii_ts/best.pt` |
| TMMFormer          | 1.128  | `tmm_ts/best.pt`           | `…/tmm_ts/best.pt` |
| Yurii + SAM        | 1.081  | `yurii_sam_ts/best.pt`     | `…/yurii_sam_ts/best.pt` |
| Yurii + WSD        | 1.082  | `yurii_wsd_ts/best.pt`     | `…/yurii_wsd_ts/best.pt` |
| Yurii + SAWD       | 1.077  | `yurii_sawd_ts/best.pt`    | `…/yurii_sawd_ts/best.pt` |
| TMM + SAM          | 1.079  | `tmm_sam_ts/best.pt`       | [`gaijingchu/checkpoints_tmm_sam_ts`](https://huggingface.co/gaijingchu/checkpoints_tmm_sam_ts) |
| TMM + WSD          | 1.086  | `tmm_wsd_ts/best.pt`       | [`gaijingchu/checkpoints_tmm_wsd_ts`](https://huggingface.co/gaijingchu/checkpoints_tmm_wsd_ts) |
| TMM + SAWD         | 1.082  | `tmm_sawd_ts/best.pt`      | [`gaijingchu/checkpoints_tmm_sawd_ts`](https://huggingface.co/gaijingchu/checkpoints_tmm_sawd_ts) |
| HBFormer           | —      | `hb_ts/best.pt`            | `…/hb_ts/best.pt` |
| RMSPropFormer      | —      | `rmsprop_ts/best.pt`       | `…/rmsprop_ts/best.pt` |
| OrthoFormer        | —      | `ortho_ts/best.pt`         | `…/ortho_ts/best.pt` |
| MuonFormer         | 1.1503 | `muon_ts/best.pt`          | [`gaijingchu/ANLP-Yuriiformer-reproduce-MuonFormer`](https://huggingface.co/gaijingchu/ANLP-Yuriiformer-reproduce-MuonFormer) → `checkpoints_muon/best.pt` |

## Param-matched controls (paper §3.2, TinyStories)

| Variant                                 | Val loss | Consolidated path | Note |
|---|---:|---|---|
| Vanilla width-900 (162.86 M)            | 1.1454 | `vanilla_w900_ts/best.pt` | clean control, single seed |
| Vanilla depth-18 (166.85 M)             | ≈1.130 | `vanilla_d18_ts/best.pt`  | requeue-disturbed, suggestive |

## Download

Single best.pt:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="gaijingchu/momentum-streams-checkpoints",
    filename="tmm_owt/best.pt",
    local_dir="./checkpoints_cache",
)
```

All paper checkpoints in one go (uses the original per-variant repos, edit list as needed):

```bash
HF_TOKEN=<optional> CACHE=./checkpoints_cache python download_ckpts.py
```

## Loading a checkpoint

The state dict contains exactly the model `state_dict()` (no optimizer state). Use the matching
model class:

```python
import torch
from tmm_model import TMMFormer

model = TMMFormer(n_layers=12, n_heads=12, d_model=768, vocab_size=50304)
state = torch.load("tmm_owt/best.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

For Vanilla / Adam / Yurii etc., import the corresponding class
(`VanillaTransformer`, `AdamFormer`, `YuriiFormer` from `model.py`, …) — see the per-variant
`*_eval_model.py` files for canonical loading wrappers.
