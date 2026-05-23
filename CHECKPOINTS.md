# Pretrained checkpoints

All checkpoints are hosted on Hugging Face. The **consolidated mirror** is

> **<https://huggingface.co/gaijingchu/momentum-streams-checkpoints>**

with one `<variant>/best.pt` per architecture. Each `best.pt` is the lowest-val-loss model
state dict from the canonical SLURM run that produced the numbers in the paper.

## OpenWebText (30k steps, 12L / 12H / d=768, paper Table 1)

| Variant              | Validation loss | Consolidated path             |
|---|---:|---|
| VanillaTransformer   | 3.0078     | `vanilla_owt/best.pt`         |
| Vanilla + pure AdamW | 3.0103     | `vanilla_adamw_owt/best.pt`   |
| AdamFormer           | 2.9911     | `adam_owt/best.pt`            |
| AdamWFormer          | 2.9883     | `adamw_owt/best.pt`           |
| TMMFormer            | **2.9342** | `tmm_owt/best.pt`             |
| HBFormer             | —          | `hb_owt/best.pt`              |
| RMSPropFormer        | —          | `rmsprop_owt/best.pt`         |

## OpenWebText with WSD / SAM / SAWD recipes (RESEARCH_NOTES.md §loss-landscape)

| Variant           | Val loss | Consolidated path           |
|---|---:|---|
| TMMFormer + SAM   | 2.940 | `tmm_sam_owt/best.pt`         |
| TMMFormer + WSD   | 2.924 | `tmm_wsd_owt/best.pt`         |
| TMMFormer + SAWD  | —     | `tmm_sawd_owt/best.pt`        |

## TinyStories (10k steps)

| Variant            | Val loss | Consolidated path        |
|---|---:|---|
| Vanilla            | 1.1569 | `vanilla_ts/best.pt`       |
| Adam               | 1.153  | `adam_ts/best.pt`          |
| AdamW              | 1.147  | `adamw_ts/best.pt`         |
| TMMFormer          | 1.128  | `tmm_ts/best.pt`           |
| MuonFormer         | 1.1503 | `muon_ts/best.pt`          |
| SOAPFormer         | 1.1431 | `soap_ts/best.pt`          |
| OrthoFormer        | —      | `ortho_ts/best.pt`         |
| HBFormer           | —      | `hb_ts/best.pt`            |
| RMSPropFormer      | —      | `rmsprop_ts/best.pt`       |
| TMM + SAM          | 1.079  | `tmm_sam_ts/best.pt`       |
| TMM + WSD          | 1.086  | `tmm_wsd_ts/best.pt`       |
| TMM + SAWD         | 1.082  | `tmm_sawd_ts/best.pt`      |

## Param-matched controls (paper §3.2, TinyStories)

| Variant                                 | Val loss  | Consolidated path           | Note |
|---|---:|---|---|
| Vanilla width-900 (162.86 M)            | 1.1454    | `vanilla_w900_ts/best.pt`   | clean control, single seed (upload pending) |
| Vanilla depth-18 (166.85 M)             | ≈1.130    | `vanilla_d18_ts/best.pt`    | requeue-disturbed, suggestive (upload pending) |

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

All paper checkpoints in one go:

```bash
HF_TOKEN=<optional> CACHE=./checkpoints_cache python hf/download_ckpts.py
```

## Loading a checkpoint

The state dict contains exactly the model `state_dict()` (no optimizer state). Use the matching
model class:

```python
import torch
from formers.tmm.model import TMMFormer

model = TMMFormer(n_layers=12, n_heads=12, d_model=768, vocab_size=50304)
state = torch.load("tmm_owt/best.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

For Vanilla / Adam / Muon / etc., use the corresponding class:
`formers.vanilla.model.VanillaTransformer`, `formers.adam.model.AdamFormer`,
`formers.adamw.model.AdamWFormer`, `formers.muon.model.MuonFormer`,
`formers.ortho.model.OrthoFormer`, `formers.rmsprop.model.RMSPropFormer`,
`formers.shampoo.model.ShampooFormer`, `formers.soap.model.SOAPFormer`,
`formers.hb.model.HBFormer`. See the per-variant `formers/<v>/eval_model.py` files for
canonical loading wrappers.
