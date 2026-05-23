"""Cross-corpus zero-shot perplexity evaluation.

For each variant, computes val perplexity on multiple corpora to test
the classic sharpness ↔ generalization claim (Foret et al. 2021 type B,
distribution shift).

Corpora:
  - OWT val             (in-distribution baseline for OWT-pretrained models)
  - WikiText-103 val    (closely related, encyclopedic web)
  - LAMBADA val         (long-range dependency, narrative)
  - Penn Treebank val   (older news style, distribution shift)
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

# Variant → (model_module, model_class, ckpt_subpath)
# ckpt_subpath is relative to $CACHE
VARIANT_MAP = {
    # OWT-pretrained
    "vanilla":    ("vanilla_model", "VanillaTransformer", "checkpoints_vanilla_owt/best.pt"),
    "adam":       ("adam_model",    "AdamFormer",         "checkpoints_adam_owt/best.pt"),
    "adamw":      ("adamw_model",   "AdamWFormer",        "checkpoints_adamw_owt/best.pt"),
    "yurii":      ("model",         "YuriiFormer",        "checkpoints_yurii_owt/best.pt"),
    "tmm":        ("tmm_model",     "TMMFormer",          "checkpoints_tmm_owt/best.pt"),
    "yurii_sam":  ("model",         "YuriiFormer",        "checkpoints_yurii_sam_owt/best.pt"),
    "yurii_wsd":  ("model",         "YuriiFormer",        "checkpoints_yurii_wsd_owt/best.pt"),
    "yurii_sawd": ("model",         "YuriiFormer",        "checkpoints_yurii_sawd_owt/best.pt"),
    "tmm_sam":    ("tmm_model",     "TMMFormer",          "checkpoints_tmm_sam_owt/best.pt"),
    "tmm_wsd":    ("tmm_model",     "TMMFormer",          "checkpoints_tmm_wsd_owt/best.pt"),
    "tmm_sawd":   ("tmm_model",     "TMMFormer",          "checkpoints_tmm_sawd_owt/best.pt"),
    "ts_tmm_sam":    ("tmm_model",     "TMMFormer",          "checkpoints_tmm_sam_ts/best.pt"),
    "ts_tmm_wsd":    ("tmm_model",     "TMMFormer",          "checkpoints_tmm_wsd_ts/best.pt"),
    "ts_tmm_sawd":   ("tmm_model",     "TMMFormer",          "checkpoints_tmm_sawd_ts/best.pt"),
    # TS-pretrained (use ts_archive path)
    "ts_vanilla":    ("vanilla_model", "VanillaTransformer", "checkpoints_ts_archive/vanilla_ts/best.pt"),
    "ts_adam":       ("adam_model",    "AdamFormer",         "checkpoints_ts_archive/adam_ts/best.pt"),
    "ts_adamw":      ("adamw_model",   "AdamWFormer",        "checkpoints_ts_archive/adamw_ts/best.pt"),
    "ts_yurii":      ("model",         "YuriiFormer",        "checkpoints_ts_archive/yurii_ts/best.pt"),
    "ts_tmm":        ("tmm_model",     "TMMFormer",          "checkpoints_ts_archive/tmm_ts/best.pt"),
    "ts_yurii_sam":  ("model",         "YuriiFormer",        "checkpoints_ts_archive/yurii_sam_ts/best.pt"),
    "ts_yurii_wsd":  ("model",         "YuriiFormer",        "checkpoints_ts_archive/yurii_wsd_ts/best.pt"),
    "ts_yurii_sawd": ("model",         "YuriiFormer",        "checkpoints_ts_archive/yurii_sawd_ts/best.pt"),
}

BLOCK_SIZE = 1024
N_BATCHES = 200
BATCH_SIZE = 8


def load_model(variant: str, device: str):
    if variant not in VARIANT_MAP:
        raise ValueError(f"unknown variant {variant}, choose from {list(VARIANT_MAP)}")
    mod_name, cls_name, ckpt_subpath = VARIANT_MAP[variant]

    # Import model class
    mod = __import__(mod_name)
    cls = getattr(mod, cls_name)
    model = cls()

    # Load checkpoint
    cache = Path(os.environ["CACHE"])
    ckpt_path = cache / ckpt_subpath
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, ckpt.get("val_loss", None)


def tokenize_corpus(name: str) -> np.ndarray:
    """Return numpy array of token IDs (concatenated with EOT separators)."""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    cache_dir = Path(os.environ["CACHE"]) / "cross_corpus_tokens"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.npy"
    if cache_file.exists():
        print(f"  [cache hit] {cache_file}", flush=True)
        return np.load(cache_file)

    from datasets import load_dataset
    print(f"  tokenizing {name} ...", flush=True)
    t0 = time.time()

    if name == "wikitext103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
        texts = [r["text"] for r in ds if r["text"].strip()]
    elif name == "lambada":
        ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        texts = [r["text"] for r in ds]
    elif name == "c4_small":
        # C4 streaming val, take 5000 docs (~5MB raw text → ~1.5M tokens)
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, r in enumerate(ds):
            if i >= 5000:
                break
            texts.append(r["text"])
    else:
        raise ValueError(f"unknown corpus {name}")

    tokens = []
    for t in texts:
        tokens.extend(enc.encode_ordinary(t))
        tokens.append(eot)
    arr = np.array(tokens, dtype=np.int32)
    np.save(cache_file, arr)
    print(f"  tokenized {name}: {len(arr):,} tokens in {time.time()-t0:.1f}s", flush=True)
    return arr


def load_owt_val_tokens() -> np.ndarray:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.openwebtext import load_owt_tokens
    return load_owt_tokens("val")


@torch.no_grad()
def compute_ppl(model, tokens: np.ndarray, n_batches: int = N_BATCHES,
                batch_size: int = BATCH_SIZE, block_size: int = BLOCK_SIZE,
                device: str = "cuda", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        starts = rng.integers(0, len(tokens) - block_size - 2, size=batch_size)
        x = torch.tensor(np.stack([tokens[s:s + block_size] for s in starts]),
                         dtype=torch.long, device=device)
        y = torch.tensor(np.stack([tokens[s + 1:s + 1 + block_size] for s in starts]),
                         dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    losses = np.array(losses)
    mean_ce = float(losses.mean())
    return {
        "ce_mean": mean_ce,
        "ce_std": float(losses.std()),
        "ppl": float(np.exp(mean_ce)),
        "n_batches": n_batches,
        "n_tokens": int(n_batches * batch_size * block_size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_MAP))
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--corpora", nargs="+",
                        default=["owt_val", "wikitext103", "lambada", "c4_small"])
    args = parser.parse_args()

    print(f"=== Cross-corpus eval: variant={args.variant} ===", flush=True)
    t0 = time.time()

    print(f"[1/{len(args.corpora)+1}] Loading model {args.variant} ...", flush=True)
    model, recorded_val_loss = load_model(args.variant, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded ({n_params:,} params, recorded val_loss={recorded_val_loss})", flush=True)

    results = {
        "variant": args.variant,
        "n_params": n_params,
        "recorded_val_loss": recorded_val_loss,
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "n_batches": N_BATCHES,
        "corpora": {},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for i, corpus in enumerate(args.corpora):
        print(f"\n[{i+2}/{len(args.corpora)+1}] {corpus} ...", flush=True)
        tt = time.time()
        try:
            if corpus == "owt_val":
                tokens = load_owt_val_tokens()
            else:
                tokens = tokenize_corpus(corpus)
            print(f"  {corpus}: {len(tokens):,} tokens; eval ...", flush=True)
            results["corpora"][corpus] = compute_ppl(model, tokens, device=args.device)
            results["corpora"][corpus]["wall_seconds"] = time.time() - tt
            r = results["corpora"][corpus]
            print(f"  {corpus}: ce={r['ce_mean']:.4f}  ppl={r['ppl']:.2f}  ({r['wall_seconds']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  {corpus}: FAILED ({type(e).__name__}: {e})", flush=True)
            results["corpora"][corpus] = {"error": f"{type(e).__name__}: {e}"}
        # Incremental save after each corpus
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    results["total_seconds"] = time.time() - t0
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== DONE: {args.variant}  total {results['total_seconds']:.1f}s ===")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
