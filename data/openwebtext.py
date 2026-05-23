"""OpenWebText data pipeline: tokenization, caching, and epoch-reordered dataloader.

Follows the paper's protocol: GPT-2 BPE tokenization, EOT between documents,
95/5 train/val split based on document index, non-overlapping 1024-token blocks.
Data is stored on compute node local storage ($CACHE/owt_data/).
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import numpy as np
import tiktoken

# Store on compute-node local storage
CACHE = os.environ.get("CACHE", "./checkpoints_cache")
DATA_DIR = os.path.join(CACHE, "owt_data")
TRAIN_CACHE = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_CACHE = os.path.join(DATA_DIR, "val_tokens.npy")


def _do_tokenize():
    """Tokenize OpenWebText on a single process and save to cache.

    Memory-efficient: writes chunks to disk then concatenates, avoiding
    holding all ~9B tokens in a Python list simultaneously.
    """
    from datasets import load_dataset
    print("Loading OpenWebText from HuggingFace (streaming)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    os.makedirs(DATA_DIR, exist_ok=True)
    CHUNK_SIZE = 200_000  # docs per chunk

    # Deterministic 95/5 split by document index (matching paper)
    rng = np.random.default_rng(seed=42)

    train_chunks, val_chunks = [], []
    train_buf, val_buf = [], []
    # Pre-generate enough random values for the split
    # OWT has ~8M docs, generate in batches
    split_batch_size = 1_000_000
    split_vals = rng.random(split_batch_size)
    split_offset = 0
    chunk_idx = 0

    print("Tokenizing documents...")
    for i, ex in enumerate(ds):
        # Get split decision
        idx_in_batch = i - split_offset
        if idx_in_batch >= len(split_vals):
            split_vals = rng.random(split_batch_size)
            split_offset = i
            idx_in_batch = 0
        is_val_doc = split_vals[idx_in_batch] < 0.05

        tokens = enc.encode(ex["text"])
        tokens.append(eot)
        if is_val_doc:
            val_buf.extend(tokens)
        else:
            train_buf.extend(tokens)

        if (i + 1) % CHUNK_SIZE == 0:
            tc = os.path.join(DATA_DIR, f"_train_chunk_{chunk_idx}.npy")
            vc = os.path.join(DATA_DIR, f"_val_chunk_{chunk_idx}.npy")
            np.save(tc, np.array(train_buf, dtype=np.uint16))
            np.save(vc, np.array(val_buf, dtype=np.uint16))
            train_chunks.append(tc)
            val_chunks.append(vc)
            train_buf, val_buf = [], []
            chunk_idx += 1
            print(f"  {i+1:,} docs tokenized (chunk {chunk_idx})")

    # Final flush
    if train_buf or val_buf:
        tc = os.path.join(DATA_DIR, f"_train_chunk_{chunk_idx}.npy")
        vc = os.path.join(DATA_DIR, f"_val_chunk_{chunk_idx}.npy")
        np.save(tc, np.array(train_buf, dtype=np.uint16))
        np.save(vc, np.array(val_buf, dtype=np.uint16))
        train_chunks.append(tc)
        val_chunks.append(vc)
        chunk_idx += 1
        print(f"  Final flush (chunk {chunk_idx}), total docs: {i+1:,}")

    del train_buf, val_buf

    # Concatenate chunks one at a time
    print("Concatenating train chunks...")
    train_tokens = np.concatenate([np.load(c) for c in train_chunks])
    np.save(TRAIN_CACHE, train_tokens)
    print(f"  Train: {len(train_tokens):,} tokens")
    del train_tokens

    print("Concatenating val chunks...")
    val_tokens = np.concatenate([np.load(c) for c in val_chunks])
    np.save(VAL_CACHE, val_tokens)
    print(f"  Val: {len(val_tokens):,} tokens")
    del val_tokens

    # Clean up chunks
    for c in train_chunks + val_chunks:
        os.remove(c)
    print(f"Saved tokenized OWT to {DATA_DIR}")


def tokenize_owt() -> tuple[np.ndarray, np.ndarray]:
    """Tokenize OpenWebText with 95/5 train/val split by document index.

    In DDP, only rank 0 tokenizes; other ranks wait for the cache files.
    """
    import time as _time

    rank = int(os.environ.get("RANK", 0))

    if not (os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE)):
        if rank == 0:
            _do_tokenize()
        else:
            # Wait for rank 0 to finish tokenization
            print(f"Rank {rank}: waiting for tokenization to complete...")
            while not (os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE)):
                _time.sleep(10)
            _time.sleep(5)  # extra wait to ensure file is fully written

    print(f"Loading cached OWT train tokens from {TRAIN_CACHE}")
    train_tokens = np.load(TRAIN_CACHE)
    print(f"Loading cached OWT val tokens from {VAL_CACHE}")
    val_tokens = np.load(VAL_CACHE)
    return train_tokens, val_tokens


def load_owt_tokens(split: str = "train") -> np.ndarray:
    """Load tokenized OWT data for the given split."""
    train_tokens, val_tokens = tokenize_owt()
    return train_tokens if split == "train" else val_tokens


class OWTDataset:
    """Deterministic epoch-reordered dataloader for OpenWebText."""

    def __init__(self, tokens: np.ndarray, block_size: int = 1024, seed: int = 42, device: str = "cuda"):
        self.tokens = tokens
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self._build_blocks()

    def _build_blocks(self):
        offset = self.rng.integers(0, self.block_size) if self.epoch > 0 else 0
        n_blocks = (len(self.tokens) - offset - 1) // self.block_size
        self.block_starts = offset + np.arange(n_blocks) * self.block_size
        self.rng.shuffle(self.block_starts)
        self.block_idx = 0

    def _next_epoch(self):
        self.epoch += 1
        self._build_blocks()

    def get_batch(self, batch_size: int) -> tuple:
        xs, ys = [], []
        for _ in range(batch_size):
            if self.block_idx >= len(self.block_starts):
                self._next_epoch()
            start = self.block_starts[self.block_idx]
            chunk = self.tokens[start : start + self.block_size + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.block_idx += 1

        import torch
        x = torch.tensor(np.stack(xs), dtype=torch.long, device=self.device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=self.device)
        return x, y


class OWTValidationDataset:
    """Fixed-order validation dataset for OpenWebText."""

    def __init__(self, tokens: np.ndarray, block_size: int = 1024, device: str = "cuda"):
        self.tokens = tokens
        self.block_size = block_size
        self.device = device
        n_blocks = (len(tokens) - 1) // block_size
        self.block_starts = np.arange(n_blocks) * block_size
        self.block_idx = 0

    def get_batch(self, batch_size: int) -> tuple:
        import torch
        xs, ys = [], []
        for _ in range(batch_size):
            if self.block_idx >= len(self.block_starts):
                self.block_idx = 0
            start = self.block_starts[self.block_idx]
            chunk = self.tokens[start : start + self.block_size + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.block_idx += 1

        x = torch.tensor(np.stack(xs), dtype=torch.long, device=self.device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=self.device)
        return x, y

    def reset(self):
        self.block_idx = 0
