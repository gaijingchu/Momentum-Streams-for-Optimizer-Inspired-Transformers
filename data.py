"""TinyStories data pipeline: tokenization, caching, and epoch-reordered dataloader.

Uses the HuggingFace datasets library to load roneneldan/TinyStories with proper
train/validation splits. Tokenizes document-by-document with GPT-2 BPE, appending
an EOT token after each document, as described in the paper.
"""

import os
import numpy as np
import tiktoken
import torch

DATA_DIR = "data/TinyStories" # TODO: change to the actual data directory   
TRAIN_CACHE = os.path.join(DATA_DIR, "train_tokens_v2.npy")
VAL_CACHE = os.path.join(DATA_DIR, "val_tokens_v2.npy")


def tokenize_split(split: str, cache_path: str) -> np.ndarray:
    """Tokenize a HuggingFace dataset split doc-by-doc with GPT-2 BPE.

    Each document is tokenized independently and an EOT token is appended.
    This matches the paper's description: "append an end-of-text token between
    documents/stories."
    """
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}")
        return np.load(cache_path)

    from datasets import load_dataset
    print(f"Loading roneneldan/TinyStories split='{split}' from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split=split)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    print(f"Tokenizing {len(ds):,} documents...")
    all_tokens = []
    for i, ex in enumerate(ds):
        tokens = enc.encode(ex["text"])
        all_tokens.extend(tokens)
        all_tokens.append(eot)
        if (i + 1) % 500000 == 0:
            print(f"  {i+1:,} / {len(ds):,} docs tokenized")

    tokens_array = np.array(all_tokens, dtype=np.uint16)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, tokens_array)
    print(f"Saved {len(tokens_array):,} tokens to {cache_path}")
    return tokens_array


def load_tokens(split: str = "train") -> np.ndarray:
    """Load tokenized data for the given split."""
    if split == "train":
        return tokenize_split("train", TRAIN_CACHE)
    else:
        return tokenize_split("validation", VAL_CACHE)


class TinyStoriesDataset:
    """Deterministic epoch-reordered dataloader for TinyStories.

    Creates non-overlapping blocks of (block_size+1) tokens.
    Between epochs, shifts block boundaries by a seeded offset and reshuffles.
    """

    def __init__(self, tokens: np.ndarray, block_size: int = 1024, seed: int = 42, device: str = "cuda"):
        self.tokens = tokens
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self._build_blocks()

    def _build_blocks(self):
        """Build non-overlapping blocks for the current epoch."""
        # Shift block boundaries by seeded offset (0 for first epoch)
        offset = self.rng.integers(0, self.block_size) if self.epoch > 0 else 0
        # Each block needs block_size+1 tokens (input + 1 target)
        n_blocks = (len(self.tokens) - offset - 1) // self.block_size
        self.block_starts = offset + np.arange(n_blocks) * self.block_size
        self.rng.shuffle(self.block_starts)
        self.block_idx = 0

    def _next_epoch(self):
        self.epoch += 1
        self._build_blocks()

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of (input_ids, targets), each shape (batch_size, block_size)."""
        xs, ys = [], []
        for _ in range(batch_size):
            if self.block_idx >= len(self.block_starts):
                self._next_epoch()
            start = self.block_starts[self.block_idx]
            chunk = self.tokens[start : start + self.block_size + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self.block_idx += 1

        x = torch.tensor(np.stack(xs), dtype=torch.long, device=self.device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=self.device)
        return x, y


class ValidationDataset:
    """Fixed-order validation dataset. Iterates through blocks sequentially."""

    def __init__(self, tokens: np.ndarray, block_size: int = 1024, device: str = "cuda"):
        self.tokens = tokens
        self.block_size = block_size
        self.device = device
        n_blocks = (len(tokens) - 1) // block_size
        self.block_starts = np.arange(n_blocks) * block_size
        self.block_idx = 0

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of (input_ids, targets). Wraps around if needed."""
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
