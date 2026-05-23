"""Vanilla nanoGPT-style Transformer baseline (GD + Lie-Trotter).

This is the standard pre-norm Transformer block:
    x = x + Attn(LN(x))
    x = x + MLP(LN(x))

In the YuriiFormer optimizer-as-architecture framing, this corresponds to
Gradient Descent with Lie-Trotter splitting (i.e. the nanoGPT baseline).
Same architecture as YuriiFormer/TMMFormer but with no velocity stream and
no learned per-layer scalars.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, MLP


class VanillaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x


class VanillaTransformer(nn.Module):
    """Vanilla pre-norm Transformer (small config: 12L/12H/768d)."""

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [VanillaBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        self.final_ln = nn.LayerNorm(d_model, bias=False)
        # Weight tying with tok_emb

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
            torch.nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(layer.mlp.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)
        return logits
