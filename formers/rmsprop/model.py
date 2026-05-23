"""RMSPropFormer: per-coordinate preconditioning + Lie-Trotter, no momentum.

RMSProp (Tieleman & Hinton 2012) is Adam minus the first-moment EMA: it tracks
only the second moment and divides each coordinate by its inverse square root.
    s_{l+1} = beta2 * s_l + (1 - beta2) * g_l ** 2
    x_{l+1} = x_l + gamma * LN_u(g_l / (sqrt(s_{l+1}) + eps))

Sits in the factorial ablation as the "No-momentum x Per-coord" cell. Isolates
per-coordinate preconditioning at fixed (no) momentum, controlling for
AdamFormer's first-moment buffer.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from formers.yurii.model import CausalSelfAttention, MLP


class RMSPropLTBlock(nn.Module):
    """RMSProp + Lie-Trotter block with state/2nd-moment streams.

    Per-substep with oracle O in {Attn, MLP}:
        g       = O(LN(x))
        s_new   = beta2 * s + (1 - beta2) * g ** 2
        update  = LN_u(g / (sqrt(s_new) + eps))
        x_new   = x + gamma * update

    4 learned scalars per layer (beta2, gamma per substep). AdamFormer with
    the m stream removed.
    """

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        self.beta2_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta2_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta2_a = torch.sigmoid(self.beta2_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        beta2_m = torch.sigmoid(self.beta2_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        # Attention substep
        g_attn = self.attn(self.ln_attn(x))
        s_half = beta2_a * s + (1 - beta2_a) * g_attn.square()
        update_attn = self.ln_update_attn(g_attn / (s_half.sqrt() + self.eps))
        x_half = x + gamma_a * update_attn

        # MLP substep
        g_mlp = self.mlp(self.ln_mlp(x_half))
        s_next = beta2_m * s_half + (1 - beta2_m) * g_mlp.square()
        update_mlp = self.ln_update_mlp(g_mlp / (s_next.sqrt() + self.eps))
        x_next = x_half + gamma_m * update_mlp

        return x_next, s_next


class RMSPropFormer(nn.Module):
    """RMSProp + Lie-Trotter RMSPropFormer (small config: 12L/12H/768d)."""

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

        # No m stream (no momentum). s stream initialized to ones, no embedding.

        self.layers = nn.ModuleList(
            [RMSPropLTBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        self.final_ln = nn.LayerNorm(d_model, bias=False)

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
        s = torch.ones_like(x)

        for layer in self.layers:
            x, s = layer(x, s)

        x = self.final_ln(x)
        return F.linear(x, self.tok_emb.weight)
