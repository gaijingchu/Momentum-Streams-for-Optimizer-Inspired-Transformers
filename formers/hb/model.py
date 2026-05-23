"""HBFormer: Polyak heavy-ball + Lie-Trotter (pure-momentum, no preconditioning).

Heavy-ball (Polyak 1964) is the simplest momentum method:
    v_{l+1} = beta * v_l + gamma * g_l           (velocity update; no lookahead)
    x_{l+1} = x_l + v_{l+1}                      (iterate update)

This is YuriiFormer with the lookahead disabled (mu = 0). Sits in the factorial
ablation as the "Heavy-ball x No-precond" cell, isolating Polyak momentum vs.
Nesterov at fixed (no) preconditioning.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from formers.yurii.model import CausalSelfAttention, MLP


class HBLTBlock(nn.Module):
    """Heavy-ball + Lie-Trotter block with state/velocity streams.

    Per-substep with oracle O in {Attn, MLP}:
        g      = O(LN(x))
        v_new  = LN_v(beta * v + gamma * g)
        x_new  = x + v_new

    4 learned scalars per layer (beta, gamma per substep). Identical to
    YuriiFormer's NesterovLTBlock with the `x_in = x + mu*v` lookahead removed.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)
        self.ln_v_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_v_mlp = nn.LayerNorm(d_model, bias=False)

        self.beta_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_a = torch.sigmoid(self.beta_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        beta_m = torch.sigmoid(self.beta_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        # Attention substep (no lookahead -> oracle evaluated at x)
        attn_out = self.attn(self.ln_attn(x))
        v_half = self.ln_v_attn(beta_a * v + gamma_a * attn_out)
        x_half = x + v_half

        # MLP substep
        mlp_out = self.mlp(self.ln_mlp(x_half))
        v_next = self.ln_v_mlp(beta_m * v_half + gamma_m * mlp_out)
        x_next = x_half + v_next

        return x_next, v_next


class HBFormer(nn.Module):
    """Heavy-ball + Lie-Trotter HBFormer (small config: 12L/12H/768d)."""

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

        self.vel_tok_emb = nn.Embedding(vocab_size, d_model)
        self.vel_pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([HBLTBlock(d_model, n_heads) for _ in range(n_layers)])

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
        v = self.vel_tok_emb(input_ids) + self.vel_pos_emb(pos)

        for layer in self.layers:
            x, v = layer(x, v)

        x = self.final_ln(x)
        return F.linear(x, self.tok_emb.weight)
