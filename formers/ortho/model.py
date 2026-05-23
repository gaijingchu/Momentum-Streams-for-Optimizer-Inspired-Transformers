"""OrthoFormer: per-token head-wise Newton-Schulz orthogonalization, no momentum.

Each oracle output is reshaped to (B, T, n_heads, head_dim) and orthogonalized
**per-token** via the quintic Newton-Schulz polar-factor iteration (matching
MuonFormer's head-wise NS — causal by construction, no cross-token mixing).

Per-substep with oracle O in {Attn, MLP}:
    g     = O(LN(x))                                   # (B, T, d)
    u     = NS_K_per_token(g, n_heads)                 # (B, T, d), causal
    x_new = x + gamma * LN_u(u)

Sits in the factorial ablation as the "No-momentum x Spectral" cell —
MuonFormer with the EMA momentum buffer removed. 2 learned scalars per layer
(gamma per substep), no auxiliary stream.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from formers.yurii.model import CausalSelfAttention, MLP


# Quintic Newton-Schulz coefficients (Bernstein & Shneydor, 2024)
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz_per_token(M: torch.Tensor, n_heads: int, K: int = 5) -> torch.Tensor:
    """Per-token head-wise Newton-Schulz orthogonalization.

    Each token's d-dim representation is reshaped as an (n_heads, head_dim)
    matrix, and the quintic Newton-Schulz polar-factor iteration is applied
    independently to each token's matrix. Causal by construction.

    Args:
        M: (B, T, d) where d = n_heads * head_dim
        n_heads: number of heads to factor d into
        K: number of NS iterations
    Returns:
        (B, T, d) with each token's head-space matrix orthogonalized.
    """
    B, T, d = M.shape
    head_dim = d // n_heads
    Y = M.view(B, T, n_heads, head_dim)
    norm = Y.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    Y = Y / norm

    I = torch.eye(head_dim, device=Y.device, dtype=Y.dtype)
    for _ in range(K):
        A = Y.transpose(-2, -1) @ Y                       # (B, T, head_dim, head_dim)
        Y = Y @ (_NS_A * I + _NS_B * A + _NS_C * A @ A)   # (B, T, n_heads, head_dim)

    return Y.reshape(B, T, d)


class OrthoBlock(nn.Module):
    """Orthogonalized + Lie-Trotter block, no momentum stream.

    Per-substep with oracle O in {Attn, MLP}:
        g     = O(LN(x))
        u     = NS_per_token(g, n_heads)
        x_new = x + gamma * LN_u(u)

    2 learned scalars per layer (gamma per substep). MuonLTBlock with the
    momentum EMA stripped: g goes directly into NS instead of the EMA m_new.
    """

    def __init__(self, d_model: int, n_heads: int, ns_iters: int = 5):
        super().__init__()
        self.n_heads = n_heads
        self.ns_iters = ns_iters

        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma_a = F.softplus(self.gamma_attn_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        # Attention substep
        g_attn = self.attn(self.ln_attn(x))
        u_attn = newton_schulz_per_token(g_attn, self.n_heads, K=self.ns_iters)
        x_half = x + gamma_a * self.ln_update_attn(u_attn)

        # MLP substep
        g_mlp = self.mlp(self.ln_mlp(x_half))
        u_mlp = newton_schulz_per_token(g_mlp, self.n_heads, K=self.ns_iters)
        x_next = x_half + gamma_m * self.ln_update_mlp(u_mlp)

        return x_next


class OrthoFormer(nn.Module):
    """Per-token head-wise NS + Lie-Trotter OrthoFormer (small config: 12L/12H/768d)."""

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        ns_iters: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # No m stream (no momentum) — only state stream x.

        self.layers = nn.ModuleList(
            [OrthoBlock(d_model, n_heads, ns_iters=ns_iters) for _ in range(n_layers)]
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

        for layer in self.layers:
            x = layer(x)

        x = self.final_ln(x)
        return F.linear(x, self.tok_emb.weight)
