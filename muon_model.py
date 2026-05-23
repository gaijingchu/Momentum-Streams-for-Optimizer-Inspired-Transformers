"""MuonFormer: Orthogonalized-momentum transformer architecture.

Each layer applies the Muon optimizer template (momentum EMA followed by
Newton-Schulz orthogonalization) twice via Lie-Trotter splitting — once
with the attention oracle and once with the MLP oracle.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, MLP


# ── Newton-Schulz orthogonalization ────────────────────────────────────────

# Quintic coefficients (Bernstein & Shneydor, 2024)
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz_per_token(M: torch.Tensor, n_heads: int, K: int = 5) -> torch.Tensor:
    """Per-token head-wise Newton-Schulz orthogonalization.

    Each token's d-dim representation is reshaped as an (n_heads, head_dim)
    matrix, and the quintic Newton-Schulz polar-factor iteration is applied
    independently to each token's matrix.  This is causal by construction
    (no cross-token mixing) and orthogonalizes the per-head contributions
    within each token.

    For a short-and-fat matrix M[t] ∈ R^{n_heads × head_dim} with
    n_heads < head_dim, the polar factor has n_heads orthonormal rows in
    the head_dim subspace — every head gets a unit-norm, mutually orthogonal
    update within the token.

    Args:
        M: input tensor of shape (B, T, d) where d = n_heads * head_dim
        n_heads: number of heads to factor d into
        K: number of quintic Newton-Schulz iterations (default 5)

    Returns:
        tensor of shape (B, T, d) with per-token head-space orthogonalized updates
    """
    B, T, d = M.shape
    head_dim = d // n_heads
    # Reshape each token's d-dim vector as an (n_heads, head_dim) matrix
    Y = M.view(B, T, n_heads, head_dim)
    # Per-token Frobenius normalization (over the 2D sub-matrix axes)
    norm = Y.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    Y = Y / norm

    I = torch.eye(head_dim, device=Y.device, dtype=Y.dtype)
    for _ in range(K):
        A = Y.transpose(-2, -1) @ Y          # (B, T, head_dim, head_dim)
        Y = Y @ (_NS_A * I + _NS_B * A + _NS_C * A @ A)  # (B, T, n_heads, head_dim)

    return Y.reshape(B, T, d)


# ── MuonFormer block ──────────────────────────────────────────────────────

class MuonLTBlock(nn.Module):
    """Muon+Lie-Trotter block with state/momentum streams.

    Per-substep (attention, then MLP):
        g     = Oracle(LN(x))
        m_new = beta * m + (1 - beta) * g                    (momentum EMA)
        u     = NS_K(m_new as (n_heads, head_dim) per token)  (head-wise orthogonalize)
        x_new = x + gamma * LN_u(u)                          (state update)

    The Newton-Schulz step is applied to each token's (n_heads, head_dim)
    matrix independently — making the operation causal and giving each
    attention head a unit-norm, mutually-orthogonal contribution to the
    token's update.
    """

    def __init__(self, d_model: int, n_heads: int, ns_iters: int = 5):
        super().__init__()
        self.n_heads = n_heads
        self.ns_iters = ns_iters

        # Pre-norm for oracles
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        # Update LayerNorm (stabilizes orthogonalized output before residual add)
        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        # Learned scalars (4 per layer): beta, gamma per substep
        # beta in (0,1) via sigmoid; gamma > 0 via softplus
        self.beta_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta_a = torch.sigmoid(self.beta_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        beta_m = torch.sigmoid(self.beta_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        # ── Attention substep ───────────────────────────────────────────
        g_attn = self.attn(self.ln_attn(x))
        m_half = beta_a * m + (1 - beta_a) * g_attn
        u_attn = newton_schulz_per_token(m_half, self.n_heads, K=self.ns_iters)
        x_half = x + gamma_a * self.ln_update_attn(u_attn)

        # ── MLP substep ─────────────────────────────────────────────────
        g_mlp = self.mlp(self.ln_mlp(x_half))
        m_next = beta_m * m_half + (1 - beta_m) * g_mlp
        u_mlp = newton_schulz_per_token(m_next, self.n_heads, K=self.ns_iters)
        x_next = x_half + gamma_m * self.ln_update_mlp(u_mlp)

        return x_next, m_next


# ── MuonFormer top-level model ────────────────────────────────────────────

class MuonFormer(nn.Module):
    """Muon+Lie-Trotter MuonFormer (small config: 12L/12H/768d)."""

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

        # Main embeddings (state stream)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Momentum embeddings (separate learned initialization)
        self.m_tok_emb = nn.Embedding(vocab_size, d_model)
        self.m_pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [MuonLTBlock(d_model, n_heads, ns_iters=ns_iters) for _ in range(n_layers)]
        )

        # Final LayerNorm before output projection
        self.final_ln = nn.LayerNorm(d_model, bias=False)

        # Weight tying: output projection shares weights with tok_emb

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following nanoGPT conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections by 1/sqrt(2*n_layers)
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
            torch.nn.init.normal_(
                layer.attn.out_proj.weight, mean=0.0, std=0.02 * scale
            )
            torch.nn.init.normal_(layer.mlp.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        # Initialize state and momentum
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        m = self.m_tok_emb(input_ids) + self.m_pos_emb(pos)

        # Apply Muon+Lie-Trotter layers
        for layer in self.layers:
            x, m = layer(x, m)

        # Final LN + weight-tied output projection
        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)

        return logits
