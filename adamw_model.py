"""AdamWFormer: AdamW-inspired adaptive-update transformer with decoupled state decay."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, MLP


class AdamWLTBlock(nn.Module):
    """AdamW+Lie-Trotter block with decoupled state decay.

    Compared to AdamLTBlock, each half-step applies a learned decay to the
    state stream before adding the adaptive update:

    Attention half-step:
        g_attn   = Attn(LN(x))
        m_half   = beta1_a * m + (1 - beta1_a) * g_attn
        s_half   = beta2_a * s + (1 - beta2_a) * g_attn^2
        x_half   = (1 - lambda_a) * x + gamma_a * LN_u(m_half / (sqrt(s_half) + eps))

    MLP half-step: same structure with g_mlp = MLP(LN(x_half)).

    lambda_a, lambda_m are learned per-layer scalars in (0, 1) via sigmoid,
    initialized near 0 so the model starts close to AdamFormer.
    """

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        # Pre-norm for oracles
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        # LayerNorm on adaptive update
        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        # Learned scalars (8 per layer)
        # beta1, beta2 in (0,1) via sigmoid; gamma > 0 via softplus
        # lambda in (0,1) via sigmoid, initialized near 0 (decoupled weight decay)
        self.beta1_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta2_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.lambda_attn_raw = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

        self.beta1_mlp_raw = nn.Parameter(torch.zeros(1))
        self.beta2_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))
        self.lambda_mlp_raw = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

    def forward(
        self, x: torch.Tensor, m: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta1_a = torch.sigmoid(self.beta1_attn_raw)
        beta2_a = torch.sigmoid(self.beta2_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        lambda_a = torch.sigmoid(self.lambda_attn_raw)

        beta1_m = torch.sigmoid(self.beta1_mlp_raw)
        beta2_m = torch.sigmoid(self.beta2_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)
        lambda_m = torch.sigmoid(self.lambda_mlp_raw)

        # ── Attention half-step ─────────────────────────────────────────
        g_attn = self.attn(self.ln_attn(x))
        m_half = beta1_a * m + (1 - beta1_a) * g_attn
        s_half = beta2_a * s + (1 - beta2_a) * g_attn.square()
        update_attn = self.ln_update_attn(m_half / (s_half.sqrt() + self.eps))
        x_half = (1 - lambda_a) * x + gamma_a * update_attn

        # ── MLP half-step ───────────────────────────────────────────────
        g_mlp = self.mlp(self.ln_mlp(x_half))
        m_next = beta1_m * m_half + (1 - beta1_m) * g_mlp
        s_next = beta2_m * s_half + (1 - beta2_m) * g_mlp.square()
        update_mlp = self.ln_update_mlp(m_next / (s_next.sqrt() + self.eps))
        x_next = (1 - lambda_m) * x_half + gamma_m * update_mlp

        return x_next, m_next, s_next


class AdamWFormer(nn.Module):
    """AdamW+Lie-Trotter AdamWFormer (small config: 12L/12H/768d)."""

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

        # Main embeddings (state stream)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 1st moment embeddings
        self.m_tok_emb = nn.Embedding(vocab_size, d_model)
        self.m_pos_emb = nn.Embedding(max_seq_len, d_model)

        # 2nd moment: initialized to ones

        # Transformer layers
        self.layers = nn.ModuleList(
            [AdamWLTBlock(d_model, n_heads) for _ in range(n_layers)]
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
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        # Initialize state, 1st moment, and 2nd moment
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        m = self.m_tok_emb(input_ids) + self.m_pos_emb(pos)
        s = torch.ones_like(x)

        # Apply AdamW+Lie-Trotter layers
        for layer in self.layers:
            x, m, s = layer(x, m, s)

        # Final LN + weight-tied output projection
        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)

        return logits
