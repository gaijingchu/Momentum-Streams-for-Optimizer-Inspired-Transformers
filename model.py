"""YuriiFormer: Nesterov+Lie-Trotter transformer architecture."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class NesterovLTBlock(nn.Module):
    """Nesterov+Lie-Trotter block with dual state/velocity streams.

    Per-layer update:
        x_in = x + mu_a * v                    (lookahead for attn)
        v_half = LN_v(beta_a * v + gamma_a * Attn(LN(x_in)))
        x_half = x + v_half
        x_in2 = x_half + mu_m * v_half         (lookahead for MLP)
        v_next = LN_v(beta_m * v_half + gamma_m * MLP(LN(x_in2)))
        x_next = x_half + v_next
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Pre-norm for oracles
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)
        # Velocity LayerNorm
        self.ln_v_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_v_mlp = nn.LayerNorm(d_model, bias=False)
        # Learned scalars (6 per layer)
        # mu, beta in (0,1) via sigmoid; gamma > 0 via softplus
        self.mu_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.mu_mlp_raw = nn.Parameter(torch.zeros(1))
        self.beta_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_a = torch.sigmoid(self.mu_attn_raw)
        beta_a = torch.sigmoid(self.beta_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        mu_m = torch.sigmoid(self.mu_mlp_raw)
        beta_m = torch.sigmoid(self.beta_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        # Attention substep
        x_in = x + mu_a * v
        attn_out = self.attn(self.ln_attn(x_in))
        v_half = self.ln_v_attn(beta_a * v + gamma_a * attn_out)
        x_half = x + v_half

        # MLP substep
        x_in2 = x_half + mu_m * v_half
        mlp_out = self.mlp(self.ln_mlp(x_in2))
        v_next = self.ln_v_mlp(beta_m * v_half + gamma_m * mlp_out)
        x_next = x_half + v_next

        return x_next, v_next


class YuriiFormer(nn.Module):
    """Nesterov+Lie-Trotter YuriiFormer (small config: 12L/12H/768d)."""

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

        # Main embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Velocity embeddings (separate from main)
        self.vel_tok_emb = nn.Embedding(vocab_size, d_model)
        self.vel_pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([NesterovLTBlock(d_model, n_heads) for _ in range(n_layers)])

        # Final LayerNorm before output projection
        self.final_ln = nn.LayerNorm(d_model, bias=False)

        # Weight tying: output projection shares weights with tok_emb
        # (no separate lm_head parameter)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following nanoGPT conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections by 1/sqrt(2*n_layers)
        # In YuriiFormer, out_proj and w2 outputs are scaled by gamma and
        # then pass through LN_v, but we still apply the nanoGPT convention
        # since the paper uses "the same attention/MLP modules"
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
            torch.nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
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

        # Initialize state and velocity
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        v = self.vel_tok_emb(input_ids) + self.vel_pos_emb(pos)

        # Apply Nesterov+Lie-Trotter layers
        for layer in self.layers:
            x, v = layer(x, v)

        # Final LN + weight-tied output projection
        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)

        return logits
