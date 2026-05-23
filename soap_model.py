"""SOAPFormer: Per-token head-wise Kronecker-preconditioned transformer.

Each layer applies a SOAP-inspired template twice via Lie-Trotter splitting.
The first moment and right covariance are tracked per-token in the
(n_heads, head_dim) head space — causal by construction, with no cross-token
or cross-batch mixing.

The right covariance R[t] ∈ R^{head_dim × head_dim} captures cross-dimension
correlations within each token's head space.  Preconditioning with R^{-1/2}
(via eigendecomposition) decorrelates and equalizes per-head-dim update rates
— the same advantage SOAP has over Adam, applied per-token.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, MLP


# ── Per-token head-wise inverse square root ───────────────────────────────

def newton_inv_sqrt(R: torch.Tensor, K: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Compute R^{-1/2} via Newton's iteration for batched symmetric PD matrices.

    Uses the iteration X_{k+1} = 0.5 * X_k * (3I - X_k @ R_norm @ X_k),
    which converges to R_norm^{-1/2}.  This is purely matrix multiplications
    — no eigendecomposition or SVD — so it handles degenerate eigenvalues
    without issue.

    Args:
        R: symmetric PD tensor of shape (..., d, d)
        K: number of Newton iterations (default 10)
        eps: regularization added to diagonal

    Returns:
        R_inv_sqrt: shape (..., d, d), approximating R^{-1/2}
    """
    d = R.shape[-1]
    I = torch.eye(d, device=R.device, dtype=R.dtype)

    # Regularize
    R_reg = R + eps * I

    # Normalize R so eigenvalues are near 1 for fast convergence
    # Use Frobenius norm as a scale estimate
    norm = R_reg.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    R_norm = R_reg / norm

    # Newton iteration: X → 0.5 * X @ (3I - X @ R_norm @ X)
    # Converges to R_norm^{-1/2} from X_0 = I
    X = I.expand_as(R_norm).clone()
    for _ in range(K):
        XRX = X @ R_norm @ X
        X = 0.5 * X @ (3.0 * I - XRX)

    # Undo normalization: R^{-1/2} = R_norm^{-1/2} / sqrt(norm)
    return X / norm.sqrt()


# ── SOAPFormer block ──────────────────────────────────────────────────────

class SOAPLTBlock(nn.Module):
    """SOAP+Lie-Trotter block with per-token head-wise preconditioning.

    Per-substep (attention, then MLP):
        g     = Oracle(LN(x))                          (B, T, d)
        G     = g.view(B, T, n_heads, head_dim)        reshape to head space
        M_new = β₁ * M + (1 - β₁) * G                 1st moment EMA  (B, T, H, D)
        R_new = β_R * R + (1 - β_R) * G^T @ G          right cov EMA   (B, T, D, D)
        U     = M_new @ R_new^{-1/2}                   preconditioned   (B, T, H, D)
        u     = LN_u(U.reshape(B, T, d))
        x_new = x + γ * u                              state update
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norm for oracles
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        # Update LayerNorm (stabilizes preconditioned update before residual add)
        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        # Learned scalars (6 per layer): beta1, betaR, gamma per substep
        # beta1, betaR in (0,1) via sigmoid; gamma > 0 via softplus
        self.beta1_attn_raw = nn.Parameter(torch.zeros(1))
        self.betaR_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta1_mlp_raw = nn.Parameter(torch.zeros(1))
        self.betaR_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: hidden state     (B, T, d)
            m: 1st moment       (B, T, n_heads, head_dim)
            R: right covariance (B, T, head_dim, head_dim)
        Returns:
            (x_next, m_next, R_next) with same shapes
        """
        beta1_a = torch.sigmoid(self.beta1_attn_raw)
        betaR_a = torch.sigmoid(self.betaR_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        beta1_m = torch.sigmoid(self.beta1_mlp_raw)
        betaR_m = torch.sigmoid(self.betaR_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        B, T, d = x.shape
        H, D = self.n_heads, self.head_dim

        # ── Attention substep ───────────────────────────────────────────
        g_attn = self.attn(self.ln_attn(x))                     # (B, T, d)
        G_attn = g_attn.view(B, T, H, D)                        # (B, T, H, D)

        m_half = beta1_a * m + (1 - beta1_a) * G_attn            # (B, T, H, D)
        GtG_attn = G_attn.transpose(-2, -1) @ G_attn             # (B, T, D, D)
        R_half = betaR_a * R + (1 - betaR_a) * GtG_attn          # (B, T, D, D)

        R_inv_sqrt = newton_inv_sqrt(R_half)                        # (B, T, D, D)
        U_attn = m_half @ R_inv_sqrt                              # (B, T, H, D)
        update_attn = self.ln_update_attn(U_attn.reshape(B, T, d))
        x_half = x + gamma_a * update_attn

        # ── MLP substep ─────────────────────────────────────────────────
        g_mlp = self.mlp(self.ln_mlp(x_half))                    # (B, T, d)
        G_mlp = g_mlp.view(B, T, H, D)                           # (B, T, H, D)

        m_next = beta1_m * m_half + (1 - beta1_m) * G_mlp        # (B, T, H, D)
        GtG_mlp = G_mlp.transpose(-2, -1) @ G_mlp                # (B, T, D, D)
        R_next = betaR_m * R_half + (1 - betaR_m) * GtG_mlp      # (B, T, D, D)

        R_inv_sqrt = newton_inv_sqrt(R_next)                        # (B, T, D, D)
        U_mlp = m_next @ R_inv_sqrt                               # (B, T, H, D)
        update_mlp = self.ln_update_mlp(U_mlp.reshape(B, T, d))
        x_next = x_half + gamma_m * update_mlp

        return x_next, m_next, R_next


# ── SOAPFormer top-level model ────────────────────────────────────────────

class SOAPFormer(nn.Module):
    """SOAP+Lie-Trotter SOAPFormer (small config: 12L/12H/768d)."""

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
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Main embeddings (state stream)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 1st moment embeddings (separate learned initialization)
        self.m_tok_emb = nn.Embedding(vocab_size, d_model)
        self.m_pos_emb = nn.Embedding(max_seq_len, d_model)

        # Right covariance R_0 = I_{head_dim} per token (identity;
        # represents isotropic prior in head-dim space)

        # Transformer layers
        self.layers = nn.ModuleList(
            [SOAPLTBlock(d_model, n_heads) for _ in range(n_layers)]
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

        # Initialize state, 1st moment (in head space), and right covariance
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        m_flat = self.m_tok_emb(input_ids) + self.m_pos_emb(pos)
        m = m_flat.view(B, T, self.n_heads, self.head_dim)

        # R_0 = I_{head_dim} broadcast to (B, T, head_dim, head_dim)
        R = torch.eye(
            self.head_dim, device=input_ids.device, dtype=x.dtype
        ).expand(B, T, -1, -1).contiguous()

        # Apply SOAP+Lie-Trotter layers
        for layer in self.layers:
            x, m, R = layer(x, m, R)

        # Final LN + weight-tied output projection
        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)

        return logits
