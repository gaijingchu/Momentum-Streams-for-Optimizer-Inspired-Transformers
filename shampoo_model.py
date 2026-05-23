"""ShampooFormer: per-token head-wise Kronecker-preconditioned, no momentum.

Each layer maintains a per-token right covariance R[t] of shape
(head_dim, head_dim). At each substep, R is updated by EMA of G^T @ G
(per-token), and the gradient is preconditioned by R^{-1/2} computed via
Newton's iteration. Per-token by construction — causal, no cross-token
or cross-batch mixing.

Per-substep with oracle O in {Attn, MLP}:
    g     = O(LN(x))                                   # (B, T, d)
    G     = g.view(B, T, n_heads, head_dim)            # head-space
    R_new = beta_R * R + (1 - beta_R) * G^T @ G        # (B, T, D, D) per-token
    U     = G @ R_new^{-1/2}                           # (B, T, H, D)
    x_new = x + gamma * LN_u(U.reshape(B, T, d))

Sits in the factorial ablation as the "No-momentum x Full-matrix" cell —
SOAPFormer with the first-moment buffer removed. 4 learned scalars per layer
(beta_R, gamma per substep). Per-token (D, D) R stream is heavy on memory,
so the corresponding training script reduces BATCH_SIZE to 4 with double
grad-accum, matching the SOAPFormer training recipe.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, MLP


def newton_inv_sqrt(R: torch.Tensor, K: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Compute R^{-1/2} via Newton's iteration for batched symmetric PD matrices.

    X_{k+1} = 0.5 * X_k * (3I - X_k @ R_norm @ X_k), converges to R_norm^{-1/2}.
    Pure matmul (no eigendecomposition); compile- and autograd-friendly.

    Args:
        R: SPD tensor of shape (..., d, d)
        K: number of Newton iterations
        eps: regularization added to diagonal
    Returns:
        R^{-1/2} with the same shape.
    """
    d = R.shape[-1]
    I = torch.eye(d, device=R.device, dtype=R.dtype)
    R_reg = R + eps * I

    norm = R_reg.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    R_norm = R_reg / norm

    X = I.expand_as(R_norm).clone()
    for _ in range(K):
        XRX = X @ R_norm @ X
        X = 0.5 * X @ (3.0 * I - XRX)

    return X / norm.sqrt()


class ShampooBlock(nn.Module):
    """Per-token head-wise preconditioned + Lie-Trotter block, no momentum.

    Per-substep with oracle O in {Attn, MLP}:
        g     = O(LN(x))                                   # (B, T, d)
        G     = g.view(B, T, n_heads, head_dim)
        R_new = beta_R * R + (1 - beta_R) * G^T @ G        # (B, T, D, D)
        U     = G @ R_new^{-1/2}                           # (B, T, H, D)
        x_new = x + gamma * LN_u(U.reshape(B, T, d))

    4 learned scalars per layer (beta_R, gamma per substep). SOAPLTBlock with
    the m EMA stripped: G replaces M in the preconditioned-update product.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        self.betaR_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.betaR_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, R: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        betaR_a = torch.sigmoid(self.betaR_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        betaR_m = torch.sigmoid(self.betaR_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        B, T, d = x.shape
        H, D = self.n_heads, self.head_dim

        # Attention substep
        g_attn = self.attn(self.ln_attn(x))                  # (B, T, d)
        G_attn = g_attn.view(B, T, H, D)                     # (B, T, H, D)
        GtG_attn = G_attn.transpose(-2, -1) @ G_attn         # (B, T, D, D)
        R_half = betaR_a * R + (1 - betaR_a) * GtG_attn      # (B, T, D, D)

        R_inv_sqrt = newton_inv_sqrt(R_half)                  # (B, T, D, D)
        U_attn = G_attn @ R_inv_sqrt                          # (B, T, H, D)
        update_attn = self.ln_update_attn(U_attn.reshape(B, T, d))
        x_half = x + gamma_a * update_attn

        # MLP substep
        g_mlp = self.mlp(self.ln_mlp(x_half))                 # (B, T, d)
        G_mlp = g_mlp.view(B, T, H, D)                        # (B, T, H, D)
        GtG_mlp = G_mlp.transpose(-2, -1) @ G_mlp             # (B, T, D, D)
        R_next = betaR_m * R_half + (1 - betaR_m) * GtG_mlp   # (B, T, D, D)

        R_inv_sqrt = newton_inv_sqrt(R_next)                   # (B, T, D, D)
        U_mlp = G_mlp @ R_inv_sqrt                             # (B, T, H, D)
        update_mlp = self.ln_update_mlp(U_mlp.reshape(B, T, d))
        x_next = x_half + gamma_m * update_mlp

        return x_next, R_next


class ShampooFormer(nn.Module):
    """Per-token head-wise SOAP w/o momentum (small config: 12L/12H/768d)."""

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

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # No m embedding (no momentum). R_0 = I_{head_dim} broadcast per-token.

        self.layers = nn.ModuleList(
            [ShampooBlock(d_model, n_heads) for _ in range(n_layers)]
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

        # R_0 = I_{head_dim} broadcast to (B, T, head_dim, head_dim)
        R = torch.eye(
            self.head_dim, device=input_ids.device, dtype=x.dtype
        ).expand(B, T, -1, -1).contiguous()

        for layer in self.layers:
            x, R = layer(x, R)

        x = self.final_ln(x)
        return F.linear(x, self.tok_emb.weight)
