"""TMMFormer: Triple-Momentum-Method + Lie-Trotter transformer architecture.

Triple Momentum Method (Van Scoy, Freeman & Lynch, IEEE L-CSS 2018) is the
fastest known first-order method for L-smooth, mu-strongly convex minimization,
matching the Drori-Teboulle / Taylor lower bound with rate (1 - sqrt(mu/L))^2.

Compared to Nesterov, TMM decouples the *gradient evaluation point* from the
*iterate update point*: it uses two separate extrapolation coefficients.
In velocity-stream form, this corresponds to having an independent learnable
scalar `nu` that controls how the new velocity is added back into the state,
distinct from `mu` which controls the lookahead used for the oracle call.

YuriiFormer (Nesterov) implicitly fixes nu = 1; TMM lets it be learned.
At initialization we set nu ~= 1 so TMMFormer reduces exactly to YuriiFormer
and any deviation is a learned strict generalization.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from formers.yurii.model import CausalSelfAttention, MLP


# softplus(NU_INIT_RAW) ~= 1.0  =>  NU_INIT_RAW = log(e - 1) ~= 0.5413
NU_INIT_RAW = 0.5413248


class TMMLTBlock(nn.Module):
    """TMM + Lie-Trotter block with state/velocity streams.

    Per-substep with oracle O in {Attn, MLP}:
        x_eval = x + mu * v                       # lookahead for gradient eval
        g      = O(LN(x_eval))
        v_new  = LN_v(beta * v + gamma * g)       # velocity update
        x_new  = x + nu * v_new                   # iterate update (TMM allows nu != 1)

    Each substep has 4 learned scalars (mu, beta, gamma, nu); two substeps per
    layer give 8 scalars total. YuriiFormer is the special case nu = 1.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)
        # Velocity LayerNorm
        self.ln_v_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_v_mlp = nn.LayerNorm(d_model, bias=False)

        # 8 learned scalars per layer
        # mu, beta in (0,1) via sigmoid; gamma, nu > 0 via softplus
        self.mu_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.nu_attn_raw = nn.Parameter(torch.full((1,), NU_INIT_RAW))
        self.mu_mlp_raw = nn.Parameter(torch.zeros(1))
        self.beta_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))
        self.nu_mlp_raw = nn.Parameter(torch.full((1,), NU_INIT_RAW))

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_a = torch.sigmoid(self.mu_attn_raw)
        beta_a = torch.sigmoid(self.beta_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        nu_a = F.softplus(self.nu_attn_raw)
        mu_m = torch.sigmoid(self.mu_mlp_raw)
        beta_m = torch.sigmoid(self.beta_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)
        nu_m = F.softplus(self.nu_mlp_raw)

        # ── Attention substep ─────────────────────────────────────────
        x_in = x + mu_a * v
        attn_out = self.attn(self.ln_attn(x_in))
        v_half = self.ln_v_attn(beta_a * v + gamma_a * attn_out)
        x_half = x + nu_a * v_half

        # ── MLP substep ───────────────────────────────────────────────
        x_in2 = x_half + mu_m * v_half
        mlp_out = self.mlp(self.ln_mlp(x_in2))
        v_next = self.ln_v_mlp(beta_m * v_half + gamma_m * mlp_out)
        x_next = x_half + nu_m * v_next

        return x_next, v_next


class TMMFormer(nn.Module):
    """Triple-Momentum + Lie-Trotter TMMFormer (small config: 12L/12H/768d)."""

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        vel_init: str = "embed",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Main embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Velocity stream initialization:
        #   "embed": learned separate velocity embeddings (default; ~+39M params)
        #   "zero" : v starts at 0 (conventional momentum init); no extra params,
        #            shrinks TMMFormer to the Vanilla parameter budget (~124M).
        self.vel_init = vel_init
        if vel_init == "embed":
            self.vel_tok_emb = nn.Embedding(vocab_size, d_model)
            self.vel_pos_emb = nn.Embedding(max_seq_len, d_model)
        elif vel_init != "zero":
            raise ValueError(f"vel_init must be 'embed' or 'zero', got {vel_init!r}")

        self.layers = nn.ModuleList(
            [TMMLTBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        self.final_ln = nn.LayerNorm(d_model, bias=False)
        # Weight tying with tok_emb (no separate lm_head parameter)

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
        if self.vel_init == "embed":
            v = self.vel_tok_emb(input_ids) + self.vel_pos_emb(pos)
        else:
            v = torch.zeros_like(x)

        for layer in self.layers:
            x, v = layer(x, v)

        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)
        return logits
