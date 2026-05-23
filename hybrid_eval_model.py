"""LM wrapper for hybrid inference: a chosen subset of layers use the full
Nesterov/TMM dynamics (velocity stream active), the rest use Vanilla residual
(x = x + Attn(LN(x)); x = x + MLP(LN(x))).

Two selection modes:
  * cutoff: layers 0..cutoff-1 use Nesterov/TMM, layers cutoff..n-1 use Vanilla.
  * explicit set: pass an iterable of layer indices (e.g. {0, 1, 11}) that
    should use the Nesterov/TMM dynamics; all other layers use Vanilla.

In the explicit-set "sandwich" mode the velocity stream v is carried unchanged
through Vanilla layers, so a late Nesterov/TMM layer still receives a v
computed by earlier Nesterov/TMM layers.

No retraining — same attention/MLP weights, only the forward changes.
"""

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
import tiktoken

from lm_eval.api.model import TemplateLM
from tqdm import tqdm


def hybrid_forward(model, input_ids, cutoff_layer):
    """Layers 0..cutoff-1 use Nesterov/TMM, layers cutoff..n-1 use Vanilla
    (velocity discarded past the cutoff)."""
    T = input_ids.shape[1]
    pos = torch.arange(T, device=input_ids.device)

    x = model.tok_emb(input_ids) + model.pos_emb(pos)
    v = model.vel_tok_emb(input_ids) + model.vel_pos_emb(pos)

    for i, layer in enumerate(model.layers):
        if i < cutoff_layer:
            x, v = layer(x, v)
        else:
            x = x + layer.attn(layer.ln_attn(x))
            x = x + layer.mlp(layer.ln_mlp(x))

    x = model.final_ln(x)
    logits = F.linear(x, model.tok_emb.weight)
    return logits


def hybrid_forward_set(model, input_ids, nesterov_layers: set):
    """Layers whose index is in nesterov_layers use Nesterov/TMM dynamics;
    all others use Vanilla residual. v is passed through Vanilla layers
    unchanged so later Nesterov/TMM layers still receive a velocity signal."""
    T = input_ids.shape[1]
    pos = torch.arange(T, device=input_ids.device)

    x = model.tok_emb(input_ids) + model.pos_emb(pos)
    v = model.vel_tok_emb(input_ids) + model.vel_pos_emb(pos)

    for i, layer in enumerate(model.layers):
        if i in nesterov_layers:
            x, v = layer(x, v)
        else:
            x = x + layer.attn(layer.ln_attn(x))
            x = x + layer.mlp(layer.ln_mlp(x))

    x = model.final_ln(x)
    logits = F.linear(x, model.tok_emb.weight)
    return logits


class HybridFormerLM(TemplateLM):
    """Wraps a YuriiFormer/TMMFormer for hybrid eval with lm-evaluation-harness.

    If `nesterov_layers` is given it takes precedence over `cutoff_layer`:
    the listed indices run Nesterov/TMM dynamics, the rest run Vanilla.
    """

    def __init__(self, checkpoint_path: str, base_model: str = "yurii",
                 cutoff_layer: int = 6,
                 nesterov_layers: Optional[Iterable[int]] = None,
                 device: str = "cuda"):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._eot_token_id = self.tokenizer.eot_token
        self._device = torch.device(device)
        self._max_length = 1024
        self.cutoff_layer = cutoff_layer
        self.nesterov_layers = set(nesterov_layers) if nesterov_layers is not None else None

        if base_model == "yurii":
            from model import YuriiFormer
            model = YuriiFormer()
        elif base_model == "tmm":
            from tmm_model import TMMFormer
            model = TMMFormer()
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict)
        model.to(self._device).eval()
        self.model = model

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        results = []
        for (context_str, continuation_str), context_enc, continuation_enc in tqdm(
            requests, disable=disable_tqdm, desc="loglikelihood"
        ):
            all_tokens = context_enc + continuation_enc
            cont_len = len(continuation_enc)

            if len(all_tokens) > self._max_length + 1:
                all_tokens = all_tokens[-(self._max_length + 1):]
                cont_len = min(cont_len, self._max_length)

            if cont_len == 0:
                results.append((-float("inf"), False))
                continue

            inp = torch.tensor(
                all_tokens[:-1], dtype=torch.long, device=self._device
            ).unsqueeze(0)

            if inp.shape[1] == 0:
                results.append((-float("inf"), False))
                continue

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if self.nesterov_layers is not None:
                    logits = hybrid_forward_set(self.model, inp, self.nesterov_layers)
                else:
                    logits = hybrid_forward(self.model, inp, self.cutoff_layer)

            pred_logits = logits[0, -cont_len:, :]
            cont_toks = torch.tensor(
                all_tokens[-cont_len:], dtype=torch.long, device=self._device
            )

            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            token_log_probs = log_probs[torch.arange(cont_len, device=self._device), cont_toks]

            total_ll = float(token_log_probs.sum())
            is_greedy = bool((pred_logits.argmax(dim=-1) == cont_toks).all())

            results.append((total_ll, is_greedy))

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("Not needed for multiple-choice tasks")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("Not needed for multiple-choice tasks")
