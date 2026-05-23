"""LM wrapper for AdamWFormer evaluation with lm-evaluation-harness."""

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from adamw_model import AdamWFormer


class AdamWFormerLM(TemplateLM):
    """Wraps AdamWFormer for use with lm-evaluation-harness."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._eot_token_id = self.tokenizer.eot_token
        self._device = torch.device(device)
        self._max_length = 1024

        model = AdamWFormer()
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
                logits = self.model(inp)

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
