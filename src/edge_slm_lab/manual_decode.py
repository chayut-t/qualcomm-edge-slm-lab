"""Greedy decoding with an explicit KV cache (Task 03 teaching code).

Data flow, batch size 1:

  prefill: input_ids [1, P] -> logits [1, P, vocab], cache K/V [1, H_kv, P, D]
  decode:  input_ids [1, 1] -> logits [1, 1, vocab], cache K/V [1, H_kv, T, D]

P = prompt length. T = tokens seen so far. For Qwen3-0.6B,
H_kv = 8 and D = 128. The last generated token is never fed back,
so the final cache length is P + generated - 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


def get_kv(cache, layer_idx: int):
    """Return (keys, values) for one layer across transformers cache APIs."""
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    if hasattr(cache, "key_cache"):  # older versions
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache[layer_idx][0], cache[layer_idx][1]


def kv_shape(cache, layer_idx: int = 0) -> tuple:
    """Key tensor shape for one layer: (batch, kv_heads, cache_len, head_dim)."""
    keys, _ = get_kv(cache, layer_idx)
    return tuple(keys.shape)


@dataclass
class DecodeTrace:
    """Record of one greedy decode run. One dict per decode step in `steps`."""

    prompt_len: int
    prefill_seconds: float
    prefill_kv_len: int
    steps: list = field(default_factory=list)
    stopped_on_eos: bool = False

    def kv_lens(self) -> list:
        return [s["kv_len"] for s in self.steps]

    def decode_seconds(self) -> list:
        return [s["seconds"] for s in self.steps]


def greedy_decode(model, input_ids, max_new_tokens: int, eos_token_id=None):
    """Prefill once, then decode one token at a time.

    input_ids: LongTensor [1, P]. Returns (new_ids, trace) where new_ids
    is a plain list of generated token ids, at most max_new_tokens long.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids shape [1, P], got {list(input_ids.shape)}")
    model.eval()
    new_ids = []
    with torch.no_grad():
        # Prefill: the whole prompt in one forward pass.
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        prefill_seconds = time.perf_counter() - t0
        cache = out.past_key_values
        # The score row of the LAST prompt token picks the first new token.
        next_id = int(out.logits[0, -1].argmax())
        trace = DecodeTrace(
            prompt_len=int(input_ids.shape[1]),
            prefill_seconds=round(prefill_seconds, 5),
            prefill_kv_len=kv_shape(cache)[2],
        )
        new_ids.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            trace.stopped_on_eos = True
            return new_ids, trace
        # Decode loop: one token in, one token out, cache grows by one.
        for step in range(1, max_new_tokens):
            step_input = torch.tensor([[next_id]], device=input_ids.device)
            t0 = time.perf_counter()
            out = model(input_ids=step_input, past_key_values=cache, use_cache=True)
            seconds = time.perf_counter() - t0
            cache = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            trace.steps.append(
                {
                    "step": step,
                    "input_len": 1,
                    "kv_len": kv_shape(cache)[2],
                    "seconds": round(seconds, 5),
                    "token_id": next_id,
                }
            )
            new_ids.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                trace.stopped_on_eos = True
                break
    return new_ids, trace
