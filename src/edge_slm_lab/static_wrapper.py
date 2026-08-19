"""Static-shape export wrappers (Task 04 teaching code).

A compiled NPU program is a frozen graph. Every input and output tensor
has a fixed name, dtype, and shape. Hugging Face hides the KV cache in a
Python object, so this module wraps the model to make the cache explicit:

  prefill graph:  input_ids [1, P]
                  -> logits [1, P, vocab], present_key_i / present_value_i
                     [1, H_kv, P, D] for every layer i
  decode graph:   input_ids [1, 1], past_key_i / past_value_i [1, H_kv, T, D]
                  -> logits [1, 1, vocab], present_* [1, H_kv, T+1, D]

P = prompt length, T = cache length, both frozen at export time.
Export uses torch.jit.trace (dynamo=False), so the TracerWarnings about
"the trace might not generalize to other inputs" are expected: we WANT a
graph that only works at one shape. That is the static contract.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

OPSET = 17

# Additive attention-mask values: 0.0 = visible, MASK_MIN = hidden.
MASK_MIN = torch.finfo(torch.float32).min


def tiny_qwen3() -> Qwen3ForCausalLM:
    """Small random Qwen3 (2 layers, hidden 64) for offline, seconds-fast labs.

    Same layer type as Qwen3-0.6B, same cache layout, just smaller numbers:
    4 query heads, 2 KV heads, head_dim 16, vocab 128.
    """
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    return Qwen3ForCausalLM(config).eval()


def past_names(n_layers: int) -> list:
    return [f"past_{t}_{i}" for i in range(n_layers) for t in ("key", "value")]


def present_names(n_layers: int) -> list:
    return [f"present_{t}_{i}" for i in range(n_layers) for t in ("key", "value")]


def _flat_kv(cache, n_layers: int) -> list:
    out = []
    for i in range(n_layers):
        out.append(cache.layers[i].keys)
        out.append(cache.layers[i].values)
    return out


class PrefillWrapper(nn.Module):
    """input_ids [1, P] -> (logits, key_0, value_0, key_1, value_1, ...)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=True)
        return (out.logits, *_flat_kv(out.past_key_values, self.n_layers))


class DecodeWrapper(nn.Module):
    """(input_ids [1, 1], past tensors) -> (logits, present tensors).

    The hidden DynamicCache is rebuilt from plain tensors so the exported
    graph carries the cache through named inputs and outputs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids, *past):
        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(past[2 * i], past[2 * i + 1], i)
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return (out.logits, *_flat_kv(out.past_key_values, self.n_layers))


def export_prefill(model, path, prompt_len: int) -> None:
    """Export a prefill graph frozen at input_ids [1, prompt_len]."""
    wrapper = PrefillWrapper(model)
    example = torch.zeros(1, prompt_len, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["input_ids"],
        output_names=["logits"] + present_names(wrapper.n_layers),
        opset_version=OPSET,
        dynamo=False,
    )


def new_kv_names(n_layers: int) -> list:
    return [f"new_{t}_{i}" for i in range(n_layers) for t in ("key", "value")]


def padded_mask(filled: int, max_len: int) -> torch.Tensor:
    """Additive mask [1, 1, 1, max_len + 1] for a padded decode step.

    Slots 0..filled-1 hold real past tokens: visible (0.0). The rest of
    the padded cache is hidden (MASK_MIN). The last position is the new
    token itself (its K/V sit at slot max_len after concat): visible.
    """
    mask = torch.full((1, 1, 1, max_len + 1), MASK_MIN)
    mask[..., :filled] = 0.0
    mask[..., -1] = 0.0
    return mask


class PaddedDecodeWrapper(nn.Module):
    """One decode graph for MANY steps: shapes fixed, values vary.

    Inputs: input_ids [1, 1], position_ids [1, 1] (RoPE position of the
    new token), attention_mask [1, 1, 1, max_len + 1], and per layer
    past_key_i / past_value_i [1, H_kv, max_len, D] — always full size,
    empty slots padded with zeros and hidden by the mask.

    Outputs: logits [1, 1, vocab] and per layer new_key_i / new_value_i
    [1, H_kv, 1, D] — only the new token's K/V. The HOST writes them
    into slot `filled` of its padded buffers. Real runtimes do the same
    bookkeeping around the compiled graph.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids, position_ids, attention_mask, *past):
        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(past[2 * i], past[2 * i + 1], i)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        news = []
        for i in range(self.n_layers):
            news.append(out.past_key_values.layers[i].keys[:, :, -1:, :])
            news.append(out.past_key_values.layers[i].values[:, :, -1:, :])
        return (out.logits, *news)


def export_padded_decode(model, path, max_len: int) -> None:
    """Export ONE decode graph usable for every step up to max_len."""
    wrapper = PaddedDecodeWrapper(model)
    cfg = model.config
    past = [
        torch.zeros(1, cfg.num_key_value_heads, max_len, cfg.head_dim)
        for _ in range(2 * wrapper.n_layers)
    ]
    example = (
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        padded_mask(1, max_len),
        *past,
    )
    torch.onnx.export(
        wrapper,
        example,
        str(path),
        input_names=["input_ids", "position_ids", "attention_mask"]
        + past_names(wrapper.n_layers),
        output_names=["logits"] + new_kv_names(wrapper.n_layers),
        opset_version=OPSET,
        dynamo=False,
    )


def export_decode(model, path, past_len: int) -> None:
    """Export a decode graph frozen at cache length past_len."""
    wrapper = DecodeWrapper(model)
    cfg = model.config
    step = torch.zeros(1, 1, dtype=torch.long)
    past = [
        torch.zeros(1, cfg.num_key_value_heads, past_len, cfg.head_dim)
        for _ in range(2 * wrapper.n_layers)
    ]
    torch.onnx.export(
        wrapper,
        (step, *past),
        str(path),
        input_names=["input_ids"] + past_names(wrapper.n_layers),
        output_names=["logits"] + present_names(wrapper.n_layers),
        opset_version=OPSET,
        dynamo=False,
    )
