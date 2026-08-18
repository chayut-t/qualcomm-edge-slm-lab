"""Focused tests for edge_slm_lab.manual_decode.

Uses a tiny random Qwen3 model (2 layers, hidden 64) so the tests run
in seconds, offline, with no download. The loop logic being tested is
identical to what the notebook runs on the real Qwen3-0.6B.
"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from edge_slm_lab.manual_decode import greedy_decode, kv_shape

PROMPT = torch.tensor([[5, 17, 42, 7]])  # [1, 4]
MAX_NEW = 10


@pytest.fixture(scope="module")
def model():
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


def test_tokens_match_hf_generate(model):
    new_ids, _ = greedy_decode(model, PROMPT, max_new_tokens=MAX_NEW)
    baseline = model.generate(
        PROMPT, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True, pad_token_id=0
    )
    assert new_ids == baseline[0, PROMPT.shape[1] :].tolist()


def test_cache_grows_one_per_step(model):
    new_ids, trace = greedy_decode(model, PROMPT, max_new_tokens=MAX_NEW)
    prompt_len = PROMPT.shape[1]
    assert trace.prefill_kv_len == prompt_len
    # Step 1 leaves the cache at prompt_len + 1, and so on.
    assert trace.kv_lens() == list(range(prompt_len + 1, prompt_len + MAX_NEW))
    # The last generated token is never fed back.
    assert trace.kv_lens()[-1] == prompt_len + len(new_ids) - 1


def test_kv_shape_uses_kv_heads_not_query_heads(model):
    with torch.no_grad():
        out = model(input_ids=PROMPT, use_cache=True)
    batch, kv_heads, cache_len, head_dim = kv_shape(out.past_key_values)
    assert (batch, kv_heads, cache_len, head_dim) == (1, 2, PROMPT.shape[1], 16)


def test_rejects_batched_input(model):
    with pytest.raises(ValueError):
        greedy_decode(model, torch.zeros(2, 4, dtype=torch.long), max_new_tokens=2)
