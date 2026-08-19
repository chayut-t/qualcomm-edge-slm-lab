"""Focused tests for edge_slm_lab.static_wrapper and edge_slm_lab.onnx_tools.

Exports the tiny random Qwen3 (2 layers) once per test session, then
checks the declared contracts, the frozen shapes, and the numerical
match between PyTorch and ONNX Runtime. Runs offline in seconds.
"""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from edge_slm_lab.manual_decode import greedy_decode
from edge_slm_lab.onnx_tools import all_shapes_fixed, graph_io
from edge_slm_lab.static_wrapper import (
    DecodeWrapper,
    PrefillWrapper,
    export_decode,
    export_padded_decode,
    export_prefill,
    padded_mask,
    past_names,
    tiny_qwen3,
)

PROMPT_LEN = 8
TOLERANCE = 1e-4  # documented fp32 tolerance: PyTorch vs ONNX Runtime


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("onnx")
    model = tiny_qwen3()
    prefill_path = tmp / "prefill.onnx"
    decode_path = tmp / "decode.onnx"
    export_prefill(model, prefill_path, prompt_len=PROMPT_LEN)
    export_decode(model, decode_path, past_len=PROMPT_LEN)
    return model, prefill_path, decode_path


def test_prefill_contract_is_fixed(setup):
    _, prefill_path, _ = setup
    io = graph_io(prefill_path)
    assert all_shapes_fixed(io)
    assert io["inputs"][0]["name"] == "input_ids"
    assert io["inputs"][0]["shape"] == [1, PROMPT_LEN]
    by_name = {t["name"]: t["shape"] for t in io["outputs"]}
    assert by_name["logits"] == [1, PROMPT_LEN, 128]
    assert by_name["present_key_0"] == [1, 2, PROMPT_LEN, 16]


def test_decode_contract_grows_cache_by_one(setup):
    _, _, decode_path = setup
    io = graph_io(decode_path)
    assert all_shapes_fixed(io)
    by_name = {t["name"]: t["shape"] for t in io["inputs"] + io["outputs"]}
    assert by_name["input_ids"] == [1, 1]
    assert by_name["past_key_0"] == [1, 2, PROMPT_LEN, 16]
    assert by_name["present_key_0"] == [1, 2, PROMPT_LEN + 1, 16]
    assert by_name["logits"] == [1, 1, 128]


def test_prefill_matches_pytorch(setup):
    model, prefill_path, _ = setup
    torch.manual_seed(1)
    prompt = torch.randint(0, 128, (1, PROMPT_LEN))
    with torch.no_grad():
        ref = PrefillWrapper(model)(prompt)
    sess = ort.InferenceSession(str(prefill_path))
    outs = sess.run(None, {"input_ids": prompt.numpy()})
    for got, want in zip(outs, ref):
        assert np.abs(got - want.numpy()).max() < TOLERANCE


def test_decode_matches_pytorch(setup):
    model, prefill_path, decode_path = setup
    torch.manual_seed(2)
    prompt = torch.randint(0, 128, (1, PROMPT_LEN))
    sess = ort.InferenceSession(str(prefill_path))
    pre_outs = sess.run(None, {"input_ids": prompt.numpy()})
    step = torch.tensor([[int(pre_outs[0][0, -1].argmax())]])
    past = [torch.from_numpy(t) for t in pre_outs[1:]]
    with torch.no_grad():
        ref = DecodeWrapper(model)(step, *past)
    dsess = ort.InferenceSession(str(decode_path))
    feed = {"input_ids": step.numpy()}
    for name, tensor in zip(past_names(2), pre_outs[1:]):
        feed[name] = tensor
    outs = dsess.run(None, feed)
    for got, want in zip(outs, ref):
        assert np.abs(got - want.numpy()).max() < TOLERANCE


def test_one_padded_graph_serves_many_steps(setup, tmp_path):
    """Part 6 extension: shapes stay fixed, mask/position values vary."""
    model, _, _ = setup
    cfg = model.config
    max_len, n_steps = 16, 8
    path = tmp_path / "padded_decode.onnx"
    export_padded_decode(model, path, max_len=max_len)
    assert all_shapes_fixed(graph_io(path))

    torch.manual_seed(3)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN))
    ref_ids, _ = greedy_decode(model, prompt, max_new_tokens=n_steps)

    with torch.no_grad():
        pre = PrefillWrapper(model)(prompt)
    past = [
        np.zeros((1, cfg.num_key_value_heads, max_len, cfg.head_dim), dtype=np.float32)
        for _ in range(2 * cfg.num_hidden_layers)
    ]
    for buf, t in zip(past, pre[1:]):
        buf[:, :, :PROMPT_LEN] = t.numpy()

    sess = ort.InferenceSession(str(path))
    filled = PROMPT_LEN
    next_id = int(pre[0][0, -1].argmax())
    ids = [next_id]
    for _ in range(n_steps - 1):
        feed = {
            "input_ids": np.array([[next_id]], dtype=np.int64),
            "position_ids": np.array([[filled]], dtype=np.int64),
            "attention_mask": padded_mask(filled, max_len).numpy(),
        }
        for name, buf in zip(past_names(cfg.num_hidden_layers), past):
            feed[name] = buf
        outs = sess.run(None, feed)
        for buf, new in zip(past, outs[1:]):
            buf[:, :, filled] = new[:, :, 0]
        filled += 1
        next_id = int(outs[0][0, -1].argmax())
        ids.append(next_id)
    assert ids == ref_ids


def test_wrong_length_input_is_rejected(setup):
    _, prefill_path, _ = setup
    sess = ort.InferenceSession(str(prefill_path))
    longer = np.zeros((1, PROMPT_LEN + 1), dtype=np.int64)
    with pytest.raises(Exception, match="Got invalid dimensions"):
        sess.run(None, {"input_ids": longer})
