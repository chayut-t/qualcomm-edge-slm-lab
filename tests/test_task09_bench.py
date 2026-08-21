"""Tests for the Task 09 diagnostic benchmark.

These tests run offline. They use a fake word-count tokenizer and a fake
runner. No model download, no torch forward passes.
"""

import json
from pathlib import Path

import pytest

from edge_slm_lab.diag_bench import (
    UNLIKE_RUNTIME_WARNING,
    bench_all,
    build_prompts,
    check_repro,
    detect_failure,
    expand_filler,
    import_geniex,
    json_validity,
    load_rows,
    make_row,
    write_rows,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_prompts.json"


def count_words(text: str) -> int:
    return len(text.split())


ENV = {
    "runtime": "pytorch_local",
    "device": "cpu (test)",
    "precision": "float32",
    "context_limit": 40960,
    "model_size_bytes": 123,
    "model_id": "Qwen/Qwen3-0.6B",
}


def fake_runner(text, gen_tokens):
    # Deterministic ids derived from the prompt; fixed fake timings.
    seed = sum(ord(c) for c in text) % 97
    new_ids = [seed + i for i in range(gen_tokens)]
    return new_ids, 0.100, 1.0, gen_tokens - 1


def fake_decode(ids):
    return "out " + " ".join(str(i) for i in ids)


def test_expand_filler_hits_target():
    text = "Read this. [FILLER] Answer now."
    filler = "one two three four five six seven eight nine ten"
    out = expand_filler(text, filler, 512, count_words)
    assert abs(count_words(out) - 512) <= 10
    plain = expand_filler("No slot here.", filler, 512, count_words)
    assert plain == "No slot here."


def test_build_prompts_matches_brief():
    config = json.loads(CONFIG_PATH.read_text())
    prompts = build_prompts(config, count_words)
    assert len(prompts) == 8
    cats = sorted(p["category"] for p in prompts)
    assert cats.count("instruction") == 2
    assert cats.count("json_extraction") == 2
    assert cats.count("summarization") == 2
    assert cats.count("multilingual") == 2
    assert any(p["language"] == "th" for p in prompts)
    assert set(p["target_tokens"] for p in prompts) == {32, 128, 512}
    for p in prompts:
        assert "[FILLER]" not in p["text"]
        assert p["prompt_tokens"] > 0


def test_build_prompts_rejects_wrong_count():
    config = json.loads(CONFIG_PATH.read_text())
    config["prompts"] = config["prompts"][:7]
    with pytest.raises(ValueError):
        build_prompts(config, count_words)


def test_json_validity():
    assert json_validity('Sure: {"a": 1, "b": "x"} done') is True
    assert json_validity('{"a": 1,,}') is False
    assert json_validity("no braces at all") is False


def test_detect_failure():
    assert detect_failure([], "") == "empty_output"
    assert detect_failure([5] * 20, "aaaa") == "token_loop"
    assert detect_failure([1, 2, 3], "fine text") is None


def test_bench_rows_schema_and_reps():
    config = json.loads(CONFIG_PATH.read_text())
    prompts = build_prompts(config, count_words)[:2]
    calls = []

    def counting_runner(text, gen_tokens):
        calls.append(text)
        return fake_runner(text, gen_tokens)

    rows = bench_all(
        prompts, counting_runner, fake_decode, ENV, "run-x",
        gen_tokens=48, reps=3, warmup=1,
        peak_fn=lambda: 1000, log=lambda *a, **k: None,
    )
    # 1 warm-up + 3 measured per prompt.
    assert len(calls) == 2 * 4
    assert len(rows) == 2 * 3
    assert [r["rep"] for r in rows[:3]] == [1, 2, 3]
    row = rows[0]
    assert row["run_id"] == "run-x"
    assert row["runtime"] == "pytorch_local"
    assert row["ttft_ms"] == 100.0
    assert row["decode_tps"] == 47.0
    assert row["gen_tokens"] == 48
    assert row["peak_rss_bytes"] == 1000
    assert row["output_sha256"] and row["output_failure"] is None
    assert row["unlike_runtime_warning"] is None


def test_null_policy_no_estimates():
    prompt = {"id": "p", "category": "instruction", "language": "en",
              "target_tokens": 32, "prompt_tokens": 30}
    # EOS on the first token: zero decode steps -> decode_tps must be null.
    def eos_runner(text, gen_tokens):
        return [7], 0.05, 0.0, 0

    rows = bench_all(
        [dict(prompt, text="hi")], eos_runner, fake_decode, ENV, "r",
        gen_tokens=48, reps=1, warmup=0,
        peak_fn=lambda: 1, log=lambda *a, **k: None,
    )
    assert rows[0]["decode_tps"] is None
    assert rows[0]["gen_tokens"] == 1
    # json_valid is null outside the json_extraction category.
    assert rows[0]["json_valid"] is None


def test_check_repro_pass_and_fail():
    prompt = {"id": "p1", "category": "instruction", "language": "en",
              "target_tokens": 32, "prompt_tokens": 30}
    def measure(sha_text):
        return {"new_ids": [1, 2], "text": sha_text, "ttft_ms": 1.0,
                "prefill_tps": 10.0, "decode_tps": 2.0, "peak_rss_bytes": 1}

    a = make_row(ENV, "a", prompt, 1, measure("same"))
    b = make_row(ENV, "b", prompt, 1, measure("same"))
    c = make_row(ENV, "c", prompt, 1, measure("different"))
    quiet = lambda *a, **k: None
    assert check_repro([a, b], "a", "b", log=quiet) is True
    assert check_repro([a, c], "a", "c", log=quiet) is False
    assert check_repro([a], "a", "missing", log=quiet) is False


def test_import_geniex_rows(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "c512.json").write_text(json.dumps({
        "schema_version": "3",
        "cell_id": "qwen3_0_6b-qairt-npu-c512",
        "plugin": "qairt",
        "device": "npu",
        "agg": {
            "ttft_ms": {"median": 20.1},
            "prefill_tps": {"median": 6359.6},
            "decode_tps": {"median": 107.3},
            "prompt_tokens": {"median": 128},
            "gen_tokens": {"median": 128},
        },
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "qnn_context_binary": {"files": [{"size_bytes": 100}, {"size_bytes": 28}]},
        "chipset": {"reference_device": "Snapdragon 8 Elite QRD",
                    "qdc_target": "SM8750"},
        "runtime_configuration": {"context_size_limit": 4096},
    }))
    rows = import_geniex(cells, manifest, "qdc-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["runtime"] == "geniex_qairt_qdc"
    assert row["precision"] == "w4a16"
    assert row["model_size_bytes"] == 128
    assert row["decode_tps"] == 107.3
    assert row["aggregation"] == "median_of_3"
    assert row["json_valid"] is None
    assert row["peak_rss_bytes"] is None
    assert row["unlike_runtime_warning"] == UNLIKE_RUNTIME_WARNING


def test_write_rows_secret_guard(tmp_path):
    out = tmp_path / "r.jsonl"
    good = {"run_id": "a", "runtime": "pytorch_local", "prompt_id": "p", "rep": 1}
    write_rows(out, [good])
    assert load_rows(out) == [good]
    bad = dict(good, note="see https://example.com/signed")
    with pytest.raises(SystemExit):
        write_rows(out, [bad])
    # The bad batch must not be partially written.
    assert load_rows(out) == [good]
