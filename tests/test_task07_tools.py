"""Tests for the Task 07 command-line tools.

These tests run offline. They build a fake bundle on disk and feed
fake profile dicts to the record builder. No AI Hub calls.
"""

import hashlib
import json

from edge_slm_lab.artifact_manifest import build_manifest, classify
from edge_slm_lab.hub_job_record import (
    build_record,
    extract_metrics,
    merge_into,
    parse_graph_name,
)
from edge_slm_lab.sanitize import check_text


def make_fake_bundle(root):
    (root / "sub").mkdir()
    (root / "part_1_of_2.serialized.bin").write_bytes(b"\x00" * 1024)
    (root / "part_2_of_2.serialized.bin").write_bytes(b"\x01" * 2048)
    (root / "genie-config.json").write_text('{"dialog": {}}')
    (root / "sub" / "tokenizer.json").write_text("{}")
    (root / "notes.txt").write_text("plain text")


def test_classify():
    assert classify("part_1_of_2.serialized.bin") == "qnn_context_binary"
    assert classify("genie-config.json") == "genie_config"
    assert classify("tokenizer.json") == "tokenizer"
    assert classify("htp_backend_ext_config.json") == "json_config"
    assert classify("weird.blob") == "other"


def test_build_manifest(tmp_path):
    make_fake_bundle(tmp_path)
    manifest = build_manifest(tmp_path)

    assert manifest["totals"]["n_files"] == 5
    assert manifest["totals"]["total_bytes"] == 1024 + 2048 + 14 + 2 + 10
    kinds = manifest["totals"]["by_kind"]
    assert kinds["qnn_context_binary"]["n_files"] == 2
    assert kinds["qnn_context_binary"]["total_bytes"] == 3072

    by_path = {e["path"]: e for e in manifest["files"]}
    expected = hashlib.sha256(b"\x00" * 1024).hexdigest()
    assert by_path["part_1_of_2.serialized.bin"]["sha256"] == expected
    assert by_path["sub/tokenizer.json"]["kind_guess"] == "tokenizer"

    assert manifest["largest_files"][0]["path"] == "part_2_of_2.serialized.bin"


def test_manifest_is_sanitized(tmp_path):
    make_fake_bundle(tmp_path)
    manifest = build_manifest(tmp_path)
    assert check_text(json.dumps(manifest)) == []


def test_check_text_catches_secrets():
    # Built at runtime so the literal never appears in the repo, which
    # would trip the pre-commit secret scan on this file itself.
    signed_url_marker = "X-" + "Amz" + "-Signature=abc"
    assert check_text(signed_url_marker) == ["X-Amz"]
    assert "hf_" in check_text("token hf_ABCDEFGHIJKLMNOPQRST")
    assert "https://" in check_text("https://signed.example/a?sig=1")
    assert check_text("just sizes and hashes 3072 bytes deadbeef") == []


def test_extract_metrics_full():
    profile = {
        "execution_summary": {
            "estimated_inference_time": 12345,
            "first_load_time": 900000,
            "warm_load_time": 40000,
            "estimated_inference_peak_memory": 350000000,
            "inference_memory_peak_range": [300000000, 350000000],
        },
        "execution_detail": [
            {"compute_unit": "NPU"},
            {"compute_unit": "NPU"},
            {"compute_unit": "CPU"},
        ],
    }
    m = extract_metrics(profile)
    assert m["estimated_inference_time_us"] == 12345
    assert m["first_load_time_us"] == 900000
    assert m["inference_memory_peak_range_bytes"] == [300000000, 350000000]
    assert m["layers_per_compute_unit"] == {"NPU": 2, "CPU": 1}


def test_extract_metrics_missing_stays_null():
    for profile in (None, {}, {"execution_summary": {}}):
        m = extract_metrics(profile)
        assert m["estimated_inference_time_us"] is None
        assert m["layers_per_compute_unit"] is None


def test_parse_graph_name():
    opts = "--qnn_options context_enable_graphs=token_ar1_cl4096_1_of_2"
    assert parse_graph_name(opts) == "token_ar1_cl4096_1_of_2"
    assert parse_graph_name("") is None
    assert parse_graph_name(None) is None
    assert parse_graph_name("--compute_unit npu") is None


def test_build_record_and_merge(tmp_path):
    record = build_record(
        job_id="jabc12345",
        job_type="ProfileJob",
        job_name="qwen3 w4a16 part 1",
        device_name="Snapdragon 8 Elite QRD",
        device_os="15",
        status_code="SUCCESS",
        status_message=None,
        profile=None,
        options="--qnn_options context_enable_graphs=token_ar1_cl4096_1_of_2",
    )
    assert record["status"]["code"] == "SUCCESS"
    assert record["graph_name"] == "token_ar1_cl4096_1_of_2"
    assert check_text(json.dumps(record)) == []

    out = tmp_path / "07_hub_job.json"
    doc = merge_into(out, record)
    out.write_text(json.dumps(doc))
    record2 = dict(record, job_id="jxyz67890")
    doc = merge_into(out, record2)
    assert set(doc["jobs"]) == {"jabc12345", "jxyz67890"}
