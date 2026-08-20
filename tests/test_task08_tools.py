"""Tests for the Task 08 QDC interactive-session tools.

These tests run offline. They build a fake bench tarball and a fake
bundle on disk. No QDC access, no network, no real 750 MiB bundle.
"""

import io
import json
import tarfile

from edge_slm_lab.qdc_artifact import (
    DEVICE_BUNDLE,
    DEVICE_ROOT,
    bench_command,
    build_execution_manifest,
    build_matrix_tsv,
    extract_bench,
    session_commands,
)
from edge_slm_lab.qdc_results import merge_record, parse_cell, parse_cells_from_dir
from edge_slm_lab.sanitize import check_text


def make_fake_bundle(root):
    root.mkdir()
    (root / "part1_of_2.bin").write_bytes(b"\x01" * 64)
    (root / "part2_of_2.bin").write_bytes(b"\x02" * 128)
    (root / "genie_config.json").write_text(
        json.dumps(
            {
                "dialog": {
                    "context": {"size": 4096},
                    "sampler": {"seed": 42, "temp": 0.8, "top-k": 40, "top-p": 0.95},
                    "engine": {
                        "n-threads": 3,
                        "backend": {
                            "type": "QnnHtp",
                            "extensions": "htp_backend_ext_config.json",
                            "QnnHtp": {
                                "kv-dim": 128,
                                "pos-id-dim": 64,
                                "rope-theta": 1000000,
                            },
                        },
                        "model": {
                            "binary": {
                                "ctx-bins": ["part1_of_2.bin", "part2_of_2.bin"]
                            }
                        },
                    },
                }
            }
        )
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "tool_versions": {"qairt": "2.45.0.260326154327"},
                "genie": {"context_lengths": [512, 1024, 2048, 3072, 4096]},
                "chipset_attributes": {
                    "name": "qualcomm-snapdragon-8-elite",
                    "htp_version": 79,
                    "reference_device": "Snapdragon 8 Elite QRD",
                },
            }
        )
    )
    (root / "sample_prompt.txt").write_text("What is gravity?")


def make_fake_bench_tar(tar_path):
    """A tarball with a top-level folder, like the real mirror asset."""

    def add(tf, name, data, mode=0o644):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = mode
        tf.addfile(info, io.BytesIO(data))

    with tarfile.open(tar_path, "w:gz") as tf:
        add(tf, "geniex-bench-android-arm64/bin/geniex-bench", b"\x7fELF", 0o755)
        add(tf, "geniex-bench-android-arm64/lib/qairt/htp-files/libx.so", b"\x00")
        add(tf, "geniex-bench-android-arm64/lib/llama_cpp/liby.so", b"\x00")


def test_matrix_tsv_exact():
    assert build_matrix_tsv(512) == (
        "qwen3_0_6b-qairt-npu-c512\tqairt\tnpu\t"
        "/data/local/tmp/pkg-geniex/qairt_bundles/qwen3_0_6b\t\t\t\t\n"
    )


def test_bench_command_flags():
    cmd = bench_command(4096)
    assert f"--matrix-file {DEVICE_ROOT}/matrix-4096.tsv" in cmd
    assert "-r 3" in cmd
    assert "-c 4096 -n 128" in cmd
    assert f"--prompt-file {DEVICE_BUNDLE}/sample_prompt.txt" in cmd
    assert "--chipset 'qualcomm-snapdragon-8-elite'" in cmd
    assert "LD_LIBRARY_PATH=" in cmd and "ADSP_LIBRARY_PATH=" in cmd


def test_extract_bench_strips_top_dir(tmp_path):
    tar_path = tmp_path / "bench.tar.gz"
    make_fake_bench_tar(tar_path)
    pkg = tmp_path / "device_pkg"
    pkg.mkdir()
    extract_bench(tar_path, pkg)
    assert (pkg / "bin" / "geniex-bench").is_file()
    assert (pkg / "lib" / "qairt" / "htp-files" / "libx.so").is_file()
    # The top-level folder name must not survive extraction.
    assert not (pkg / "geniex-bench-android-arm64").exists()


def test_session_commands_order(tmp_path):
    text = session_commands(
        tmp_path / "device_pkg", tmp_path / "bundle", [512, 4096], tmp_path / "pulled"
    )
    # Tunnel before adb, both pushes before chmod, both benches before pull.
    order = [
        "ssh -i <PEM>",
        "adb kill-server",
        f"adb push {tmp_path / 'device_pkg'}/. {DEVICE_ROOT}",
        f"adb push {tmp_path / 'bundle'}/. {DEVICE_BUNDLE}",
        "chmod 755",
        "matrix-512.tsv",
        "matrix-4096.tsv",
        "adb pull",
        "rm -rf",
    ]
    pos = -1
    for marker in order:
        new_pos = text.find(marker)
        assert new_pos > pos, f"out of order or missing: {marker}"
        pos = new_pos


def test_execution_manifest_sections(tmp_path):
    bundle_dir = tmp_path / "bundle"
    make_fake_bundle(bundle_dir)
    manifest = build_execution_manifest(bundle_dir, [512, 4096])

    for section in (
        "qnn_context_binary",
        "qairt",
        "geniex",
        "chipset",
        "runtime_configuration",
    ):
        assert section in manifest, section

    binaries = manifest["qnn_context_binary"]["files"]
    assert [b["name"] for b in binaries] == ["part1_of_2.bin", "part2_of_2.bin"]
    assert binaries[0]["size_bytes"] == 64
    assert len(binaries[0]["sha256"]) == 64
    assert manifest["qairt"]["version"] == "2.45.0.260326154327"
    assert manifest["chipset"]["soc"] == "sm8750"
    assert manifest["runtime_configuration"]["benchmark_context_lengths"] == [512, 4096]
    # Compile-time facts and run-time choices must stay distinguishable.
    assert manifest["qnn_context_binary"]["fixed_at"] == "compile time"
    assert manifest["geniex"]["chosen_at"] == "run time"
    assert check_text(json.dumps(manifest)) == []


def make_cell(ctx=512, ttft=210.5, prefill=850.0, decode=22.3):
    return {
        "schema_version": "3",
        "cell_id": f"qwen3_0_6b-qairt-npu-c{ctx}",
        "plugin": "qairt",
        "device": "npu",
        "agg": {
            "ttft_ms": {"median": ttft},
            "prefill_tps": {"median": prefill},
            "decode_tps": {"median": decode},
            "prompt_tokens": {"median": 38},
            "gen_tokens": {"median": 128},
        },
        "params": {"n_ctx": ctx},
    }


def test_parse_cell():
    parsed = parse_cell(make_cell())
    assert parsed["context_length"] == 512
    assert parsed["ttft_ms"] == 210.5
    assert parsed["decode_tps"] == 22.3
    assert parsed["prompt_tokens"] == 38


def test_parse_cell_rejects_wrong_schema_and_missing_medians():
    assert parse_cell({"schema_version": "2"}) is None
    incomplete = make_cell()
    del incomplete["agg"]["decode_tps"]
    assert parse_cell(incomplete) is None


def test_parse_cells_from_dir(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "cell512.json").write_text(json.dumps(make_cell(512)))
    (tmp_path / "cell4096.json").write_text(json.dumps(make_cell(4096, ttft=950.0)))
    (tmp_path / "broken.json").write_text("not json {")
    (tmp_path / "other.json").write_text(json.dumps({"schema_version": "1"}))
    cells = parse_cells_from_dir(tmp_path)
    assert sorted(c["context_length"] for c in cells) == [512, 4096]


def test_merge_record_accumulates(tmp_path):
    out = tmp_path / "08_qdc_geniex.json"
    doc = merge_record(out, "session-a", {"cells": []})
    out.write_text(json.dumps(doc))
    doc = merge_record(out, "session-b", {"cells": [parse_cell(make_cell())]})
    assert set(doc["sessions"]) == {"session-a", "session-b"}
    assert doc["sessions"]["session-b"]["cells"][0]["context_length"] == 512
