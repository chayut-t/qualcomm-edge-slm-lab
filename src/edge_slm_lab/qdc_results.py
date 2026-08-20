"""Record the geniex-bench results pulled from a QDC interactive session.

After the session you have a folder of cell JSONs on T9 (from
``adb pull /data/local/tmp/QDC_logs/results ...``). This tool parses
them and merges the whitelisted metrics into
``results/08_qdc_geniex.json``, keyed by session ID.

Usage (from the repo root):

    uv run python -m edge_slm_lab.qdc_results \
        /Volumes/T9/qualcomm-edge-slm-lab/artifacts/task08/pulled/results \
        --session-id <SESSION_ID>

Offline: no QDC access needed. Only whitelisted metric fields are
copied, and the record must pass the ``check_text`` secret scan before
it is written — same policy as Task 07.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from edge_slm_lab.sanitize import check_text

DEFAULT_RECORD = Path("results/08_qdc_geniex.json")


def parse_cell(cell: dict) -> dict | None:
    """One geniex-bench result cell (schema_version "3") to a flat record.

    Returns None for anything that is not a v3 cell. Missing medians make
    the cell unusable, so those return None too (never estimated).
    """
    if not isinstance(cell, dict) or cell.get("schema_version") != "3":
        return None
    agg = cell.get("agg") or {}
    params = cell.get("params") or {}

    def med(key: str):
        return (agg.get(key) or {}).get("median")

    ttft_ms = med("ttft_ms")
    prefill_tps = med("prefill_tps")
    decode_tps = med("decode_tps")
    if ttft_ms is None or prefill_tps is None or decode_tps is None:
        return None

    cell_id = cell.get("cell_id") or ""
    _, sep, suffix = cell_id.rpartition("-c")
    ctx = int(suffix) if sep and suffix.isdigit() else int(params.get("n_ctx") or 0)
    if ctx == 0:
        return None
    return {
        "cell_id": cell_id,
        "plugin": cell.get("plugin") or "",
        "device_alias": cell.get("device") or "",
        "context_length": ctx,
        "ttft_ms": float(ttft_ms),
        "prefill_tps": float(prefill_tps),
        "decode_tps": float(decode_tps),
        "prompt_tokens": int((agg.get("prompt_tokens") or {}).get("median") or 0),
        "gen_tokens": int((agg.get("gen_tokens") or {}).get("median") or 0),
    }


def parse_cells_from_dir(root: Path) -> list[dict]:
    cells = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed = parse_cell(data)
        if parsed is not None:
            cells.append(parsed)
    return cells


def merge_record(output: Path, session_id: str, record: dict) -> dict:
    doc = {"schema_version": 1, "sessions": {}}
    if output.exists():
        doc = json.loads(output.read_text())
    doc["sessions"][session_id] = record
    return doc


def write_record(output: Path, doc: dict) -> None:
    text = json.dumps(doc, indent=2) + "\n"
    problems = check_text(text)
    if problems:
        print(f"ERROR: record failed the secret scan: {problems}", file=sys.stderr)
        raise SystemExit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="The adb-pulled results folder")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--note",
        default=None,
        help="Optional one-line note, e.g. an on-device error you saw",
    )
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"ERROR: not a directory: {args.results_dir}", file=sys.stderr)
        return 1

    cells = parse_cells_from_dir(args.results_dir)
    print(f"Parsed {len(cells)} geniex-bench cells from {args.results_dir}")
    for c in cells:
        print(
            f"  [{c['cell_id']}] ctx={c['context_length']} "
            f"TTFT={c['ttft_ms']:.1f} ms  prefill={c['prefill_tps']:.1f} tok/s  "
            f"decode={c['decode_tps']:.2f} tok/s"
        )
    if not cells:
        print(
            "WARNING: no schema_version 3 cell JSONs found. Recording the "
            "empty result anyway — an unsuccessful run is still evidence."
        )

    record = {
        "route": "qdc_interactive_session",
        "qdc_device": "Snapdragon 8 Elite (QRD8750, SM8750)",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": args.note,
        "cells": cells,
    }
    doc = merge_record(args.record, args.session_id, record)
    write_record(args.record, doc)
    print(f"Record updated: {args.record}")
    return 0 if cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
