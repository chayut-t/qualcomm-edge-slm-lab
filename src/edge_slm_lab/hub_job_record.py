"""Save a sanitized record of one Qualcomm AI Hub job.

Fetches the job by ID, waits for it to finish, then writes a record
with a fixed set of fields: job ID, type, device, status, and (for a
finished profile job) the key metrics. Metrics that the profile does
not report stay null. Nothing is estimated. No URLs or tokens are
stored. Records merge into one JSON file keyed by job ID, so you can
record several jobs into the same file.

Usage:
    uv run python -m edge_slm_lab.hub_job_record JOB_ID \
        -o results/07_hub_job.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import re

from edge_slm_lab.sanitize import check_text

# All times from AI Hub profiles are microseconds.
METRIC_KEYS = {
    "estimated_inference_time_us": "estimated_inference_time",
    "first_load_time_us": "first_load_time",
    "warm_load_time_us": "warm_load_time",
    "estimated_inference_peak_memory_bytes": "estimated_inference_peak_memory",
}


def extract_metrics(profile: dict | None) -> dict:
    """Pull a fixed set of metrics out of a raw AI Hub profile dict.

    Missing values become None. Unknown extra keys are ignored, never
    copied, so signed URLs in the raw profile can not leak through.
    """
    summary = (profile or {}).get("execution_summary") or {}
    metrics: dict = {
        ours: summary.get(theirs) for ours, theirs in METRIC_KEYS.items()
    }
    peak_range = summary.get("inference_memory_peak_range")
    metrics["inference_memory_peak_range_bytes"] = (
        list(peak_range) if peak_range else None
    )

    detail = (profile or {}).get("execution_detail") or []
    units: dict[str, int] = {}
    for layer in detail:
        unit = layer.get("compute_unit")
        if unit:
            units[unit] = units.get(unit, 0) + 1
    metrics["layers_per_compute_unit"] = units or None
    return metrics


def parse_graph_name(options: str | None) -> str | None:
    """Pull the profiled graph name out of the job's submit options.

    Multi-graph context binaries are profiled with
    --qnn_options context_enable_graphs=<graph name>. Returns None when
    the job did not name a graph (single-graph model, or the job failed
    for exactly that reason).
    """
    if not options:
        return None
    match = re.search(r"context_enable_graphs=([\w,]+)", options)
    return match.group(1) if match else None


def build_record(
    job_id: str,
    job_type: str,
    job_name: str | None,
    device_name: str | None,
    device_os: str | None,
    status_code: str,
    status_message: str | None,
    profile: dict | None,
    options: str | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "job_type": job_type,
        "job_name": job_name,
        "device": {"name": device_name, "os": device_os},
        "options": options or None,
        "graph_name": parse_graph_name(options),
        "status": {"code": status_code, "message": status_message},
        "metrics": extract_metrics(profile),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def merge_into(output: Path, record: dict) -> dict:
    """Add one record to the output file, keyed by job ID."""
    doc = {"schema_version": 1, "jobs": {}}
    if output.exists():
        doc = json.loads(output.read_text())
    doc["jobs"][record["job_id"]] = record
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import qai_hub

    job = qai_hub.get_job(args.job_id)
    print(f"Found {type(job).__name__} '{job.name}' — waiting for it to finish...")
    status = job.wait()

    profile = None
    if type(job).__name__ == "ProfileJob" and status.success:
        profile = job.download_profile()

    device = getattr(job, "device", None)
    record = build_record(
        job_id=job.job_id,
        job_type=type(job).__name__,
        job_name=job.name,
        device_name=getattr(device, "name", None),
        device_os=getattr(device, "os", None),
        status_code=str(status.code),
        status_message=status.message,
        profile=profile,
        options=getattr(job, "options", None),
    )

    doc = merge_into(args.output, record)
    text = json.dumps(doc, indent=2)
    problems = check_text(text)
    if problems:
        print(f"REFUSED to write: found {problems} in record", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")

    print(f"Job {record['job_id']} ({record['job_type']}) on "
          f"{record['device']['name']}: {record['status']['code']}")
    for key, value in record["metrics"].items():
        print(f"  {key}: {value}")
    print(f"Wrote {args.output} ({len(doc['jobs'])} job(s) recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
