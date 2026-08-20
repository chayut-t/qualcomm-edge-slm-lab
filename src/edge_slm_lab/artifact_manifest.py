"""Build a sanitized manifest of a fetched model bundle.

Input:  a directory that holds one extracted bundle (any depth).
Output: a JSON manifest with, per file: relative path, size in bytes,
        sha256 hash, and a guessed kind. Plus totals grouped by kind.
        No URLs, no tokens, no absolute paths inside the JSON.

Usage:
    uv run python -m edge_slm_lab.artifact_manifest BUNDLE_DIR \
        -o results/07_artifact_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from edge_slm_lab.sanitize import check_text


def classify(name: str) -> str:
    """Guess what a bundle file is from its name. Heuristic, not truth."""
    lower = name.lower()
    if lower.endswith(".bin"):
        return "qnn_context_binary"
    if "tokenizer" in lower:
        return "tokenizer"
    if lower.endswith(".json"):
        return "genie_config" if "genie" in lower else "json_config"
    if lower.endswith((".yaml", ".yml")):
        return "yaml_config"
    if lower.endswith(".so"):
        return "shared_library"
    if lower.endswith((".md", ".txt")) or "license" in lower:
        return "text"
    return "other"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(bundle_dir: Path) -> dict:
    files = sorted(p for p in bundle_dir.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found under {bundle_dir}")

    entries = []
    by_kind: dict[str, dict[str, int]] = {}
    for p in files:
        kind = classify(p.name)
        size = p.stat().st_size
        entries.append(
            {
                "path": str(p.relative_to(bundle_dir)),
                "size_bytes": size,
                "sha256": sha256_of(p),
                "kind_guess": kind,
            }
        )
        agg = by_kind.setdefault(kind, {"n_files": 0, "total_bytes": 0})
        agg["n_files"] += 1
        agg["total_bytes"] += size

    largest = sorted(entries, key=lambda e: e["size_bytes"], reverse=True)[:5]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle_dir_name": bundle_dir.name,
        "files": entries,
        "totals": {
            "n_files": len(entries),
            "total_bytes": sum(e["size_bytes"] for e in entries),
            "by_kind": by_kind,
        },
        "largest_files": [
            {"path": e["path"], "size_bytes": e["size_bytes"]} for e in largest
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.bundle_dir.is_dir():
        print(f"Not a directory: {args.bundle_dir}", file=sys.stderr)
        return 1

    manifest = build_manifest(args.bundle_dir)

    text = json.dumps(manifest, indent=2)
    problems = check_text(text)
    if problems:
        print(f"REFUSED to write: found {problems} in manifest", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")

    mib = manifest["totals"]["total_bytes"] / (1 << 20)
    print(f"Bundle: {manifest['bundle_dir_name']}")
    print(f"Files:  {manifest['totals']['n_files']}  ({mib:,.1f} MiB total)")
    for kind, agg in sorted(manifest["totals"]["by_kind"].items()):
        kmib = agg["total_bytes"] / (1 << 20)
        print(f"  {kind:<20} {agg['n_files']:>3} files  {kmib:>10,.1f} MiB")
    print("Largest files:")
    for e in manifest["largest_files"]:
        print(f"  {e['size_bytes'] / (1 << 20):>10,.1f} MiB  {e['path']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
