# Environment record

Claude updates this file when the environment changes.

## Local machine (recorded at initialization, 2026-08-06)

- OS: macOS 15.7.7 (Darwin 24.6.0)
- Architecture: arm64 (Apple Silicon)
- Python: 3.11 (managed by `uv 0.11.32`)
- Git: 2.50.1
- Free disk at init: about 52 GiB
- External drive for large files: `/Volumes/T9` (about 1.5 TiB free at
  2026-08-06). Use `/Volumes/T9/qualcomm-edge-slm-lab/` for model
  weights, bundles, and caches. Check it is mounted before use.

## Python environment

- Created with `uv venv --python 3.11` (`.venv/`)
- Core packages: `requirements-core.txt`
- Qualcomm heavy package: `requirements-qualcomm.txt` (install when a
  task asks)

Task 01's notebook records the full package inventory automatically.
