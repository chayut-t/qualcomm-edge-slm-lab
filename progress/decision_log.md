# Decision log

Record changes to the plan, version pins, or official commands here.
One dated entry per decision. Newest first.

## 2026-08-07 — uv build fix

- `uv run` failed with `error in 'egg_base' option: 'src' does not exist
  or is not a directory`. Cause: `pyproject.toml` declared a package in
  `src/`, which a later task creates. Fix: `[tool.uv] package = false`
  in `pyproject.toml`. Remove that line when `src/edge_slm_lab` exists.

## 2026-08-06 — External drive for large files

- Large downloads (weights, ONNX, QNN bundles, caches) go to
  `/Volumes/T9/qualcomm-edge-slm-lab/` when needed. The internal disk
  has about 52 GiB free; the T9 drive has about 1.5 TiB.

## 2026-08-06 — Initialization

- Project initialized from `PROJECT_INIT.md`.
- Pinned `qai-hub-models-cli==0.59.0` (requirements-core.txt) and
  `qai-hub-models==0.59.0` (requirements-qualcomm.txt). Both versions
  confirmed on PyPI on 2026-08-06.
- Qualcomm reference pinned to `qualcomm/ai-hub-models` tag `v0.59.0`.
- Local machine: macOS 15.7.7, Apple Silicon (arm64), Python 3.11 via `uv`.
