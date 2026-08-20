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

## 2026-08-17 — Task 03: repository becomes an installable package

Removed `[tool.uv] package = false` from `pyproject.toml` because
`src/edge_slm_lab/` now exists (planned since Task 01). The environment
gets the package with `uv pip install -e .`. No dependency versions
changed.

## 2026-08-19 — Task 07: added the AI Hub client

- Added `qai-hub==0.55.0` to `requirements-core.txt`. It is the AI Hub
  client (CLI `qai-hub` plus the `qai_hub` Python package). Task 07
  needs it to submit and record hosted-device jobs. Installed with
  `uv pip install qai-hub` on 2026-08-19; 0.55.0 was the version PyPI
  resolved.
- It authenticates through the existing `~/.qai_hub/client.ini` from
  Task 01. No new credentials were created.

## 2026-08-20 — Task 08: installed the QDC Python SDK

- `qualcomm-device-cloud-sdk==0.4.1`, wheel from Qualcomm Software Center
  (catalog item `Qualcomm_Device_Cloud_SDK`, shipped inside
  `qualcomm_device_cloud_sdk-0.4.1.zip`). Extracted to
  `.ai-local/qdc-sdk/` (gitignored); installed with `uv pip install`.
- Not on PyPI. Pinned in `requirements-qdc.txt` as a direct file
  reference to the wheel in `.ai-local/qdc-sdk/`. Backup copies of the
  zip and wheel: `/Volumes/T9/qualcomm-edge-slm-lab/sdk-archive/`.
  Reinstall with `uv pip install -r requirements-qdc.txt`.
- API surface checked against `src/edge_slm_lab/qdc_run.py`: all eight
  functions and six model enums exist; `submit_job` signature matches
  the official 0.4.1 Appium sample. Added "nologs" as a terminal log
  upload status (seen in that sample).
- Auth: QDC API key in `.ai-local/secrets/qdc.env` as `QDC_API_TOKEN`.
  Key expires 6 months after generation.

## 2026-08-20 — Task 08: pivoted to the interactive-session route

- The learner's QDC User Settings shows only an SSH Keys tab; the API
  Keys tab from the QDC docs is not present. Without an API key the
  REST/SDK automated-job route cannot authenticate.
- Pivot: run the GenieX benchmark in a QDC interactive session that
  the learner drives over an SSH tunnel with adb. SSH key exists
  (qdc_id_2026-8-20_1527, expires 2026-10-19). This also fits the
  learning goal better: every device step is typed by the learner.
- Tooling: `qdc_run.py` (SDK submit/collect) removed; replaced by
  `qdc_results.py` (parse pulled cell JSONs). `qdc_artifact.py` now
  builds an adb device package (geniex-bench 86.4 MiB from the public
  mirror + matrix TSVs) instead of an Appium zip. On-device commands
  mirror the pinned `test_geniex_bench_android.py` step by step.
- `qualcomm-device-cloud-sdk==0.4.1` stays installed and pinned in
  `requirements-qdc.txt` but is unused on this route.
