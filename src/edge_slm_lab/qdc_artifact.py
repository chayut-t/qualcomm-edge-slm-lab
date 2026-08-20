"""Build the device package for a QDC interactive session, plus the
execution manifest.

Task 08 runs the full GenieX loop on a QDC phone. You drive the phone
yourself over an SSH tunnel with adb. This tool prepares everything the
phone needs, so the interactive session spends minutes on running, not
on preparing:

1. ``device_pkg/`` — the directory you will ``adb push``:

   - ``bin/``, ``lib/``, ...   geniex-bench for android-arm64, downloaded
     from the public mirror and unpacked
   - ``matrix-512.tsv``, ``matrix-4096.tsv``  one benchmark row each,
     telling geniex-bench what to run at that context length

   The 750.8 MiB W4A16 bundle is NOT copied in here. You push it to the
   phone straight from its Task 07 location (second push command).

2. ``commands.txt`` — the exact session commands in order (tunnel, push,
   run, pull, clean). The lesson explains each one.

3. ``results/08_execution_manifest.json`` — the record that separates
   what is compiled into the QNN context binaries from what QAIRT,
   GenieX, the chipset, and the runtime configuration each control.

Usage (from the repo root):

    uv run python -m edge_slm_lab.qdc_artifact \
        --out-dir /Volumes/T9/qualcomm-edge-slm-lab/artifacts/task08
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from edge_slm_lab.sanitize import check_text

# Where the Task 07 bundle was fetched.
DEFAULT_BUNDLE_DIR = Path(
    "/Volumes/T9/qualcomm-edge-slm-lab/artifacts/task07/"
    "qwen3_0_6b-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite"
)

# Values below mirror the pinned _shared/llm/qdc code for the qairt plugin.
MODEL_ID = "qwen3_0_6b"
PLUGIN = "qairt"
DEVICE_ALIAS = "npu"
N_GEN = 128
REPS = 3
CHIPSET = "qualcomm-snapdragon-8-elite"
QDC_TARGET = "SM8750"
# On-device paths (same as the pinned Android driver).
DEVICE_ROOT = "/data/local/tmp/pkg-geniex"
DEVICE_BUNDLE = f"{DEVICE_ROOT}/qairt_bundles/{MODEL_ID}"
DEVICE_RESULTS = "/data/local/tmp/QDC_logs/results"
DEVICE_CACHE = "/data/local/tmp/geniex-cache"
# Unversioned "latest stable" mirror, same as geniex_version=None upstream.
BENCH_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
    "qai-hub-geniex/geniex-bench-android-arm64.tar.gz"
)

DEFAULT_CONTEXT_LENGTHS = [512, 4096]


def build_matrix_tsv(ctx: int) -> str:
    """One benchmark row, tab-separated, exactly as the pinned driver
    writes it: cell_id, plugin, device alias, model reference, then four
    empty columns (variant/image slots unused here)."""
    cell_id = f"{MODEL_ID}-{PLUGIN}-{DEVICE_ALIAS}-c{ctx}"
    return f"{cell_id}\t{PLUGIN}\t{DEVICE_ALIAS}\t{DEVICE_BUNDLE}\t\t\t\t\n"


def bench_command(ctx: int) -> str:
    """The on-device geniex-bench call for one context length.

    Flags mirror the pinned driver's qairt branch: -r 3 repetitions,
    -n 128 generated tokens, the bundle's own sample prompt, and a
    context length that must be one of the compiled ones.
    """
    lib = f"{DEVICE_ROOT}/lib"
    env = (
        f"LD_LIBRARY_PATH={lib}:{lib}/llama_cpp:{lib}/qairt "
        f"ADSP_LIBRARY_PATH={lib} "
        f"GENIEX_PLUGIN_PATH={lib}"
    )
    return (
        f"cd {DEVICE_ROOT} && {env} ./bin/geniex-bench "
        f"--matrix-file {DEVICE_ROOT}/matrix-{ctx}.tsv "
        f"--output-json-dir {DEVICE_RESULTS} -r {REPS} "
        f"-c {ctx} -n {N_GEN} "
        f"--prompt-file {DEVICE_BUNDLE}/sample_prompt.txt "
        f"--mm-data-dir {DEVICE_CACHE} --chipset '{CHIPSET}'"
    )


def session_commands(
    device_pkg: Path, bundle_dir: Path, context_lengths: list[int], pull_dir: Path
) -> str:
    """The whole session as numbered commands. Written to commands.txt
    and printed. <PEM>, <SESSION_ID>, and <PORT> come from the QDC
    Connect dialog when the session is running."""
    lines = [
        "# QDC interactive session commands - Task 08",
        "# <PEM> = your QDC SSH private key file",
        "# <SESSION_ID>, <PORT> = shown in the session's Connect dialog",
        "",
        "# [terminal 1] the tunnel. Leave it running the whole session.",
        "ssh -i <PEM> -L <PORT>:sa<SESSION_ID>.sa.svc.cluster.local:5037 "
        "-N sshtunnel@ssh.qdc.qualcomm.com",
        "",
        "# [terminal 2] everything below. Local adb server must be off",
        "# so the tunnel port is free.",
        "adb kill-server",
        "export ADB_SERVER_SOCKET=tcp:127.0.0.1:<PORT>",
        "adb devices        # must list one device",
        "",
        "# push geniex-bench + matrix files, then the bundle (the slow part)",
        f"adb push {device_pkg}/. {DEVICE_ROOT}",
        f"adb push {bundle_dir}/. {DEVICE_BUNDLE}",
        "",
        "# make the tools runnable and stage the backend libraries",
        f'adb shell "find {DEVICE_ROOT}/bin -type f -exec chmod 755 {{}} +"',
        f'adb shell "cp {DEVICE_ROOT}/lib/qairt/htp-files/*.so {DEVICE_ROOT}/lib/"',
        f'adb shell "cp {DEVICE_ROOT}/lib/llama_cpp/*.so {DEVICE_ROOT}/lib/"',
        f'adb shell "mkdir -p {DEVICE_CACHE} {DEVICE_RESULTS}"',
        "",
    ]
    for ctx in context_lengths:
        lines += [
            f"# benchmark at context length {ctx}",
            f'adb shell "{bench_command(ctx)}"',
            "",
        ]
    lines += [
        "# pull the result cells back to the Mac",
        f"adb pull {DEVICE_RESULTS} {pull_dir}",
        "",
        "# clean the phone, then press Complete Session in the web UI",
        f'adb shell "rm -rf {DEVICE_ROOT} {DEVICE_CACHE}"',
        "",
        "# afterwards, back in the repo root:",
        f"# uv run python -m edge_slm_lab.qdc_results {pull_dir}/results "
        "--session-id <SESSION_ID>",
    ]
    return "\n".join(lines) + "\n"


def download_bench(tar_path: Path) -> None:
    if tar_path.exists():
        print(f"bench tar already present: {tar_path}")
        return
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading geniex-bench (about 86 MiB) to {tar_path}")
    with urllib.request.urlopen(BENCH_URL) as resp, open(tar_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"downloaded {tar_path.stat().st_size / 2**20:.1f} MiB")


def extract_bench(tar_path: Path, device_pkg: Path) -> None:
    """Unpack the tar into device_pkg, stripping the top-level folder,
    with the same unsafe-path guard as the pinned driver."""
    if (device_pkg / "bin" / "geniex-bench").exists():
        print(f"bench already extracted in {device_pkg}")
        return
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        top = members[0].name.split("/", 1)[0] if members else ""
        for m in members:
            if not m.name.startswith(top + "/") or m.name == top + "/":
                continue
            rel = m.name[len(top) + 1 :]
            dst = device_pkg / rel
            real_dst = dst.resolve()
            if not str(real_dst).startswith(str(device_pkg.resolve()) + "/"):
                raise ValueError(f"unsafe tar member path: {m.name!r}")
            if m.isdir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            with open(dst, "wb") as out:
                shutil.copyfileobj(f, out)
            dst.chmod(m.mode | 0o400)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_execution_manifest(bundle_dir: Path, context_lengths: list[int]) -> dict:
    """Separate what each layer of the runtime stack fixes or controls.

    Reads only the bundle's own metadata.json and genie_config.json.
    Every value is copied from those files or from this build's
    parameters. Nothing is estimated.
    """
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    genie_config = json.loads((bundle_dir / "genie_config.json").read_text())
    engine = genie_config["dialog"]["engine"]
    backend = engine["backend"]
    htp = backend[backend["type"]]
    chipset_attrs = metadata["chipset_attributes"]

    ctx_bins = engine["model"]["binary"]["ctx-bins"]
    binaries = []
    for name in ctx_bins:
        path = bundle_dir / name
        binaries.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )

    return {
        "schema_version": 1,
        "task": 8,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_dir_name": bundle_dir.name,
        "route": "QDC interactive session: learner-driven adb over an SSH tunnel",
        "qnn_context_binary": {
            "role": "the compiled program: graphs, weights, quantization",
            "fixed_at": "compile time",
            "files": binaries,
            "compiled_context_lengths": metadata["genie"]["context_lengths"],
            "graph_pattern": "prompt_ar128 and token_ar1 per context length",
        },
        "qairt": {
            "role": "the runtime that loads context binaries onto the NPU",
            "fixed_at": "SDK version fixed when the bundle was compiled",
            "version": metadata["tool_versions"]["qairt"],
            "backend": backend["type"],
            "backend_extensions": backend["extensions"],
            "kv_dim": htp["kv-dim"],
            "pos_id_dim": htp["pos-id-dim"],
            "rope_theta": htp["rope-theta"],
        },
        "geniex": {
            "role": "the generative-AI interface driving prefill and decode",
            "chosen_at": "run time",
            "plugin": PLUGIN,
            "bench_tool": "geniex-bench (android-arm64, latest stable mirror)",
            "device_alias": DEVICE_ALIAS,
            "n_gen": N_GEN,
            "repetitions_per_cell": REPS,
            "prompt_file": "sample_prompt.txt",
        },
        "chipset": {
            "role": "the silicon the binaries were compiled for",
            "fixed_at": "compile time",
            "name": chipset_attrs["name"],
            "soc": "sm8750",
            "htp_version": chipset_attrs["htp_version"],
            "reference_device": chipset_attrs["reference_device"],
            "qdc_target": QDC_TARGET,
        },
        "runtime_configuration": {
            "role": "editable settings read when GenieX starts",
            "chosen_at": "run time, limited by compile-time choices",
            "benchmark_context_lengths": context_lengths,
            "context_size_limit": genie_config["dialog"]["context"]["size"],
            "sampler": genie_config["dialog"]["sampler"],
            "n_threads": engine["n-threads"],
            "note": "-c must name a compiled context length; others fail",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where the device package, tar, and pulled results live (T9)",
    )
    parser.add_argument(
        "--context-lengths",
        default=",".join(str(c) for c in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated; each must be compiled into the bundle",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("results/08_execution_manifest.json"),
    )
    args = parser.parse_args(argv)

    context_lengths = [int(c) for c in args.context_lengths.split(",")]
    if not (args.bundle_dir / "genie_config.json").is_file():
        print(f"ERROR: bundle dir looks wrong: {args.bundle_dir}", file=sys.stderr)
        return 1

    compiled = json.loads((args.bundle_dir / "metadata.json").read_text())["genie"][
        "context_lengths"
    ]
    bad = [c for c in context_lengths if c not in compiled]
    if bad:
        print(
            f"ERROR: context lengths {bad} are not compiled into the bundle "
            f"(compiled: {compiled})",
            file=sys.stderr,
        )
        return 1

    device_pkg = args.out_dir / "device_pkg"
    device_pkg.mkdir(parents=True, exist_ok=True)
    tar_path = args.out_dir / "geniex-bench-android-arm64.tar.gz"
    download_bench(tar_path)
    extract_bench(tar_path, device_pkg)
    if not (device_pkg / "bin" / "geniex-bench").exists():
        print("ERROR: bin/geniex-bench missing after extraction", file=sys.stderr)
        return 1

    for ctx in context_lengths:
        (device_pkg / f"matrix-{ctx}.tsv").write_text(build_matrix_tsv(ctx))

    pull_dir = args.out_dir / "pulled"
    pull_dir.mkdir(parents=True, exist_ok=True)
    commands = session_commands(device_pkg, args.bundle_dir, context_lengths, pull_dir)
    (args.out_dir / "commands.txt").write_text(commands)

    manifest = build_execution_manifest(args.bundle_dir, context_lengths)
    problems = check_text(json.dumps(manifest))
    if problems:
        print(f"ERROR: manifest failed the secret scan: {problems}", file=sys.stderr)
        return 1
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")

    pkg_bytes = sum(p.stat().st_size for p in device_pkg.rglob("*") if p.is_file())
    print(f"device package: {device_pkg}  ({pkg_bytes / 2**20:.1f} MiB)")
    print(f"commands:       {args.out_dir / 'commands.txt'}")
    print(f"manifest:       {args.manifest_out}")
    print()
    print(commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
