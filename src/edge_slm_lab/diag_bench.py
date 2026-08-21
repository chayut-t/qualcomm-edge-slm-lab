"""Task 09 small diagnostic benchmark.

Measures TTFT and decode tokens/s separately for eight prompts on the
local float Qwen3-0.6B, and imports the Task 08 device medians as
labeled unlike-runtime rows. One warm-up run, then three measured runs
per prompt. Results append to a JSONL file, one row per measured run.

Usage (from the repo root):

    uv run python -m edge_slm_lab.diag_bench --run-id fp32-a
    uv run python -m edge_slm_lab.diag_bench --run-id fp32-b
    uv run python -m edge_slm_lab.diag_bench --run-id bf16 --dtype bfloat16
    uv run python -m edge_slm_lab.diag_bench --check-repro fp32-a fp32-b
    uv run python -m edge_slm_lab.diag_bench --run-id qdc-774303 \
        --import-geniex /Volumes/T9/qualcomm-edge-slm-lab/artifacts/task08/pulled/results

Missing metrics are recorded as null, never estimated. Every line must
pass the check_text secret scan before it is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from edge_slm_lab.qdc_results import parse_cells_from_dir
from edge_slm_lab.sanitize import check_text

MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPTS = Path("configs/benchmark_prompts.json")
DEFAULT_OUT = Path("results/09_diag_bench.jsonl")
DEFAULT_MANIFEST = Path("results/08_execution_manifest.json")
GEN_TOKENS_MIN, GEN_TOKENS_MAX = 32, 64
CATEGORIES = ("instruction", "json_extraction", "summarization", "multilingual")
TARGETS = (32, 128, 512)
FILLER_SLOT = "[FILLER]"
MAX_FILLER_REPEATS = 400
UNLIKE_RUNTIME_WARNING = (
    "Unlike runtime path: w4a16 on a Snapdragon NPU vs local float on this "
    "machine, different prompt and sampler. Do not compare with "
    "pytorch_local rows directly."
)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def peak_rss_bytes() -> int:
    """Process-wide peak resident memory. On macOS ru_maxrss is bytes."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v if sys.platform == "darwin" else v * 1024)


def expand_filler(text: str, filler: str, target_tokens: int, count_tokens) -> str:
    """Repeat the filler sentence until the templated prompt reaches target.

    count_tokens(text) -> int must count tokens the same way the real
    benchmark does (chat template included). Picks the repeat count whose
    token count is closest to target_tokens.
    """
    if FILLER_SLOT not in text:
        return text

    def with_n(n: int) -> str:
        return text.replace(FILLER_SLOT, " ".join([filler] * n) if n else "")

    prev = with_n(0)
    prev_count = count_tokens(prev)
    for n in range(1, MAX_FILLER_REPEATS + 1):
        cand = with_n(n)
        count = count_tokens(cand)
        if count >= target_tokens:
            if abs(count - target_tokens) <= abs(prev_count - target_tokens):
                return cand
            return prev
        prev, prev_count = cand, count
    return prev


def build_prompts(config: dict, count_tokens) -> list[dict]:
    """Validate the prompt set against the task brief and expand fillers."""
    prompts = config.get("prompts") or []
    if len(prompts) != 8:
        raise ValueError(f"the brief fixes eight prompts, got {len(prompts)}")
    per_cat = {c: sum(1 for p in prompts if p["category"] == c) for c in CATEGORIES}
    if any(n != 2 for n in per_cat.values()):
        raise ValueError(f"need exactly two prompts per category, got {per_cat}")
    if not any(p.get("language") == "th" for p in prompts):
        raise ValueError("the brief requires a Thai prompt")
    if set(p["target_tokens"] for p in prompts) != set(TARGETS):
        raise ValueError(f"target_tokens must cover {TARGETS}")

    built = []
    for p in prompts:
        text = expand_filler(
            p["text"], p.get("filler", ""), p["target_tokens"], count_tokens
        )
        built.append(
            {
                "id": p["id"],
                "category": p["category"],
                "language": p.get("language"),
                "target_tokens": p["target_tokens"],
                "text": text,
                "prompt_tokens": count_tokens(text),
            }
        )
    return built


def json_validity(text: str):
    """True if the output contains one parseable JSON object, else False."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        json.loads(text[start : end + 1])
        return True
    except json.JSONDecodeError:
        return False


def detect_failure(new_ids: list, text: str):
    """Visible output failures only. None means no failure detected."""
    if not new_ids or not text.strip():
        return "empty_output"
    if len(new_ids) >= 12 and len(set(new_ids[-12:])) == 1:
        return "token_loop"
    return None


def make_row(env: dict, run_id: str, prompt: dict, rep, measure: dict) -> dict:
    """One JSONL row. Missing metrics stay null, never estimated."""
    text = measure.get("text") or ""
    new_ids = measure.get("new_ids") or []
    json_valid = (
        json_validity(text) if prompt["category"] == "json_extraction" else None
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_utc": utcnow(),
        **env,
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "language": prompt.get("language"),
        "target_tokens": prompt.get("target_tokens"),
        "prompt_tokens": prompt.get("prompt_tokens"),
        "gen_tokens": len(new_ids) if new_ids else measure.get("gen_tokens"),
        "rep": rep,
        "aggregation": measure.get("aggregation", "single_run"),
        "ttft_ms": measure.get("ttft_ms"),
        "prefill_tps": measure.get("prefill_tps"),
        "decode_tps": measure.get("decode_tps"),
        "peak_rss_bytes": measure.get("peak_rss_bytes"),
        "json_valid": measure.get("json_valid", json_valid),
        "output_failure": measure.get(
            "output_failure", detect_failure(new_ids, text) if new_ids else None
        ),
        "output_sha256": sha256_text(text) if text else None,
        "output_chars": len(text) if text else None,
        "unlike_runtime_warning": measure.get("unlike_runtime_warning"),
    }


def bench_all(
    prompts: list[dict],
    runner,
    decode_text,
    env: dict,
    run_id: str,
    gen_tokens: int,
    reps: int,
    warmup: int,
    peak_fn=peak_rss_bytes,
    log=print,
) -> list[dict]:
    """runner(text, gen_tokens) -> (new_ids, ttft_s, decode_s, steps)."""
    rows = []
    for prompt in prompts:
        log(f"[{prompt['id']}] prompt_tokens={prompt['prompt_tokens']} "
            f"(target {prompt['target_tokens']})")
        for _ in range(warmup):
            runner(prompt["text"], gen_tokens)
        shas = []
        for rep in range(1, reps + 1):
            new_ids, ttft_s, decode_s, steps = runner(prompt["text"], gen_tokens)
            text = decode_text(new_ids)
            prompt_tokens = prompt["prompt_tokens"]
            measure = {
                "new_ids": new_ids,
                "text": text,
                "ttft_ms": round(ttft_s * 1000.0, 3),
                "prefill_tps": round(prompt_tokens / ttft_s, 1) if ttft_s > 0 else None,
                "decode_tps": (
                    round(steps / decode_s, 2) if steps > 0 and decode_s > 0 else None
                ),
                "peak_rss_bytes": peak_fn(),
            }
            row = make_row(env, run_id, prompt, rep, measure)
            rows.append(row)
            shas.append(row["output_sha256"])
            preview = text.replace("\n", " ")[:60]
            log(f"  rep {rep}: TTFT={row['ttft_ms']} ms  "
                f"decode={row['decode_tps']} tok/s  gen={row['gen_tokens']}  "
                f"out='{preview}'")
        if len(set(shas)) != 1:
            log(f"  WARNING: outputs differ across reps for {prompt['id']} "
                "(greedy decode should be deterministic)")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        line = json.dumps(row, ensure_ascii=False)
        problems = check_text(line)
        if problems:
            print(f"ERROR: row failed the secret scan: {problems}", file=sys.stderr)
            raise SystemExit(1)
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_repro(rows: list[dict], run_a: str, run_b: str, log=print) -> bool:
    """Two runs reproduce when every (prompt, rep) has the same output hash."""

    def index(run_id):
        return {
            (r["prompt_id"], r["rep"]): r
            for r in rows
            if r["run_id"] == run_id and r["runtime"] == "pytorch_local"
        }

    ra, rb = index(run_a), index(run_b)
    if not ra or not rb:
        log(f"ERROR: no pytorch_local rows for '{run_a}' or '{run_b}'")
        return False
    shared = sorted(set(ra) & set(rb))
    ok = set(ra) == set(rb)
    if not ok:
        log(f"row sets differ: {sorted(set(ra) ^ set(rb))}")
    for key in shared:
        a, b = ra[key], rb[key]
        match = a["output_sha256"] == b["output_sha256"]
        ok = ok and match
        log(f"{key[0]:<12} rep {key[1]}  tokens {'SAME' if match else 'DIFFER'}  "
            f"TTFT {a['ttft_ms']} vs {b['ttft_ms']} ms  "
            f"decode {a['decode_tps']} vs {b['decode_tps']} tok/s")
    log(f"reproducibility: {'PASS' if ok else 'FAIL'} "
        "(same greedy tokens; timing may vary a little)")
    return ok


def import_geniex(results_dir: Path, manifest_path: Path, run_id: str) -> list[dict]:
    """Task 08 device medians as unlike-runtime rows, nulls for the rest."""
    manifest = json.loads(manifest_path.read_text())
    chipset = manifest["chipset"]
    bundle_bytes = sum(
        f["size_bytes"] for f in manifest["qnn_context_binary"]["files"]
    )
    env = {
        "runtime": "geniex_qairt_qdc",
        "device": f"{chipset['reference_device']} ({chipset['qdc_target']})",
        "precision": "w4a16",
        "context_limit": manifest["runtime_configuration"]["context_size_limit"],
        "model_size_bytes": bundle_bytes,
        "model_id": MODEL_ID,
    }
    rows = []
    for cell in parse_cells_from_dir(results_dir):
        prompt = {
            "id": cell["cell_id"],
            "category": None,
            "language": None,
            "target_tokens": None,
            "prompt_tokens": cell["prompt_tokens"],
        }
        measure = {
            "gen_tokens": cell["gen_tokens"],
            "aggregation": "median_of_3",
            "ttft_ms": cell["ttft_ms"],
            "prefill_tps": cell["prefill_tps"],
            "decode_tps": cell["decode_tps"],
            "peak_rss_bytes": None,
            "json_valid": None,
            "output_failure": None,
            "unlike_runtime_warning": UNLIKE_RUNTIME_WARNING,
        }
        rows.append(make_row(env, run_id, prompt, None, measure))
    return rows


def run_local(args) -> int:
    import platform

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def to_ids(text):
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
        )

    def count_tokens(text) -> int:
        return int(to_ids(text).shape[1])

    prompts = build_prompts(json.loads(args.prompts.read_text()), count_tokens)
    if args.only:
        prompts = [p for p in prompts if p["id"] == args.only]
        if not prompts:
            print(f"ERROR: no prompt with id '{args.only}'", file=sys.stderr)
            return 1

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Loading {MODEL_ID} on {device.type} as {args.dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = model.to(device).eval()

    if device.type == "mps":
        sync = torch.mps.synchronize
    elif device.type == "cuda":
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None  # noqa: E731

    eos_ids = {tok.eos_token_id}
    cfg_eos = model.generation_config.eos_token_id
    if isinstance(cfg_eos, int):
        eos_ids.add(cfg_eos)
    elif cfg_eos:
        eos_ids.update(cfg_eos)

    def runner(text, gen_tokens):
        input_ids = to_ids(text).to(device)
        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, use_cache=True)
            next_id = int(out.logits[0, -1].argmax())  # forces device sync
            sync()
            ttft_s = time.perf_counter() - t0
            cache = out.past_key_values
            new_ids = [next_id]
            steps = 0
            sync()
            t1 = time.perf_counter()
            while len(new_ids) < gen_tokens and next_id not in eos_ids:
                step_input = torch.tensor([[next_id]], device=device)
                out = model(input_ids=step_input, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                next_id = int(out.logits[0, -1].argmax())
                new_ids.append(next_id)
                steps += 1
            sync()
            decode_s = time.perf_counter() - t1
        return new_ids, ttft_s, decode_s, steps

    def decode_text(ids):
        return tok.decode(ids, skip_special_tokens=True)

    env = {
        "runtime": "pytorch_local",
        "device": f"{device.type} ({platform.machine()}, {platform.system()})",
        "precision": args.dtype,
        "context_limit": int(model.config.max_position_embeddings),
        "model_size_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        "model_id": MODEL_ID,
    }
    rows = bench_all(
        prompts, runner, decode_text, env, args.run_id,
        args.gen_tokens, args.reps, args.warmup,
    )
    write_rows(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None, help="Name for this run")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gen-tokens", type=int, default=48)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--only", default=None, help="Run one prompt id only")
    parser.add_argument("--import-geniex", type=Path, default=None,
                        help="Folder of pulled Task 08 geniex-bench cell JSONs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-repro", nargs=2, metavar=("RUN_A", "RUN_B"))
    args = parser.parse_args(argv)

    if args.check_repro:
        ok = check_repro(load_rows(args.out), *args.check_repro)
        return 0 if ok else 1

    if args.run_id is None:
        print("ERROR: --run-id is required", file=sys.stderr)
        return 1
    existing = load_rows(args.out)
    if any(r["run_id"] == args.run_id for r in existing):
        print(f"ERROR: run_id '{args.run_id}' already exists in {args.out}. "
              "Pick a new one.", file=sys.stderr)
        return 1

    if args.import_geniex:
        rows = import_geniex(args.import_geniex, args.manifest, args.run_id)
        if not rows:
            print("ERROR: no geniex-bench cells found", file=sys.stderr)
            return 1
        write_rows(args.out, rows)
        print(f"Imported {len(rows)} device rows to {args.out} "
              "(labeled unlike-runtime)")
        return 0

    if not GEN_TOKENS_MIN <= args.gen_tokens <= GEN_TOKENS_MAX:
        print(f"ERROR: --gen-tokens must be {GEN_TOKENS_MIN}-{GEN_TOKENS_MAX} "
              "(fixed by the task brief)", file=sys.stderr)
        return 1
    return run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
