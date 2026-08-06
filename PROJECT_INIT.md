# Project Initialization Brief: Qualcomm Edge SLM Lab

> **Instruction to Claude Code:** Read this entire file before acting. This is a learning project. Initialize only the small project framework described in **Section 14**, prepare Task 01, validate the framework, and stop. Do not generate lessons, notebooks, code, tests, or answers for future tasks.

## 1. Project identity

- **Suggested folder and GitHub repository name:** `qualcomm-edge-slm-lab`
- **Project title:** Qualcomm Edge SLM Lab: Qwen3-0.6B from PyTorch to Snapdragon
- **Duration:** 14 calendar days
- **Working pattern:** 10 focused tasks, normally 2.5-3.5 hours each
- **Primary model:** `Qwen/Qwen3-0.6B`
- **Pinned Qualcomm reference:** `qualcomm/ai-hub-models` tag `v0.59.0`
- **Main deployment precision:** W4A16
- **Target:** a compatible Snapdragon device through Qualcomm Device Cloud (QDC), or a hosted Qualcomm device through AI Hub Workbench

## 2. Goal

Learn how a small open-weight language model changes from a dynamic Hugging Face model into a static, quantized, chipset-specific program that runs on a Snapdragon NPU.

This is not a product-development project. The main deliverable is the learner's understanding, built by reading concise HTML lessons and running code personally. The learner is not required to keep a written learning journal.

By the end, the learner should be able to explain this flow without Claude Code:

```text
Hugging Face model
    -> autoregressive prefill and decode
    -> explicit tensor and KV-cache contracts
    -> static graph / ONNX representation
    -> Qualcomm model adapter
    -> AIMET quantization concepts
    -> QNN context binary compiled for a chipset
    -> QAIRT and GenieX execution
    -> measurement on Snapdragon hardware
```

## 3. The required learning loop

The project must follow this loop for every task:

1. The learner asks Claude Code: **`Give me my next task.`**
2. Claude reads the project state and generates one task pack only.
3. The learner opens and reads that task's HTML lesson.
4. Claude generates a complete, runnable notebook or Python lab. Do not leave writing assignments or large blank `TODO` sections.
5. The learner runs the notebook or regular Python code personally.
6. The learner changes at least one important input when the lesson asks, then inspects the output.
7. The notebook or script records objective evidence automatically: executed cells, test results, output files, and important measurements.
8. The learner asks Claude: **`Review my current task.`**
9. Claude inspects the saved outputs and checks. It helps fix problems and marks the task complete when the execution gate is satisfied.
10. The learner asks for the next task.

Claude must not generate all course material during initialization. Just-in-time generation is intentional: each lesson should use results, questions, and problems from the previous task.

## 4. Commands the learner will use

Claude Code must recognize these plain-language commands.

During initialization, also create each command as a project-local Claude Code skill under `.claude/skills/`:

- `next-task` for `Give me my next task.`
- `review-task` for `Review my current task.`
- `help-task` for `Help me with the current task.`
- `show-progress` for `Show my progress.`

Each `SKILL.md` holds that command's rules from this section. The learner can type `/next-task` or say the plain sentence. Both must behave the same.

### `Read PROJECT_INIT.md and initialize the project.`

Create the minimal framework, generate Task 01, run lightweight validation, and stop.

### `Give me my next task.`

- Read `progress/state.json` and the latest automatically recorded run evidence.
- If a task is already active, point the learner back to it. Do not create another task.
- If the prior task is complete, generate only the next task pack.
- Tell the learner exactly what to read and run, and which outputs to inspect.
- Stop before running the learner's local lab. The learner controls execution.

### `Review my current task.`

- Inspect the current task's executed notebook, code, tests, and generated outputs.
- Give focused hints for incomplete or incorrect work.
- Mark the task complete only when its gate is satisfied.
- Do not generate the next task until the learner asks for it.

### `Help me with the current task.`

- Diagnose only the current blocker.
- Give one useful hint or the smallest code change first.
- Prefer a small fix or a focused explanation. Keep the lab runnable.
- Do not unlock future tasks.

### `Show my progress.`

Summarize completed tasks, the active task, execution evidence, current blockers, and the next locked task. Do not generate new task content.

## 5. Learning-first contract for Claude Code

Create `CLAUDE.md` from these rules during initialization. Preserve their meaning.

### 5.1 Role

Act as a technical coach and pair programmer. Optimize for understanding, not file count or speed of implementation.

### 5.2 One task at a time

- Only one task may have status `active` or `awaiting_review`.
- Generate only the files needed for the active task.
- Do not create placeholder lesson pages or notebooks for future tasks.
- Do not silently start the next task.
- Use previous run results and errors to adjust the next lesson's explanation and difficulty.

### 5.3 Before the learner runs code

For every meaningful experiment:

1. Point the learner to the relevant section of the HTML lesson.
2. Provide the smallest runnable experiment that demonstrates the idea.
3. State which output, shape, graph, or performance behavior to watch.
4. Wait for the learner to run it.

Predictions may appear as optional thought prompts in the HTML page, but the learner must not be required to write them down.

### 5.4 Notebook policy

Prefer a notebook when the learner benefits from:

- running code cell by cell;
- changing values;
- viewing tensor shapes or intermediate values;
- plotting results;
- comparing two parameter settings or two runtime behaviors.

Each notebook must contain:

- a short purpose and environment note;
- a link to the matching HTML lesson;
- small executable steps;
- complete runnable code generated by Claude;
- clearly labeled parameters the learner can change;
- assertions or checks that give useful feedback;
- short notes beside important outputs that explain what to inspect;
- a final automated summary cell that reports checks and saved artifacts.

Do not pre-run a notebook or embed final outputs. The learner must run it.
Do not require markdown answers, predictions, observations, or reflections inside the notebook.

### 5.5 Regular Python policy

Use regular Python for reusable, testable, automated, command-line, or remote-device work. When regular Python is more appropriate:

1. Explain the file and its data flow in the matching HTML lesson.
2. State important inputs, outputs, and tensor shapes.
3. Generate a complete implementation that is small enough to read.
4. Add a focused test or self-check.
5. Ask the learner to run the command, change a meaningful option when useful, and inspect the result.

### 5.6 Explanation policy

- Use simple English suitable for a non-native speaker.
- Use correct technical jargon when it improves precision.
- Define new jargon at first use and add it to `docs/glossary.html`.
- Prefer short sentences and concrete examples.
- Keep exact tensor names, dimensions, formulas, commands, and APIs.
- Explain every symbol used in a formula.
- Label facts, experimental observations, and inferences clearly.
- Link primarily to official Qualcomm, Qwen, PyTorch, ONNX, and AIMET sources.
- Follow the learner's "Plain" output style in lessons, pages, and messages: short sentences (about 20 words max), common words, one word for one meaning, no slang or idioms.
- Never simplify code, paths, commands, flags, identifiers, or error text. Copy them exactly. Simplify the words around them.

### 5.7 Completion policy

Generated files do not prove learning. A task is complete only when:

- the learner ran the important notebook cells or commands;
- required automated checks pass, or a real external blocker is recorded by Claude;
- expected output artifacts or measurements exist;
- the learner inspected the specific outputs highlighted in the HTML lesson;
- any required parameter-change experiment was run.

Do not require essays, written predictions, written observations, reflection paragraphs, or a teach-back. If a concept check is useful, use a few optional multiple-choice questions in HTML or a notebook self-check. They must not block progress.

### 5.8 Safety and repository hygiene

- Never print or commit tokens, credentials, or signed download URLs.
- Store secrets only in gitignored locations (`.ai-local/`, `.env`). Never in tracked files, code, or notebooks.
- Before every git commit, scan the staged changes for secrets: API keys, tokens, signed URLs, credential file paths. Do not commit if anything matches.
- At initialization, create a git pre-commit hook that blocks commits containing likely secrets. Keep the hook simple and readable.
- Never commit model weights, QNN context binaries, large ONNX files, datasets, or benchmark caches.
- Ask before starting paid RunPod resources.
- Do not install system packages globally.
- Pin a known working environment and record version changes.
- Never fabricate performance results or fake passing tests.

## 6. Task pack contract

When a task starts, Claude generates one self-contained task pack.

### 6.1 Required task-pack files

Every task creates:

```text
docs/tasks/<NN>-<task-name>.html
progress/runs/<NN>-run.json
```

It also creates only the notebook, source file, test, config, or tool needed for that task. A task with no useful coding lab should use a code-reading worksheet or a small command exercise instead of forcing a notebook.

### 6.2 Required HTML lesson structure

Every task page must contain:

1. Why this matters for Qualcomm work
2. What the learner will understand after the task
3. Prerequisites from earlier tasks
4. A simple mental model
5. Exact technical details, tensor shapes, or runtime boundaries
6. One worked example
7. An optional "Think before running" prompt with no answer field
8. Hands-on instructions in execution order
9. How to read the generated notebook or Python files
10. Exact outputs, shapes, graphs, or measurements to inspect
11. Common mistakes and debugging clues
12. A five-question knowledge check with answers inside collapsed `<details>` elements
13. Completion gate based on execution evidence
14. A page glossary and official sources

### 6.3 HTML usability

The learning pages are plain static HTML. They must work when `docs/index.html` is opened directly. Do not use React, Next.js, a build step, CDN resources, or a required web server.

Use:

- an 850-950 px maximum reading width;
- readable type and generous line spacing;
- visible keyboard focus and accessible contrast;
- responsive layout for a laptop or tablet;
- horizontally scrollable code blocks;
- distinct callouts for `Key idea`, `Watch for`, `Try it`, `Warning`, and `Completion gate`;
- navigation to the index, glossary, progress page, and previous completed task;
- small inline SVG diagrams only when a graph or tensor flow is clearer than prose.

### 6.4 Task handoff message

After generating a task pack, Claude must stop and give a short handoff in this order:

1. Open this HTML lesson.
2. Open this notebook or file.
3. Run these cells or commands yourself.
4. Change this parameter if the task includes a comparison.
5. Inspect these specific outputs.
6. Save the executed notebook or generated results, then say `Review my current task`.

## 7. Project scope

### 7.1 Required core

- Run Qwen3-0.6B with Hugging Face.
- Understand Qwen architecture, GQA, RoPE, and attention shapes.
- Run and inspect a small greedy-decoding loop.
- Inspect KV-cache growth.
- Compare prefill and one-token decode contracts.
- Inspect an ONNX graph or reduced static wrapper.
- Read the relevant Qualcomm Qwen adapter and export path.
- Perform a small quantization fundamentals lab.
- Fetch or export a published W4A16 QNN context binary.
- Run or profile the model on real Qualcomm hardware.
- Use GenieX on QDC when a compatible bundle/device combination is available.
- Run a small diagnostic benchmark.
- Trace the complete model-to-NPU flow in a final results notebook.

### 7.2 Optional stretch work

- Run the official AIMET W4A16 quantization flow on RunPod.
- Compare a custom quantized checkpoint with Qualcomm's published checkpoint.
- Compare GenieX `qairt` and `llama.cpp` paths when both are available.

### 7.3 Explicitly out of scope

- Fine-tuning or LoRA
- Android UI development
- A hosted learning website
- Supporting another model architecture
- Writing custom NPU kernels
- Reimplementing Qualcomm's complete exporter
- Production inference serving
- Full speculative decoding
- Large benchmark suites
- Long-context experiments beyond the compiled bundle

If time is short, remove stretch work. Do not skip the learner's hands-on run.

## 8. Technical basis

These details were verified on 2026-08-06. The pinned release is used for reproducibility, even if a newer release exists when the project is run. If an official command or service changes, update only the current task, record the difference in `progress/decision_log.md`, and preserve the learning goal.

### 8.1 Qwen3-0.6B reference configuration

Qualcomm's pinned implementation defines:

- 28 decoder layers;
- hidden size 1,024;
- 16 query attention heads;
- 8 key/value heads;
- explicit head dimension 128;
- grouped-query attention (GQA);
- thinking-mode support;
- W4A16 as the main Qualcomm deployment precision;
- two deployed parts: token embedding, then all transformer layers plus the LM head.

The published Qualcomm quantization recipe is:

```text
SpinQuant R2 + R3 -> AdaScale -> calibration
```

### 8.2 Qualcomm stack mental model

- **AIMET:** model-efficiency and quantization tooling.
- **AI Hub Workbench:** managed compilation, correctness checks, profiling, inference, and downloadable deployment artifacts.
- **QNN context binary:** a compiled graph artifact prepared for a target Qualcomm platform.
- **QAIRT / Qualcomm AI Engine Direct:** the runtime and deployment suite that executes compiled graphs.
- **GenieX:** the public/community generative-AI interface. Its `qairt` path uses chipset-specific NPU bundles; its `llama.cpp` path uses GGUF models and is a different execution path.
- **QDC:** interactive remote access to Qualcomm devices.

Bundle precision, graph shapes, KV-cache layout, context limit, and chipset compatibility may be fixed during compilation. Do not treat them as normal runtime choices.

### 8.3 Compatibility fallback

Do not assume every bundle exists for every QDC device. Task 01 must discover the currently available combination.

Use this order:

1. Qwen3-0.6B W4A16 QAIRT bundle on a compatible QDC device through GenieX.
2. Qwen3-0.6B W4A16 compiled, profiled, and inferred on a hosted AI Hub physical device.
3. Qwen3-0.6B GGUF through GenieX on QDC, plus a separate QAIRT profile through AI Hub. Label these as different runtime paths.

Direct QDC deployment is not required if asset compatibility blocks it. The learner must document the boundary and complete at least one real-device Qualcomm run or profile.

### 8.4 Baseline for the W4A16 comparison

W4A16 needs a reference point for both accuracy and efficiency.

Research result (checked 2026-08-06): Qualcomm publishes only three Qwen3-0.6B checkpoints at `v0.59.0`: `DEFAULT`, `DEFAULT_W4A16`, and `DEFAULT_Q4_0`. There is no published W16A16 on-device bundle, and compiling a custom one is out of scope. So W16A16 is not a practical baseline here.

Use these baselines instead:

- **Accuracy baseline:** the original Hugging Face model in float precision (FP32 or BF16), run locally with PyTorch. Same weights, no quantization error. Compare deterministic outputs and simple quality checks against W4A16 device outputs.
- **Efficiency reference:** record local float TTFT and decode tokens/s next to the W4A16 device numbers. Label them as different runtime paths on different hardware. This shows the deployment gain. It is not a controlled precision comparison.
- **Optional same-device comparison:** `DEFAULT_Q4_0` (GGUF through the GenieX `llama.cpp` path) on the same device, clearly labeled as a different runtime path.

If Task 07 discovery shows the export path supports a float precision on a hosted device, one float profile job may be added as a stretch comparison. Do not assume it exists.

## 9. Ten-task learning path

Each task is generated only when it becomes active.

### Task 01 — Orientation, access, and stack map

**Lab form:** HTML lesson + environment notebook + discovery commands  
**Main output:** automatically generated environment inventory, available device/bundle route, first stack map  
**Core question:** What does each layer of the Qualcomm deployment stack do?

The notebook records OS, architecture, Python, disk, Hugging Face access, QDC access, and AI Hub access. Claude confirms current CLI syntax with `--help` before using commands such as:

```bash
qai-hub-models info Qwen3-0.6B
qai-hub-models perf Qwen3-0.6B
qai-hub-models numerics Qwen3-0.6B
qai-hub-models devices
qai-hub-models chipsets
qai-hub-models runtimes
```

**Gate:** the discovery notebook ran, credentials are safe, and Claude recorded a preferred deployment route plus fallback from the discovered environment.

### Task 02 — Qwen architecture and Hugging Face baseline

**Lab form:** HTML lesson + notebook  
**Main output:** deterministic generation, parameter summary, attention-shape calculations  
**Core question:** How do Qwen's embedding, transformer blocks, GQA, RoPE, and LM head connect?

The notebook calculates and visualizes query, key, and value shapes using 16 query heads, 8 KV heads, and head dimension 128. It then reruns one shape example with a changed sequence length.

**Gate:** all cells ran, shape assertions passed, and the changed-sequence-length comparison was saved.

### Task 03 — Manual decoding and KV cache

**Lab form:** HTML lesson + notebook + small Python module + focused test  
**Main output:** runnable greedy decode loop and cache-shape trace  
**Core question:** Why are prefill and repeated one-token decode different?

Claude generates a compact, fully runnable central loop and explains it line by line in HTML. The learner runs it, changes the number of generated tokens, and watches the cache grow.

**Gate:** deterministic tokens match a short Hugging Face baseline, cache-shape checks pass, and results from two generation lengths are saved.

### Task 04 — Static graph contracts and ONNX

**Lab form:** HTML lesson + notebook + ONNX inspection helper  
**Main output:** prefill/decode contract table, graph input/output list, small numerical comparison  
**Core question:** Why does ahead-of-time NPU compilation need explicit shapes and interfaces?

Use a reduced wrapper if exporting the full Qwen model would distract from the concept.

**Gate:** graph inspection completed, contracts were generated automatically, and the numerical comparison passed its documented tolerance.

### Task 05 — Qualcomm's Qwen model adapter

**Lab form:** HTML lesson + guided source-navigation notebook  
**Main output:** automatically generated map from Qualcomm classes/functions to the deployment stack  
**Core question:** What does Qualcomm add around the original model to make deployment possible?

Read the pinned README, `model.py`, export code, quantization code, and only the shared adapter code needed to follow the path. Do not copy large source blocks.

**Gate:** the notebook located the required classes/functions, generated the two-part boundary map, and all source-location checks passed.

### Task 06 — Quantization and AIMET concepts

**Lab form:** HTML lesson + hands-on fake-quantization notebook  
**Main output:** weight/error plots and explanation of W4A16, scales, grouping, and calibration  
**Core question:** How can low-bit weights reduce cost, and where does error enter?

The core lab implements simple symmetric fake weight quantization on a small linear layer. It compares bit widths and grouping. It clearly distinguishes the teaching code from Qualcomm's full SpinQuant, AdaScale, and AIMET recipe.

The HTML lesson must explain SpinQuant and AdaScale at concept level:

- **SpinQuant:** what the R2 and R3 rotations are, how rotating weights and activations spreads outlier values, and why that lowers quantization error.
- **AdaScale:** what problem per-channel or learned scaling solves, and what would degrade without it.
- Where each step sits in the recipe `SpinQuant R2 + R3 -> AdaScale -> calibration`.

Concept level only. Do not reimplement either method.

**Gate:** the learner ran all quantization cells, changed bit width or group size, and saved the resulting error comparison plots.

**Optional extension after the gate:** run the official quantization flow on RunPod. Ask before starting paid resources and stop debugging after two focused hours.

### Task 07 — Published W4A16 artifact and AI Hub

**Lab form:** HTML lesson + command-line tools  
**Main output:** sanitized artifact manifest and one hosted-device compile/profile/inference record  
**Core question:** What changes during conversion, compilation, and device execution?

Start from current official command help. Possible patterns include:

```bash
qai-hub-models fetch Qwen3-0.6B \
  --runtime qnn_context_binary \
  --precision w4a16

qai-hub-models export qwen3_0_6b \
  --checkpoint DEFAULT_W4A16 \
  --target-runtime qnn_context_binary \
  --device "<verified device name>"
```

Do not paste a device name from this brief without current discovery.

**Gate:** one compatible W4A16 artifact is fetched or exported and one Qualcomm hosted-device job is recorded.

### Task 08 — QAIRT and GenieX on QDC

**Lab form:** HTML lesson + device command script/checklist  
**Main output:** one QDC execution or a clearly documented compatibility fallback  
**Core question:** What is compiled into the bundle, and what does GenieX control at runtime?

**Gate:** the device script completed or the fallback ran, and the generated execution manifest distinguishes the QNN context binary, QAIRT, GenieX, chipset, and runtime configuration.

### Task 09 — Small diagnostic benchmark

**Lab form:** HTML lesson + reusable benchmark script  
**Main output:** small JSONL result set  
**Core question:** Why must TTFT and decode throughput be measured separately?

Use:

- prompt lengths near 32, 128, and 512 tokens;
- one warm-up and three measured runs;
- eight prompts: two instruction, two JSON extraction, two summarization, and two multilingual prompts including Thai;
- 32-64 generated tokens where practical.

The benchmark script records runtime, device, precision, context limit, model/bundle size, TTFT when available, decode tokens/s, peak memory when available, JSON validity, and visible output failures. It uses `null` for missing metrics and never estimates them.

**Gate:** results are reproducible and unlike runtime paths are not compared without a warning.

### Task 10 — Results analysis and end-to-end review

**Lab form:** HTML lesson + results notebook  
**Main output:** plots, limitations, an automatically generated final report, and an interactive pipeline trace  
**Core question:** How do the outputs from all earlier tasks connect into one model-to-NPU path?

**Gate:** the notebook loads the earlier run records, generates the final plots and report, and all end-to-end trace checks pass. A spoken or written teach-back is optional, not required.

## 10. Progressive repository structure

The repository grows as tasks are unlocked. Future task files must not exist yet.

### 10.1 Structure immediately after initialization

```text
qualcomm-edge-slm-lab/
├── PROJECT_INIT.md
├── CLAUDE.md
├── README.md
├── .claude/
│   ├── settings.json
│   └── skills/
│       ├── next-task/SKILL.md
│       ├── review-task/SKILL.md
│       ├── help-task/SKILL.md
│       └── show-progress/SKILL.md
├── .ai-local/                 (gitignored: Claude's workspace and scratch space)
│   └── secrets/
│       └── qai-hub.env        (copied at init)
├── pyproject.toml
├── requirements-core.txt
├── requirements-qualcomm.txt
├── requirements-gpu.txt
├── .gitignore
├── .env.example
├── Makefile
├── docs/
│   ├── index.html
│   ├── glossary.html
│   ├── progress.html
│   ├── assets/
│   │   └── styles.css
│   └── tasks/
│       └── 01-orientation-and-stack.html
├── notebooks/
│   └── 01_environment_and_access.ipynb
├── progress/
│   ├── state.json
│   ├── decision_log.md
│   ├── environment.md
│   └── runs/
│       └── 01-run.json
├── results/
│   └── README.md
└── artifacts/
    └── README.md
```

### 10.2 Expected structure after all tasks

Later files are created only when their task starts:

```text
docs/tasks/01-...html through 10-...html
notebooks/01_environment_and_access.ipynb
notebooks/02_qwen_baseline.ipynb
notebooks/03_manual_decode_and_kv_cache.ipynb
notebooks/04_static_graphs_and_onnx.ipynb
notebooks/06_quantization_fundamentals.ipynb
notebooks/10_results_analysis.ipynb
src/edge_slm_lab/manual_decode.py
src/edge_slm_lab/shape_trace.py
src/edge_slm_lab/onnx_tools.py
src/edge_slm_lab/metrics.py
tools/capture_qualcomm_inventory.py
tools/inspect_bundle.py
tools/run_device_benchmark.py
tests/test_manual_decode.py
tests/test_shape_contracts.py
tests/test_metrics.py
configs/benchmark_prompts.json
progress/runs/01-run.json through 10-run.json
results/final_report.html
```

This is a target, not an instruction to create these files early.

### 10.3 Artifact policy

`artifacts/` stores local models, ONNX graphs, encodings, QNN bundles, and device packages. Git must ignore everything there except `README.md`.

`results/` stores small JSON, JSONL, CSV, Markdown, and plots. Do not store credentials, signed URLs, machine identifiers, or large raw logs.

`.ai-local/` is Claude Code's local workspace. Git ignores all of it. Claude uses it for scratch files, intermediate outputs, working notes, and downloaded helper material. Secrets live only in `.ai-local/secrets/`. Nothing a task needs as evidence may live only in `.ai-local/` — evidence goes to `progress/` or `results/`.

## 11. State and progress rules

Create `progress/state.json` during initialization:

```json
{
  "schema_version": 1,
  "current_task": 1,
  "status": "active",
  "completed_tasks": [],
  "blocked_reason": null,
  "updated_at": "<ISO-8601 UTC timestamp>"
}
```

Allowed statuses are:

- `active`: task pack exists and the learner is working;
- `awaiting_review`: the learner asks Claude to review;
- `task_complete`: the current task passed its gate, and the learner has not asked for the next task yet;
- `blocked`: an external dependency prevents the current path;
- `complete`: used only when Task 10 and the project are complete.

When a task passes review: add its number to `completed_tasks`, set `status` to `task_complete`, and keep `current_task` unchanged. When the learner asks for the next task: increment `current_task`, generate the task pack, and set `status` to `active`.

Do not use a simple file-exists check as completion evidence. Update state atomically and keep `docs/progress.html` consistent with it.

For each task, Claude maintains `progress/runs/<NN>-run.json` from objective evidence. The learner does not edit it manually. Record only:

- which notebook cells or commands ran;
- check and test results;
- important output artifact paths;
- parameters used for the required comparison;
- external blocker details, if any;
- completion timestamp and status.

Keep human-written progress notes optional. Do not ask the learner to summarize what was learned.

## 12. Environment plan

### Local environment: required

Use local resources for HTML, notebooks, Hugging Face baseline, decoding, shape tracing, reduced ONNX work, CLI discovery, and results analysis.

Preferred Python is 3.11. Suggested packages include PyTorch, Transformers, JupyterLab, NumPy, pandas, matplotlib, ONNX, ONNX Runtime, pytest, and `qai-hub-models-cli`.

Use `uv` for Python environment management. Create the environment with `uv venv --python 3.11` and install with `uv pip install -r requirements-core.txt`. Makefile targets must use `uv` too. Do not use plain `pip` or `conda`.

Note: `qai-hub-models-cli` and `qai-hub-models` are two different PyPI packages. Both were verified available at version `0.59.0` on 2026-08-06.

- `qai-hub-models-cli`: the lightweight CLI for discovery commands. Pin `qai-hub-models-cli==0.59.0` in `requirements-core.txt`.
- `qai-hub-models`: the full model package with heavy dependencies. Pin `qai-hub-models==0.59.0` in `requirements-qualcomm.txt`.

Keep Qualcomm model-specific dependencies separate because they are heavier and more platform-sensitive.

### Qualcomm cloud/device environment: required

Use AI Hub Workbench for discovery, compilation, numerical checks, profiling, inference, and artifact download. Use QDC for interactive device work when compatible.

Never store API tokens in the repository. `.env.example` contains variable names only.

At initialization, copy the existing credential file from `/Users/chayut/projects/slm-deployment-lab/.ai-local/secrets/qai-hub.env` to `.ai-local/secrets/qai-hub.env` in this project. Git must ignore `.ai-local/`. Never print, log, or commit its contents.

Use the CLI (`qai-hub`, `qai-hub-models`) for Qualcomm AI Hub operations whenever possible. Avoid the web UI except when a step has no CLI equivalent.

### RunPod CUDA environment: optional

A GPU is not required for the core project. Use RunPod only for optional custom AIMET quantization or faster evaluation after the published W4A16 path works.

For a low-friction optional setup, prefer Linux x86, Python 3.10 or 3.11, a CUDA version compatible with the pinned AIMET wheel, 48 GB VRAM, at least 64 GB host RAM, and 75-100 GB storage. A smaller GPU may work, but the full pipeline uses more memory than raw model weights.

## 13. Concepts covered by the final end-to-end review

1. What happens during prefill and one-token decode?
2. What is stored in the KV cache, and which dimension grows?
3. Why does GQA use fewer KV heads than query heads?
4. Why can `head_dim` differ from `hidden_size / query_heads`?
5. Why does an NPU compiler prefer fixed shapes and explicit interfaces?
6. What crosses the Qualcomm model-part boundary?
7. What does W4A16 mean?
8. What are calibration data and quantization encodings?
9. What do SpinQuant and AdaScale try to improve?
10. How do ONNX, a QNN context binary, QAIRT, and GenieX differ?
11. Why is a bundle tied to a chipset and compile-time configuration?
12. Why are TTFT and decode tokens/second separate metrics?

## 14. Exact initialization instructions for Claude Code

When the learner says **`Read PROJECT_INIT.md and initialize the project`**, do the following and then stop.

### Step 1 — Inspect safely

- Confirm the current directory is the intended project folder.
- Inspect OS, architecture, available Python, Git, disk, and existing files.
- Do not install global packages, download model weights, open paid resources, or display credentials.

### Step 2 — Create the minimal framework

- Create only the structure in Section 10.1.
- Write `CLAUDE.md` from Section 5.
- Write a concise `README.md` containing the goal, learning loop, prerequisites, and learner commands from Section 4.
- Create `.gitignore` before any download. Protect credentials, environments, notebook checkpoints, caches, `artifacts/`, model files, ONNX binaries, QNN bundles, and large results.
- Create `.env.example` with variable names only.
- Create conservative dependency files. Pin `qai-hub-models-cli==0.59.0` in `requirements-core.txt` and `qai-hub-models==0.59.0` in `requirements-qualcomm.txt`, as described in Section 12. If a pinned version is not available on PyPI at initialization time, report the mismatch instead of silently choosing another release.
- Make the repository Claude-Code-native: `CLAUDE.md` at the root, the four command skills in `.claude/skills/` (Section 4), and `.claude/settings.json` with safe project permissions.
- Copy the credential file described in Section 12 into `.ai-local/secrets/qai-hub.env`. Confirm `.ai-local/` is gitignored before copying. Never print its contents.
- Create the Python environment with `uv` as described in Section 12.
- Create the git pre-commit secret-scan hook from Section 5.8.
- Initialize local Git if needed. Do not create or publish a GitHub remote.

### Step 3 — Create the portal shell

- Create shared CSS, index, glossary, and progress pages.
- `docs/progress.html` is the project dashboard. It shows overall status, a short project summary, a task table with statuses, execution-evidence highlights, and current blockers. Keep it consistent with `progress/state.json` after every state change.
- The index shows all 10 task titles and statuses, but links only to generated tasks.
- Locked tasks show only their title and one-sentence goal. Do not write their lesson content.
- Seed the glossary only with terms needed for Task 01. Add later terms just in time.

### Step 4 — Generate Task 01 only

- Create the complete Task 01 HTML lesson using Section 6.
- Create `notebooks/01_environment_and_access.ipynb` with complete runnable cells and an automated final summary cell.
- Create `progress/runs/01-run.json`; Claude updates it from execution evidence, not learner prose.
- Set `progress/state.json` to Task 01 `active`.
- Do not create Task 02 files or placeholder pages.

### Step 5 — Validate lightly

- Validate JSON, notebook structure, HTML internal links, and Python project metadata.
- Open-file links must work without a web server.
- Do not download Qwen or call services requiring credentials during initialization.
- Show the created tree and report platform risks.

### Step 6 — Hand off and stop

Tell the learner:

1. how to open `docs/tasks/01-orientation-and-stack.html`;
2. how to launch `notebooks/01_environment_and_access.ipynb`;
3. which cells and commands the learner should run;
4. which outputs to inspect;
5. how to save the executed notebook so Claude can review it;
6. to say `Review my current task` when finished.

Do not generate Task 02 or offer to solve Task 01.

## 15. Definition of done

The core project is complete when:

- all 10 tasks passed their gates in sequence;
- the learner ran all important notebooks and commands;
- the manual decode and focused tests pass;
- a compatible W4A16 context binary was fetched or exported;
- one real Qualcomm device run or profile was recorded;
- the small benchmark and results analysis are complete;
- the final report was generated from real run records;
- limitations and failed experiments are captured automatically or by Claude;
- no secret or large binary is tracked by Git.

Custom AIMET quantization and speculative decoding are not required.

## 16. Common scope traps

- Generating all lesson pages at initialization
- Filling future notebooks with placeholder or solved code
- Treating generated files as evidence of learning
- Requiring the learner to write predictions, observations, reflections, or reports
- Building a polished web application instead of plain HTML
- Quantizing before the published W4A16 path works
- Treating GGUF and QAIRT as the same runtime
- Exporting the full model from scratch only for the ONNX lesson
- Collecting too many benchmark prompts
- Losing several days to a device queue instead of using the fallback
- Committing model artifacts or credentials

## 17. Glossary seed for Task 01

- **AIMET:** Qualcomm's AI Model Efficiency Toolkit for quantization and model optimization.
- **AI Hub Workbench:** Qualcomm's managed service for compiling, testing, profiling, and running models on hosted devices.
- **Backend:** The lower-level runtime that executes model operations on a compute unit.
- **Context binary:** A compiled QNN graph artifact prepared for a target Qualcomm platform.
- **GenieX:** A public generative-AI interface that coordinates model execution through supported backends.
- **NPU:** Hardware specialized for neural-network computation.
- **QAIRT:** Qualcomm AI Runtime, the deployment and runtime suite also described in current public material as Qualcomm AI Engine Direct.
- **QDC:** Qualcomm Device Cloud, which provides remote access to Qualcomm hardware.
- **QNN:** Qualcomm Neural Network graph and runtime APIs within the deployment stack.

Add terms such as prefill, decode, KV cache, GQA, RoPE, QDQ, encoding, calibration, and W4A16 only when the relevant task starts.

## 18. Official references

Use official sources before blogs or forum posts.

1. Qualcomm AI Hub Models Qwen3-0.6B README, pinned `v0.59.0`:  
   https://github.com/qualcomm/ai-hub-models/blob/v0.59.0/src/qai_hub_models/models/qwen3_0_6b/README.md
2. Qualcomm Qwen3-0.6B adapter code, pinned `v0.59.0`:  
   https://github.com/qualcomm/ai-hub-models/blob/v0.59.0/src/qai_hub_models/models/qwen3_0_6b/model.py
3. Qualcomm AI Hub Models releases:  
   https://github.com/qualcomm/ai-hub-models/releases
4. Qualcomm AI Hub Qwen3-0.6B model page:  
   https://aihub.qualcomm.com/models/qwen3_0_6b
5. Qualcomm AI Hub Workbench documentation:  
   https://app.aihub.qualcomm.com/docs/index.html
6. Qualcomm QAIRT documentation:  
   https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-100/introduction.html
7. GenieX overview:  
   https://geniex.aihub.qualcomm.com/en/get-started/what-is-geniex
8. GenieX platforms and runtimes:  
   https://geniex.aihub.qualcomm.com/en/get-started/platforms
9. GenieX supported models and precision guide:  
   https://geniex.aihub.qualcomm.com/en/models/supported
10. GenieX QAIRT plugin:  
    https://github.com/qualcomm/geniex-qairt-plugin
11. Qwen3-0.6B model card:  
    https://huggingface.co/Qwen/Qwen3-0.6B
12. AIMET releases:  
    https://github.com/quic/aimet/releases

## 19. Final reminder to Claude Code

The learner should finish with fewer mysteries, not merely more files. Generate one learning task, let the learner read and run it, verify real execution evidence, and only then prepare the next task. Keep writing by the learner optional and minimal.
