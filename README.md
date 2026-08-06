# Qualcomm Edge SLM Lab

Qwen3-0.6B from PyTorch to Snapdragon. A 14-day personal learning project.

## Goal

Learn how a small open-weight language model changes from a dynamic
Hugging Face model into a static, quantized, chipset-specific program
that runs on a Snapdragon NPU.

The path:

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

## How this works

Ten tasks, one at a time. For each task:

1. Ask Claude Code: `Give me my next task.`
2. Read the task's HTML lesson in `docs/tasks/`.
3. Run the generated notebook or Python code yourself.
4. Change at least one important input when the lesson asks.
5. Ask Claude Code: `Review my current task.`
6. When the task passes its gate, ask for the next one.

Claude generates each task only when it becomes active. Future tasks
do not exist yet.

## Commands

Say the plain sentence or type the slash command. Both work the same.

| Sentence | Skill |
|---|---|
| `Give me my next task.` | `/next-task` |
| `Review my current task.` | `/review-task` |
| `Help me with the current task.` | `/help-task` |
| `Show my progress.` | `/show-progress` |

## Prerequisites

- macOS or Linux with Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- A Hugging Face account (for model downloads)
- A Qualcomm AI Hub account and API token
- Optional: Qualcomm Device Cloud access, RunPod (stretch work only)

## Setup

```bash
uv venv --python 3.11
uv pip install -r requirements-core.txt
```

Credentials go in `.ai-local/secrets/` (gitignored). Never commit tokens.

## Start

Open `docs/index.html` in a browser. Then ask Claude Code for your task.
