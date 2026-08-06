# Qualcomm Edge SLM Lab — common commands. All Python work goes through uv.

.PHONY: env install install-qualcomm lab test check-json

env:
	uv venv --python 3.11

install:
	uv pip install -r requirements-core.txt

install-qualcomm:
	uv pip install -r requirements-qualcomm.txt

lab:
	uv run jupyter lab

test:
	uv run pytest

check-json:
	uv run python -c "import json,glob; [json.load(open(p)) for p in glob.glob('progress/**/*.json', recursive=True)]; print('all JSON valid')"
