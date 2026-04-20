# FinQA chatbot developer Makefile.
# The one-command Colab entry point is `bash setup.sh`. This Makefile is for
# day-to-day local development.

PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help install install-dev install-serve lint typecheck test test-fast \
	smoke ingest serve serve-fino1 serve-stop eval ablate bench ui clean distclean

help:
	@echo "FinQA chatbot targets:"
	@echo "  install        install runtime dependencies"
	@echo "  install-dev    install dev + test dependencies"
	@echo "  install-serve  install vllm (GPU only)"
	@echo "  lint           run ruff"
	@echo "  typecheck      run mypy"
	@echo "  test           run full test suite"
	@echo "  test-fast      run fast tests only (excludes slow, gpu, integration)"
	@echo "  smoke          tiny end-to-end smoke test"
	@echo "  ingest         build the FinQA retrieval index"
	@echo "  serve          start vllm OpenAI-compatible server + FastAPI + Gradio UI"
	@echo "  serve-fino1    start stack with Fin-o1-8B profile (L4)"
	@echo "  serve-stop     stop any local vllm / uvicorn processes"
	@echo "  eval           run eval on FinQA dev (200 sample default)"
	@echo "  ablate         run the full ablation matrix"
	@echo "  bench          run the GPU benchmark matrix"
	@echo "  ui             start only the Gradio chat UI (requires serve running)"
	@echo "  clean          remove pycache, caches, build artifacts"
	@echo "  distclean      clean + remove data / runs / indices"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,openai]"

install-serve:
	$(PIP) install -e ".[serve]"

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow and not gpu and not integration"

smoke:
	$(PY) -m pytest -m integration -q

ingest:
	$(PY) -m finqa_bot.cli ingest --split dev

serve:
	bash setup.sh

serve-fino1:
	bash setup.sh --fin-o1

serve-stop:
	- pkill -f "vllm.entrypoints.openai.api_server" || true
	- pkill -f "finqa_bot.ui.api" || true
	- pkill -f "finqa_bot.ui.gradio_app" || true

eval:
	$(PY) -m finqa_bot.cli eval --split dev --n 200

ablate:
	$(PY) -m finqa_bot.cli ablate --out docs/ABLATIONS.md

bench:
	$(PY) -m finqa_bot.cli bench --out docs/GPU_BENCHMARK.md

ui:
	$(PY) -m finqa_bot.ui.gradio_app

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

distclean: clean
	rm -rf data/raw data/indices data/checkpoints data/cache data/hf data/logs runs outputs
