# FinQA Chatbot

A production-leaning chatbot that answers numerical-reasoning questions over
financial filings from the FinQA dataset (Chen et al., 2021). It is built for a
PhD interview at a GPU inference infrastructure startup, so every architectural
choice trades off correctness, latency, and observability rather than just
accuracy.

This is **not** a vanilla Agentic-RAG + Program-of-Thought submission.
Section two of `docs/TECHNICAL_REPORT.md` spells out what is different.

## Why this stack

Most FinQA chatbots this year look like:

1. FAISS over arbitrary text chunks.
2. OpenAI `gpt-4o` emits free-form Python in a PoT prompt.
3. Python `exec` runs the code.
4. LangSmith traces are captured.

Four problems with that recipe:

1. **Hallucinated numbers are invisible.** The executor will happily run a
   program that uses `1,248,991` when the filing says `1,284,991`. The test
   passes because the output is close; the answer is wrong. We solve this with a
   dedicated groundedness verifier that rejects programs containing literals
   that are not present in the retrieved context after normalization.
2. **No unit / scale handling.** FinQA tables are almost always stated in
   "millions" or "thousands". A program that answers in raw units against a
   gold answer that is normalized is off by 1e6. A dedicated `ScaleExtractor`
   (regex + model fallback) attaches an explicit `scale_factor` to every chunk.
3. **GPU inference is treated as a black box.** Most submissions call an API.
   The interview panel wants to see vLLM quantization choices, prefix-cache hit
   rates, continuous-batching configuration, and an explicit quantization
   sweep. We ship a reproducible `vllm serve` launcher, a benchmark matrix,
   and a Prometheus dashboard for KV-cache usage.
4. **"Add LangSmith" is not observability.** In production, what matters is
   drift (PSI on input embeddings), retrieval hit-rate on a frozen golden set,
   and a calibration curve between self-consistency agreement and correctness.
   We ship Prometheus metrics for all of these plus a Grafana dashboard.

## Repo layout

```
configs/        GPU-specific YAML (t4.yaml, l4.yaml) + eval.yaml (slices, ablations)
docs/           Technical report, ablations, GPU benchmark, production plan, architecture
ops/            Prometheus config + Grafana dashboard JSON
scripts/        Auto-generated: run_vllm.sh (do not edit by hand)
src/finqa_bot/  Typed library code
tests/          pytest suite (no network, no GPU required)
setup.sh        One-command bootstrap for Colab
```

Key library modules:

- `finqa_bot.data` - FinQA downloader and row-granular chunker matching
  `gold_inds` convention (`text_i`, `table_i`).
- `finqa_bot.retrieval` - dense (Qwen3-Embedding) + BM25 + RRF + reranker,
  scale extractor, and hybrid retriever with doc-id filtering.
- `finqa_bot.execution` - deterministic DSL executor + two number parsers
  (one byte-for-byte with the official `evaluate.py`, one semantic).
- `finqa_bot.graph` - LangGraph `StateGraph`: router -> retrieve -> scale ->
  generate (grammar-constrained) -> execute -> verify -> self-consistency ->
  format | refuse.
- `finqa_bot.verification` - groundedness + unit checks.
- `finqa_bot.serving` - `vllm serve` argv builder with GPU-compat guardrails,
  OpenAI-compatible client wrappers, and health/metrics probes.
- `finqa_bot.eval` - async harness with exact FinQA-metric replicas, sympy
  program accuracy, ablation runner, and GPU benchmark.
- `finqa_bot.monitoring` - PSI drift, calibration bins, Prometheus registry.
- `finqa_bot.ui` - FastAPI `/ask` + `/chat` SSE endpoint and Gradio demo.

## One-command bootstrap (Colab or Linux)

```bash
git clone <this repo>
cd finqa-chatbot
bash setup.sh
```

`setup.sh` auto-detects T4 vs L4 via `nvidia-smi`, picks the right config,
installs deps, downloads FinQA, builds the index, boots vLLM + FastAPI +
Gradio, and prints the URLs.

Useful flags:

- `--ingest-only` - download + build index, then exit.
- `--no-vllm` - skip vLLM (CPU / smoke only).
- `--share` - tunnel the Gradio UI publicly.
- `--t4` / `--l4` - force GPU tier.
- `--fin-o1` - run `TheFinAI/Fin-o1-8B` using `configs/l4_fino1.yaml`.
- `--persist-gdrive` - restore/save `data/indices/dev` and `data/raw/finqa/dev.json` to Google Drive.
- `--persist-dir=/path` - restore/save index cache to a custom persistent path.

Example (L4 + Fin-o1-8B):

```bash
bash setup.sh --fin-o1
```

Example (Colab + persistent index cache in Google Drive):

```bash
# after mounting Drive
bash setup.sh --fin-o1 --persist-gdrive
```

With persistence enabled, `setup.sh` restores cached index artifacts before
ingest and syncs them back after ingest, so new Colab sessions can skip the
expensive embedding/index build when cache already exists.

## Running the pieces by hand

```bash
make install         # runtime dependencies
make install-dev     # + pytest / ruff / mypy
make install-serve   # + vllm

make ingest          # download FinQA + build hybrid index
make eval            # run eval on dev[:200], dump JSON
make ablate          # full ablation matrix -> docs/ABLATIONS.md
make bench           # GPU benchmark matrix -> docs/GPU_BENCHMARK.md

finqa-bot api        # FastAPI (SSE + /metrics)
finqa-bot demo       # Gradio (talks to FastAPI)
finqa-bot ask "..."  # one-shot CLI query

make test            # pytest (no network, no GPU)
make lint            # ruff
make typecheck       # mypy
```

## Where to read next

1. `docs/TECHNICAL_REPORT.md` - dataset analysis, method rationale, eval strategy.
2. `docs/ARCHITECTURE.md` - LangGraph diagram and node-by-node contracts.
3. `docs/PRODUCTION.md` - Kubernetes serving plan, LMCache / prefix-cache
   sharing, rollout + observability.
4. `docs/ABLATIONS.md` - produced by `make ablate`. Empty until you run it.
5. `docs/GPU_BENCHMARK.md` - produced by `make bench`.

## License

Apache-2.0.
