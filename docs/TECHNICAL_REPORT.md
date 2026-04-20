# FinQA Chatbot Technical Report

## 1. Dataset analysis

FinQA (Chen et al., EMNLP 2021) is a numerical-reasoning QA dataset built from
the earnings reports of 373 S&P 500 companies. 8,281 examples are split into
train (6,251) / dev (883) / test (1,147). Each example contains:

- A short natural-language question.
- `pre_text` - narrative paragraphs appearing before the supporting table.
- `post_text` - narrative paragraphs after the table.
- `table` - a 2D list of strings; row 0 is a header.
- A gold **program** written in a small DSL (10 operators: `add`, `subtract`,
  `multiply`, `divide`, `exp`, `greater`, `table_sum`, `table_average`,
  `table_max`, `table_min`).
- A gold numeric (or `yes`/`no`) **answer**.
- **`gold_inds`** - the subset of `pre_text` / `post_text` / table rows that
  were needed to answer, keyed `text_i` / `table_i`.

### What makes FinQA unique

1. **Mixed-modality context.** The answer frequently requires joining a
   sentence in the narrative ("amounts are in millions") with a row of the
   table ("Revenue: 9,413"). Text-only RAG misses the sentence; table-only
   RAG misses the qualifier.
2. **Implicit units.** The table rarely states its units inline. Whether a
   value is in dollars, thousands, or millions is almost always in the
   pre-text or the table header.
3. **Multi-step arithmetic.** 60%+ of questions require 2 or more DSL steps.
   Many questions ask for percent change ("what was the growth?"), which
   collapses into `subtract`, `divide`, `multiply(..., 100)`. Models that emit
   a single rounded float fail these.
4. **Sparse supervision on the DSL.** The program is the only explanation
   of *how* the answer was derived. That is precious signal for us - we can
   verify programs deterministically.
5. **Gold supporting facts.** Retrieval recall against `gold_inds` is
   measurable without human judgment.

### Assumptions

- The official `dev.json` / `test.json` are canonical. We download directly
  from `github.com/czyssrs/FinQA/main/dataset` so our numbers are comparable
  to every published result.
- Scale is uniform within a document. We set `doc.scale_factor` once and
  apply it per-document.
- Gold programs are reference, not ground truth of uniqueness. Two programs
  that compute the same value in different orders should both count.
- Questions that require multiple filings (cross-doc) are out of scope (they
  are not present in FinQA).

## 2. Method selection

### What standard submissions look like

A representative FinQA submission today pipes together:

- FAISS over arbitrary text chunks (sometimes the whole `pre_text`).
- OpenAI `gpt-4o` in a free-form PoT prompt that produces Python.
- A Python `exec` runner (sometimes wrapped in a sandbox).
- LangSmith tracing.

This works for the happy path and earns ~70% execution accuracy on dev. It
leaves the following money on the table:

1. **Numeric hallucinations are undetected.** The executor doesn't know the
   numbers came from the context. A `1,284,991` becomes `1,248,991`, the
   program still runs, and the answer is close but wrong.
2. **Scale mistakes.** "In millions" is a text sentence; the tokenizer drops
   it in a chunk boundary half the time.
3. **Row-selection errors.** FinQA's own error analysis (Fig 4-5 of the
   paper) shows ~30% of errors are picking the wrong row or wrong year.
4. **No GPU observability.** The API hides all inference signal - KV-cache
   hit rate, TTFT, prefill vs decode breakdown, quantization error.
5. **No calibration.** The system answers confidently on every question;
   there is no refusal signal tied to self-consistency.

### Our choices

| Component | Standard approach | Ours | Why it matters for FinQA |
|---|---|---|---|
| Retrieval | FAISS over paragraph chunks | Hybrid (FAISS + BM25 + RRF) with row-granular chunks keyed `table_i` matching FinQA `gold_inds`; reranker over top-k | Row-level retrieval lifts recall@5 on tables; `gold_inds` parity gives a real recall metric, not a proxy. |
| Scale handling | Implicit in prompt | Dedicated `ScaleExtractor` (regex + LM fallback) attaching typed `scale_factor` to every chunk | Largest single error class in FinQA; regex alone covers ~95% of tables. |
| Generation | Free-form Python (or JSON) | Grammar-constrained `AnswerEnvelope` (Pydantic) via vLLM `structured_outputs` / xgrammar. DSL = FinQA's own 10-op DSL. | Eliminates parse failures; grounds answers in the canonical DSL, not arbitrary Python. |
| Execution | `exec` | Deterministic 10-op DSL executor with `#n` back-refs, `const_*` constants, and row-label resolution for `table_*` aggregates | No RCE surface; 100% reproducible; every literal can be audited. |
| Verification | Implicit trust in LLM | Groundedness gate (every literal in the program must appear in retrieved context after normalization) + unit-sanity gate | Catches hallucinated numbers and numbers-in-wrong-scale without a supervised classifier. |
| Self-consistency | Single sample or majority-vote-over-text | K=3 samples, vote on *executed* values (not strings), tie-broken by groundedness | Self-consistency over program *results* is stricter than text voting and handles math that is the same modulo reordering. |
| Routing | Always use the big model | Qwen3-1.7B classifier routes to `rLLM-FinQA-4B` specialist (if enabled) or Qwen3-8B/14B-AWQ generalist, or refuses | Showcases cost / latency routing for a GPU-infra company; the specialist is a finance-tuned 4B model, so simple queries don't pay the 14B tax. |
| Serving | OpenAI API | `vllm serve` with AWQ / Marlin-AWQ / FP8 weights + FP8 KV cache depending on GPU; prefix-caching + chunked prefill + speculative decoding toggles exposed via config | Direct demonstration of the inference-optimization skills the role requires. |
| Observability | LangSmith traces | Prometheus metrics for drift (PSI), retrieval hit-rate, refusal rate, self-consistency calibration; vLLM native metrics for KV-cache; Grafana dashboard | Production monitoring in the actual sense, not tracing for debugging. |
| Evaluation | One exe-accuracy number | Exact FinQA-metric replica + sympy program accuracy + retrieval recall@k + groundedness + refusal rate + calibrated latency + cost-per-correct, all with bootstrap CIs | Explains *why* a change helped, not just that it did. |

### Open-source model picks (inference-only)

- Generator, T4: `Qwen/Qwen3-8B-AWQ`. AWQ fits in 16 GB with 4k context and
  supports vLLM `structured_outputs` via xgrammar.
- Generator, L4: `Qwen/Qwen3-14B-AWQ` with `awq_marlin` kernel and `fp8_e4m3`
  KV cache.
- Specialist (optional): `rLLM/rLLM-FinQA-4B`. Finance-tuned, tiny, and
  handles simple lookups and single-ratio questions with high accuracy.
- Router: `Qwen/Qwen3-1.7B` (transformers backend - the router is cheap enough
  that colocating it with vLLM is wasteful).
- Embedding: `Qwen/Qwen3-Embedding-0.6B` with instruction-aware prompts.
  Fallback to `BAAI/bge-small-en-v1.5` on failure.
- Reranker: `Qwen/Qwen3-Reranker-0.6B` (cross-encoder). Fallback to
  `BAAI/bge-reranker-v2-m3` then `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Fine-tuning is explicitly out of scope. The goal is inference-time
optimization.

## 3. Evaluation strategy

### Metrics

- **Execution accuracy** - exact replica of the official `evaluate.py`:
  round-to-5-decimals equality with the percent-divide-by-100 convention.
  We keep our own semantic number parser for everything else to avoid
  importing the official script's quirks into production.
- **Program accuracy** - sympy-based symbolic equivalence. Two FinQA programs
  match iff `simplify(pred_expr - gold_expr) == 0`, handling commutativity
  and reordering for free. This is forgiving of legitimate variation in
  step order but strict about correctness.
- **Retrieval recall@k** and **full-recall@k** against `gold_inds`.
- **Groundedness rate** - fraction of programs whose every literal is present
  in retrieved context after normalization.
- **Refusal rate** - fraction of questions where the graph hits the refuse
  edge (ungrounded literal, no majority-agreement, or out-of-scope router).
- **Unit correctness** - fraction of answers whose `answer_form` / `scale`
  are consistent with the document scale.
- **Latency p50/p90/p95/p99** end-to-end; vLLM `/metrics` exposes TTFT, ITL,
  throughput.
- **Cost per correct answer** - for vLLM we use hourly GPU rates; for the
  OpenAI baseline we use published per-token prices.
- All rates are reported with a 95% bootstrap CI (1,000 resamples).

### Slices and ablations

Declared in `configs/eval.yaml`:

- `dev_200` - 200 samples, shuffled with seed 13.
- `dev_multi_step` - samples where gold program has 2+ steps.
- `adversarial_row_deleted_20` - 20 dev samples with the gold table row
  deleted from the chunks; measures calibrated refusal.

Ablation suite (produced by `make ablate`):

- **D1 typed_vs_free_form** - disable typed output; regenerate free-form DSL.
- **D2 scale_on/off** - disable `ScaleExtractor`.
- **D3 router_on/off** - always use generalist.
- **D4 K=1 + groundedness off** - single-sample, no groundedness gate.
- **D5 prefix_cache_on/off** - turn off vLLM prefix caching.

Each ablation runs the full dev_200 slice and writes one row to
`docs/ABLATIONS.md`.

### How we measure numerical-reasoning quality beyond exe-accuracy

- **Program accuracy** catches the case where the answer is right by luck
  but the reasoning is wrong.
- **Groundedness rate** catches hallucinated numbers (the dominant error
  mode per FinQA's own analysis).
- **Retrieval recall@k** separates retrieval failure from reasoning failure.
- **Calibration bins** (agreement vs correctness) surface whether the
  refusal signal is actually usable in production.
- **Adversarial perturbations** (delete the gold row) measure whether the
  system knows when it does not know.

## 4. Production monitoring plan

### Four drift signals

1. **Input drift (PSI)** on the embedding of each incoming question, binned
   into 10 equal-width buckets against a reference distribution from
   training. Alert at PSI > 0.2.
2. **Retrieval drift** - rolling hit-rate@k on a frozen golden set of 50
   questions that we rerun every hour. Alert on a 10% absolute drop.
3. **Output drift** - distribution summaries over the last window:
   mean program length, operator mix, refusal rate, groundedness rate.
   Alert on a 2-sigma change.
4. **Calibration drift** - bin self-consistency agreement vs downstream
   correctness (when feedback is available). A flat curve means the
   confidence signal has degenerated.

### What we scrape

Two `/metrics` endpoints:

- vLLM's native `/metrics` - KV-cache usage, prefix-cache hit rate,
  prompt/generation tokens, request latency histograms.
- Our `/metrics` (FastAPI) - `finqa_questions_total`, `finqa_refusals_total`,
  `finqa_ungrounded_programs_total`, `finqa_executions_total`, routing
  counters, latency histogram, consistency-agreement histogram, PSI gauge,
  retrieval-recall gauge.

A Grafana dashboard (`ops/grafana/finqa_dashboard.json`) stitches these
together so on-call sees KV-cache pressure, refusal rate, and input drift on
one pane.

### Maintenance / improvement plan

- **Weekly**: rerun the frozen 50-question golden set; track recall@k.
- **Nightly**: rerun dev_200 against the production bundle; track
  execution accuracy and cost-per-correct.
- **Monthly**: re-extract scale-factor regex coverage from new filings; if
  regex hit-rate drops below 90%, retrain the tiny scale classifier.
- **Quarterly**: refresh embeddings + reranker on the latest Qwen family;
  re-run the full ablation matrix to make sure no component has regressed.

Details in `docs/PRODUCTION.md`.
