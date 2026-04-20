# FinQA ablation matrix

This file is **regenerated** by `make ablate` (or `finqa-bot ablate`). The
run-generated version overwrites everything below the next section heading.

## How to regenerate

```bash
# 1. Boot vLLM (setup.sh does this automatically)
bash setup.sh --no-ui

# 2. Run the ablation matrix on dev_200
make ablate            # writes docs/ABLATIONS.md

# Or pick a slice / size explicitly:
finqa-bot ablate --slice dev_200 --n 200
finqa-bot ablate --slice dev_full --n 883        # full dev set
finqa-bot ablate --slice adversarial_row_deleted_20 --n 20
```

## What each ablation toggles

The spec list lives in `configs/eval.yaml` under `ablations`. Each spec
overrides exactly one component so the delta vs the `default` row is causal:

- **D1_typed_vs_free_form** - disables the `AnswerEnvelope` grammar
  constraint; generate emits free-form DSL text that the parser may reject.
- **D2_scale_off** - disables `ScaleExtractor`; generation has to infer
  scale from prose.
- **D3_no_router_always_generalist** - bypasses the Qwen3-1.7B router;
  every question hits the 8B/14B generalist.
- **D4_sc_k1** - disables self-consistency (K=1 instead of K=3).
- **D4_groundedness_off** - bypasses the groundedness verifier; any
  generated program is accepted.
- **D5_prefix_cache_off** - restarts vLLM without `--enable-prefix-caching`
  and reruns; tests the latency / throughput contribution.
- **baseline_openai** - swaps the local LLM for OpenAI `gpt-4o-mini` but
  keeps retrieval / execution / verification identical. Primary use: cost
  per correct answer against an external reference.

## Metric definitions (one line each)

- **ExeAcc** - exact replica of the official FinQA `evaluate.py`:
  round-to-5 equality with the percent-by-100 convention.
- **95% CI** - bootstrap CI on ExeAcc, 1,000 resamples with seed 13.
- **ProgAcc** - sympy-based program equivalence (see
  `src/finqa_bot/eval/program_metric.py`).
- **Recall@5** - fraction of gold supporting indices recovered in top-5
  retrieved chunks; keyed by FinQA `text_i` / `table_i`.
- **Grounded** - fraction of programs whose every literal appears in the
  retrieved context after normalization.
- **Refused** - fraction of questions where the graph took the refuse edge.
- **p50 / p95 (ms)** - end-to-end request latency.
- **$/correct** - GPU-hours at the GPU's on-demand price divided by the
  number of correct answers; for the `baseline_openai` row this uses
  OpenAI's published per-token pricing.

## Latest run

_This section is overwritten every time `make ablate` runs._

```
(no results yet - run `make ablate` to populate)
```

| Ablation | ExeAcc | 95% CI | ProgAcc | Recall@5 | Grounded | Refused | p50 (ms) | p95 (ms) | $/correct |
|---|---|---|---|---|---|---|---|---|---|
| default | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D1_typed_vs_free_form | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D2_scale_off | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D3_no_router_always_generalist | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D4_sc_k1 | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D4_groundedness_off | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D5_prefix_cache_off | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| baseline_openai | TBD | [TBD, TBD] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## How to read this table

For a well-behaved pipeline you should observe:

- `D1_typed_vs_free_form`: ExeAcc drops 2-5 points, refusal rate rises
  (parse failures).
- `D2_scale_off`: ExeAcc drops the most on table-heavy slices
  (`const_scaled`).
- `D3_no_router_always_generalist`: ExeAcc roughly unchanged; latency rises
  and cost per correct rises (specialist was saving money).
- `D4_sc_k1`: ExeAcc drops 1-2 points; refusal rate drops (no disagreement
  to refuse on).
- `D4_groundedness_off`: ExeAcc may move slightly, but the calibration
  curve (agreement vs correctness) flattens sharply - that is the signal
  we care about.
- `D5_prefix_cache_off`: ExeAcc unchanged; p50 / p95 rise 20-40%.
- `baseline_openai`: comparable ExeAcc, 3-10x higher cost per correct on
  dev_200.

If any of these differ markedly, that is interesting and usually points to
a bug or a slice-specific effect worth a sentence in the report.
