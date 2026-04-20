# GPU benchmark

This file is **regenerated** by `make bench` (or `finqa-bot bench`). The
run-generated content overwrites everything below the next section heading.

## How to regenerate

```bash
# 1. Boot vLLM (setup.sh does this automatically)
bash setup.sh --no-ui

# 2. Run the probe against the live server
make bench            # writes docs/GPU_BENCHMARK.md

# Or explicitly:
finqa-bot bench --n 100
```

For the full quantization sweep you need to restart vLLM between rows
(different `--quantization` flag). `scripts/run_vllm.sh` is auto-generated
from `configs/t4.yaml` / `configs/l4.yaml`; change
`generator.quantization` + `generator.kv_cache_dtype` and re-run
`setup.sh --no-ui` between rows.

## What we measure

For each `(quantization, kv_cache_dtype, batch_size)` cell:

- **tok/s** - throughput, decode tokens per wall-clock second.
- **TTFT p50 / p95 (ms)** - time-to-first-token; proxy for prefill cost.
- **ITL p50 / p95 (ms)** - inter-token latency; proxy for decode cost.
- **E2E p50 / p95 (ms)** - end-to-end request latency (TTFT + decode).
- **KV usage** - `vllm:kv_cache_usage_perc` sampled during the probe.
- **Prefix-cache hit rate** - delta in `vllm:prefix_cache_hits_total`
  over the probe window.
- **ExeAcc delta** - FinQA dev_200 execution-accuracy delta vs the FP16
  baseline; this is the accuracy price of quantization.

## What "good" looks like on T4 (16 GB, Turing)

- FP16 on a 14B model does not fit. Start with `Qwen3-8B-AWQ`.
- AWQ baseline: ~80-100 tok/s at batch=1, ~250-400 tok/s at batch=8.
- TTFT p50 at 1-2k context: 500-900 ms.
- Marlin and FP8 kernels are **not available** on Turing. The runner will
  refuse to launch with `quantization=awq_marlin` or `kv_cache_dtype=fp8*`.
- Prefix caching should push TTFT p50 below 200 ms on repeated prompts.

## What "good" looks like on L4 (24 GB, Ada)

- Qwen3-14B-AWQ with `awq_marlin` + `fp8_e4m3` KV cache fits comfortably.
- AWQ -> Marlin-AWQ: ~1.4-1.8x throughput on decode.
- FP8 KV cache: ~2x effective KV capacity, letting `max_num_seqs` grow to
  ~64 without OOM.
- TTFT p50 at 1-2k context: 250-450 ms.
- Prefix caching + LMCache should keep TTFT p50 below 100 ms steady-state.

## Latest run

_This section is overwritten every time `make bench` runs._

```
(no results yet - run `make bench` to populate)
```

| Quantization | kv_cache_dtype | batch | tok/s | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | ExeAcc delta |
|---|---|---|---|---|---|---|---|---|
| fp16 | auto | 1 | baseline | baseline | baseline | baseline | baseline | 0.000 |
| fp16 | auto | 8 | TBD | TBD | TBD | TBD | TBD | 0.000 |
| awq | auto | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| awq | auto | 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| awq_marlin | auto | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| awq_marlin | auto | 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| fp8 | auto | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| fp8 | fp8_e4m3 | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| fp8 | fp8_e4m3 | 8 | TBD | TBD | TBD | TBD | TBD | TBD |

## Speculative decoding

Qwen3-1.7B is a strong draft model for Qwen3-14B on the short, structured
`format` node (the grammar-constrained JSON emission). Expected speedup:
1.3-1.6x TTFT on that node without ExeAcc regression, because the draft
model only proposes tokens inside the grammar.

To enable, add `--speculative-model Qwen/Qwen3-1.7B --num-speculative-tokens 5`
to `configs/l4.yaml` `generator.extra_args` and rerun `make bench`.

## How to read this file

We care about three things in order:

1. **Accuracy is preserved.** ExeAcc delta should be within one point of
   FP16 for any row we consider shipping. If it drops more, the quant
   config is broken.
2. **Throughput is up per dollar.** The benchmark is not about peak tok/s;
   it is about tok/s at the latency budget (p95 E2E <= 3.5 s).
3. **KV cache has headroom.** `vllm:kv_cache_usage_perc` at steady state
   should be < 0.7; otherwise a traffic spike evicts prefix caches and
   TTFT blows up.
