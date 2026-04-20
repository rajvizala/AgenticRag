# Production plan

This document covers what happens past the `setup.sh` Colab demo: how the
same chatbot runs in a multi-tenant, multi-GPU, SLA-driven environment.

## Target SLAs

- p95 end-to-end latency: 3.5 s for single-doc FinQA queries.
- 99.0% availability, single region.
- <2% refusal rate on in-distribution questions; refusals must carry a
  machine-readable reason.
- <1% groundedness violations in production logs.

## Serving topology

### vLLM pool

- Deploy vLLM with `--tensor-parallel-size 1` on L4 (24 GB) or A10G
  (24 GB) nodes; Qwen3-14B-AWQ fits in a single replica.
- Use `--enable-prefix-caching --enable-chunked-prefill`.
- Use `awq_marlin` kernel on Ada+ GPUs; fall back to `awq` on Turing.
- Use `--kv-cache-dtype fp8_e4m3` on Ada+; doubles effective KV capacity.
- `--max-num-seqs 32 --max-num-batched-tokens 8192` as a starting point;
  tune via the GPU benchmark (`docs/GPU_BENCHMARK.md`).
- Minimum 2 replicas behind a Gateway; HPA on KV-cache usage (`vllm:kv_cache_usage`)
  rather than CPU.

### Prefix-cache sharing (LMCache)

Our graph's system prompt + structured-output grammar is identical across
requests. That is the exact scenario LMCache was built for.

- Attach LMCache as a shared remote KV store for all vLLM replicas.
- Expected hit rate on the prompt prefix: >95% (the prompt is literally
  constant).
- Expected TTFT reduction: 30-50% on cache hits, per LMCache benchmarks.

### Router / embedding / reranker pool

- Qwen3-1.7B router, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B are small
  enough to run on CPU or a single T4.
- Deploy as a separate `infer-tier` (transformers backend) so generator
  autoscaling does not steal their quota.
- Prefer a single sidecar process per pod to keep latency deterministic.

### API tier

- FastAPI app runs on CPU nodes. 2 replicas minimum.
- Exposes `/ask`, `/chat` (SSE), `/health`, `/metrics`.
- Implements a per-tenant token bucket and a concurrency gate via
  `anyio.Semaphore` to avoid overloading vLLM during spikes.

### Gradio

- Gradio is a *demo* front-end. In production the UI is a separate Next.js
  app that consumes `/chat`.

## Kubernetes manifest outline

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: {name: vllm-finqa}
spec:
  replicas: 2
  strategy: {type: RollingUpdate, rollingUpdate: {maxSurge: 1, maxUnavailable: 0}}
  template:
    spec:
      containers:
      - name: vllm
        image: ghcr.io/vllm-project/vllm-openai:latest
        resources:
          limits: {nvidia.com/gpu: "1"}
        args:
        - "--model"
        - "Qwen/Qwen3-14B-AWQ"
        - "--quantization"
        - "awq_marlin"
        - "--kv-cache-dtype"
        - "fp8_e4m3"
        - "--enable-prefix-caching"
        - "--enable-chunked-prefill"
        - "--max-num-seqs"
        - "32"
        - "--max-num-batched-tokens"
        - "8192"
        env:
        - name: VLLM_USE_V1
          value: "1"
        readinessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 120
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: vllm-finqa}
spec:
  scaleTargetRef: {kind: Deployment, name: vllm-finqa}
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric: {name: vllm_kv_cache_usage}
      target: {type: AverageValue, averageValue: "0.7"}
```

A matching `Deployment` runs the FastAPI app on CPU nodes; a `ServiceMonitor`
scrapes both pools into Prometheus.

## Rollout strategy

- **Canary**: 5% traffic to new generator weights or new prompt templates.
  Gate promotion on:
  - Execution accuracy on dev_200 >= prod - 0.5%.
  - Groundedness rate >= prod - 1%.
  - p95 latency within +10% of prod.
  - Refusal rate within +/- 0.5% of prod.
- **Full rollout**: only after 1 hour clean canary.
- **Auto-rollback**: triggered by any of the gates above breached for 5
  consecutive minutes.

## Observability

- **Prometheus** scrapes vLLM `/metrics` and FastAPI `/metrics` per
  `ops/prometheus/prometheus.yml`.
- **Grafana** dashboard `ops/grafana/finqa_dashboard.json` surfaces QPS,
  latency, KV-cache pressure, refusal rate, retrieval recall, PSI input
  drift, and the calibration curve.
- **Structured logs** (JSON) for every graph run include `thread_id`,
  `route`, `scale_factor`, `retrieval.recall`, `program`, `answer`,
  `verification.reasons`. Shipped to OpenSearch / Loki.

## Cost controls

- Router sends `lookup` questions to the 4B specialist when available; that
  alone saves ~2-3x on simple queries.
- `awq_marlin` + FP8 KV cache roughly doubles throughput per dollar on L4.
- Prefix caching + LMCache cuts prefill cost on repeated system prompts.
- `max_num_seqs` is tuned from the benchmark to the point where p95 latency
  just meets SLA; beyond that, add replicas instead of more sequences.

## Security

- vLLM is never exposed publicly. Only the FastAPI app has an ingress.
- DSL executor is deterministic and sandbox-free; it is *not* Python `exec`.
  This is a deliberate attack-surface reduction.
- Structured output removes the "LLM emits shell metacharacters" class of
  bugs by construction.
- Per-tenant tokens are enforced at the API gateway; rate limit at the
  FastAPI tier is defense-in-depth.

## What to build next

Ordered by expected impact:

1. **LMCache integration** - highest ROI; single-session prefix hits are
   already common in eval, fleet-wide hits should be >95%.
2. **Online learning loop on the router** - feed resolved vs refused into
   a weekly retraining of the 1.7B classifier.
3. **Retrieval ANN index on FAISS-HNSW** when index grows past 1M chunks;
   swap from flat IP.
4. **Speculative decoding** with Qwen3-1.7B as draft model on the format
   step where outputs are short and predictable (>1.5x TTFT win in offline
   tests).
5. **Multi-doc retrieval** - lift the current single-doc constraint using
   a document classifier on top of the embedder.
6. **Auto-tuned quantization** - nightly sweep of AWQ vs FP8 vs GPTQ on the
   frozen golden set; pick per-model.
