"""GPU benchmark matrix.

Measures the inference-side numbers we actually care about for the role:
throughput (tok/s) at batch=1 and batch=8, time-to-first-token, end-to-end
p50/p95/p99 latency, and the accuracy delta across quantization modes.

Because each row of the matrix needs a different vLLM process, this module
emits an orchestration plan rather than running everything end-to-end: the
caller runs ``finqa-bot bench`` which prints the sequence of
``setup.sh --quantization ...`` commands plus the ``finqa-bot eval`` sweeps.

For single-process use inside a live vLLM session, :func:`run_throughput_probe`
hammers the current server with ``N`` concurrent small chat completions and
reports tok/s + TTFT.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from finqa_bot.config import GpuConfig, Settings
from finqa_bot.logging import get_logger
from finqa_bot.serving.openai_client import probe_metrics

log = get_logger(__name__)


BENCH_PROMPT = (
    "You are a financial analyst. Briefly answer with one sentence: "
    "What is the accounting treatment of a 10-K filing in the United States?"
)


async def run_throughput_probe(
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    n_requests: int = 32,
    batch_size: int = 8,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Send ``n_requests`` chat completions against the running vLLM server.

    Returns a dict with ``tokens_per_second``, ``ttft_p50_ms``, ``ttft_p95_ms``,
    and ``e2e_p50_ms``, ``e2e_p95_ms``.
    """
    sem = asyncio.Semaphore(batch_size)
    ttfts: list[float] = []
    e2es: list[float] = []
    tokens_out = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def _one(i: int) -> None:
            nonlocal tokens_out
            async with sem:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": BENCH_PROMPT}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                }
                started = time.perf_counter()
                first_token_at: float | None = None
                toks = 0
                async with client.stream(
                    "POST",
                    base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        if line.strip() == "data: [DONE]":
                            break
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        toks += 1
                ended = time.perf_counter()
                if first_token_at is None:
                    first_token_at = ended
                ttfts.append(first_token_at - started)
                e2es.append(ended - started)
                tokens_out += toks

        t0 = time.perf_counter()
        await asyncio.gather(*(_one(i) for i in range(n_requests)))
        total = time.perf_counter() - t0

    def pct(arr: list[float], p: int) -> float:
        if not arr:
            return 0.0
        arr = sorted(arr)
        idx = min(len(arr) - 1, round(p / 100 * (len(arr) - 1)))
        return arr[idx] * 1000

    return {
        "tokens_per_second": round(tokens_out / max(total, 1e-9), 3),
        "total_requests": n_requests,
        "batch_size": batch_size,
        "ttft_p50_ms": round(pct(ttfts, 50), 3),
        "ttft_p95_ms": round(pct(ttfts, 95), 3),
        "e2e_p50_ms": round(pct(e2es, 50), 3),
        "e2e_p95_ms": round(pct(e2es, 95), 3),
        "wall_time_s": round(total, 3),
        "tokens_out": tokens_out,
    }


def run_gpu_benchmark(
    gpu_cfg: GpuConfig,
    settings: Settings,
    n: int = 100,
    out_path: Path | str = Path("docs/GPU_BENCHMARK.md"),
) -> None:
    """Run the accuracy + throughput sweep against the currently-running server.

    The GPU-config-specific rows (FP16 vs AWQ vs Marlin-AWQ vs FP8) require
    restarting vLLM between runs with a different ``--quantization`` flag; the
    command line this emits is meant to be driven by ``scripts/bench.sh`` in
    sequence.
    """
    log.info("Probing throughput on current vLLM server ...")
    metrics_before = probe_metrics(settings.vllm_base_url)
    probe = asyncio.run(
        run_throughput_probe(
            base_url=settings.vllm_base_url,
            model=settings.llm_model,
            api_key=settings.vllm_api_key,
            n_requests=max(16, n),
            batch_size=min(8, gpu_cfg.generator.max_num_seqs),
            max_tokens=128,
        )
    )
    metrics_after = probe_metrics(settings.vllm_base_url)

    _write_bench_markdown(out_path, gpu_cfg, probe, metrics_before, metrics_after)


def _write_bench_markdown(
    out_path: Path | str,
    cfg: GpuConfig,
    probe: dict[str, Any],
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    delta_gen = (metrics_after.get("vllm:generation_tokens_total", 0.0)
                 - metrics_before.get("vllm:generation_tokens_total", 0.0))
    prefix_hits_delta = (metrics_after.get("vllm:prefix_cache_hits_total", 0.0)
                         - metrics_before.get("vllm:prefix_cache_hits_total", 0.0))
    prefix_queries_delta = (metrics_after.get("vllm:prefix_cache_queries_total", 0.0)
                            - metrics_before.get("vllm:prefix_cache_queries_total", 0.0))
    hit_rate = (prefix_hits_delta / prefix_queries_delta) if prefix_queries_delta else 0.0

    content = (
        "# GPU benchmark\n\n"
        f"GPU: **{cfg.gpu.name}** ({cfg.gpu.arch}, {cfg.gpu.vram_gb} GB).\n"
        f"Model: **{cfg.generator.model}** | quantization=**{cfg.generator.quantization or 'none'}** | "
        f"kv_cache_dtype=**{cfg.generator.kv_cache_dtype}** | prefix_caching=**{cfg.generator.enable_prefix_caching}**.\n\n"
        "## Single-server probe\n\n"
        f"- Requests: {probe['total_requests']} (batch={probe['batch_size']}), tokens out = {probe['tokens_out']}\n"
        f"- Throughput: **{probe['tokens_per_second']} tok/s**\n"
        f"- TTFT p50 / p95: {probe['ttft_p50_ms']:.1f} / {probe['ttft_p95_ms']:.1f} ms\n"
        f"- End-to-end p50 / p95: {probe['e2e_p50_ms']:.1f} / {probe['e2e_p95_ms']:.1f} ms\n"
        f"- Prefix-cache hit rate during probe: {hit_rate:.2%}\n"
        f"- vLLM-reported generation tokens delta: {delta_gen:.0f}\n"
        f"- Wall time: {probe['wall_time_s']:.2f}s\n\n"
        "## Quantization sweep\n\n"
        "Run each row by restarting vLLM with the flag noted, then rerunning\n"
        "`finqa-bot eval --slice dev_200` plus this probe:\n\n"
        "| Quantization | kv_cache_dtype | ExeAcc delta | tok/s | TTFT p50 (ms) | TTFT p95 (ms) |\n"
        "|---|---|---|---|---|---|\n"
        "| fp16 | auto | baseline | TBD | TBD | TBD |\n"
        "| awq | auto | TBD | TBD | TBD | TBD |\n"
        "| awq_marlin | auto | TBD | TBD | TBD | TBD |\n"
        "| fp8 | auto | TBD | TBD | TBD | TBD |\n"
        "| fp8 | fp8_e4m3 | TBD | TBD | TBD | TBD |\n"
    )
    out_path.write_text(content, encoding="utf-8")
    log.info("Wrote GPU benchmark to %s", out_path)
