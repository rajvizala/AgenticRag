"""Async evaluation harness.

Runs a FinQA slice through the graph and produces per-sample records plus an
aggregate :class:`~finqa_bot.types.EvalSummary`.

Public entry point: :func:`run_eval`. It:

1. Loads the slice (named or by split+n).
2. Builds the graph with the requested feature flags.
3. Fans out graph invocations via ``asyncio.gather`` under a semaphore.
4. Computes exe-accuracy, program-accuracy, retrieval recall, unit correctness,
   groundedness rate, refusal rate, latency percentiles, and cost-per-correct.
5. Emits JSON + a short Markdown footer to ``runs/<run>/`` so ablations and
   the docs pipeline can pull the numbers.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from finqa_bot.config import EvalConfig, GpuConfig, Settings, apply_overrides, pipeline_flags
from finqa_bot.eval.finqa_metric import exe_equal, official_str_to_num, percent_or_decimal_equal
from finqa_bot.eval.program_metric import program_match_symbolic
from finqa_bot.eval.slices import load_slice
from finqa_bot.execution.dsl import dump_program
from finqa_bot.graph.graph import BuildOptions, GraphRunner, build_graph
from finqa_bot.graph.state import initial_state
from finqa_bot.logging import get_logger
from finqa_bot.retrieval.indexer import CorpusIndex, build_index
from finqa_bot.types import EvalRecord, EvalSample, EvalSummary, GraphState
from finqa_bot.verification.units import check_units

log = get_logger(__name__)


# Published USD cost per 1M tokens for the OpenAI baseline (as of 2025-06).
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.500, 10.000),
    "gpt-3.5-turbo": (0.500, 1.500),
}

# Hourly GPU cost estimates for cost-per-correct reporting (Colab / on-demand).
GPU_HOURLY_USD: dict[str, float] = {
    "t4": 0.35,
    "l4": 0.70,
    "a100": 1.50,
    "h100": 2.80,
}


async def run_eval(
    gpu_cfg: GpuConfig,
    eval_cfg: EvalConfig,
    settings: Settings,
    slice_name: str | None = None,
    split: str = "dev",
    n: int | None = 200,
    concurrency: int | None = None,
    ablation_id: str | None = None,
    out_path: Path | str = Path("runs/eval_latest.json"),
    progress_fn: Callable[["EvalRecord", int, int], None] | None = None,
) -> "EvalSummary":
    """Run evaluation and return an :class:`EvalSummary`.

    *progress_fn*, if provided, is called after every sample with
    ``(record, completed_count, total_count)``.  Because asyncio is
    single-threaded the callback runs synchronously in the event loop —
    keep it fast (a plain ``print`` / ``typer.echo`` is fine).
    """
    active_cfg = gpu_cfg
    feature_flags: dict[str, bool] = {}
    if ablation_id:
        ablation = next((a for a in eval_cfg.ablations if a.id == ablation_id), None)
        if ablation is None:
            raise KeyError(f"Unknown ablation id: {ablation_id}")
        feature_flags = pipeline_flags(ablation.overrides)
        try:
            active_cfg = apply_overrides(gpu_cfg, ablation.overrides)
        except KeyError as exc:
            log.warning("Ignoring override for ablation %s: %s", ablation_id, exc)

    samples = _resolve_samples(eval_cfg, settings, slice_name, split, n)
    if not samples:
        raise RuntimeError("No samples to evaluate; check slice config.")

    index = _maybe_build_index(settings, active_cfg, split)
    runner = build_graph(
        gpu_cfg=active_cfg,
        settings=settings,
        options=BuildOptions(index=index, feature_flags=feature_flags),
    )

    conc = concurrency or eval_cfg.batching.eval_concurrency
    semaphore = asyncio.Semaphore(max(1, conc))
    t0 = time.perf_counter()
    total = len(samples)
    # Mutable counter — asyncio is single-threaded so no lock needed.
    _completed: list[int] = [0]

    async def _one(sample: EvalSample) -> EvalRecord:
        async with semaphore:
            record = await _evaluate_sample(sample, runner)
        _completed[0] += 1
        if progress_fn is not None:
            progress_fn(record, _completed[0], total)
        return record

    records: list[EvalRecord] = await asyncio.gather(*(_one(s) for s in samples))
    wall = time.perf_counter() - t0

    summary = _summarize(
        records=records,
        slice_name=slice_name or f"{split}_n{len(samples)}",
        wall_time_s=wall,
        eval_cfg=eval_cfg,
        ablation_id=ablation_id or "",
        settings=settings,
        gpu_cfg=active_cfg,
    )
    _persist(out_path, records, summary, ablation_id)
    return summary


def _resolve_samples(
    eval_cfg: EvalConfig,
    settings: Settings,
    slice_name: str | None,
    split: str,
    n: int | None,
) -> list[EvalSample]:
    data_dir = settings.finqa_data_dir
    if slice_name:
        return load_slice(eval_cfg, slice_name, data_dir=data_dir, n_override=n)
    from finqa_bot.data.sample import load_samples

    split_path = data_dir / "raw" / "finqa" / f"{split}.json"
    return load_samples(split_path, n=n, seed=13)


def _maybe_build_index(settings: Settings, cfg: GpuConfig, split: str) -> CorpusIndex | None:
    try:
        return build_index(settings.finqa_data_dir, cfg, split=split, rebuild=False)
    except Exception as exc:
        log.warning("Index unavailable (%s); running without retriever.", exc)
        return None


async def _evaluate_sample(sample: EvalSample, runner: GraphRunner) -> EvalRecord:
    state: GraphState = initial_state(sample.question, doc_id=sample.id)
    state["document"] = sample.document
    started = time.perf_counter()
    err: str | None = None
    try:
        final: GraphState = await runner.ainvoke(state)
    except Exception as exc:
        log.exception("Graph invocation failed for %s", sample.id)
        err = str(exc)
        final = state
    elapsed = time.perf_counter() - started

    envelope = final.get("envelope")
    execution = final.get("execution")
    ground = final.get("groundedness")
    predicted_value: Any = None
    predicted_program = ""
    if envelope is not None:
        predicted_program = dump_program(envelope.program)
    if execution is not None and execution.ok:
        predicted_value = execution.value
    elif envelope is not None:
        predicted_value = envelope.answer_value

    gold_answer = sample.gold_answer
    if isinstance(gold_answer, str):
        gold_answer = official_str_to_num(gold_answer)
    exe_correct = False
    if predicted_value is not None and gold_answer is not None and gold_answer != "n/a":
        exe_correct = exe_equal(predicted_value, gold_answer) or percent_or_decimal_equal(
            predicted_value, gold_answer
        )

    prog_correct: bool | None = None
    if sample.gold_program and predicted_program:
        try:
            prog_correct = program_match_symbolic(predicted_program, sample.gold_program)
        except Exception:
            prog_correct = False

    hits = final.get("hits") or []
    retrieved_ids = [h.chunk.id for h in hits]
    gold_inds = sample.gold_inds
    hit_set = set(retrieved_ids)
    gold_set = set(gold_inds)
    recall = (len(hit_set & gold_set) / len(gold_set)) if gold_set else 0.0
    full = gold_set.issubset(hit_set) and bool(gold_set)

    unit_report = check_units(envelope, sample.document) if envelope is not None else None

    return EvalRecord(
        sample_id=sample.id,
        question=sample.question,
        predicted_answer=predicted_value,
        gold_answer=gold_answer,
        execution_correct=bool(exe_correct),
        program_correct=prog_correct,
        predicted_program=predicted_program,
        gold_program=sample.gold_program,
        retrieval_recall=recall,
        retrieval_full_recall=full,
        retrieved_ids=retrieved_ids,
        gold_inds=gold_inds,
        grounded=bool(ground.ok) if ground else False,
        refused=bool(final.get("refused", False)),
        unit_correct=bool(unit_report.ok) if unit_report else True,
        latency_s=round(elapsed, 4),
        error=err,
    )


def _summarize(
    records: list[EvalRecord],
    slice_name: str,
    wall_time_s: float,
    eval_cfg: EvalConfig,
    ablation_id: str,
    settings: Settings,
    gpu_cfg: GpuConfig,
) -> EvalSummary:
    n = len(records)
    if not n:
        return EvalSummary(
            slice=slice_name,
            n=0,
            execution_accuracy=0.0,
            program_accuracy=0.0,
            unit_correctness=0.0,
            groundedness_rate=0.0,
            refusal_rate=0.0,
        )
    exe = [r.execution_correct for r in records]
    prog = [r.program_correct for r in records if r.program_correct is not None]
    units = [r.unit_correct for r in records]
    ground = [r.grounded for r in records]
    refused = [r.refused for r in records]
    latencies = [r.latency_s for r in records]
    latencies_sorted = sorted(latencies)

    def pct(arr: list[float], p: int) -> float:
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, round(p / 100 * (len(arr) - 1)))
        return arr[idx]

    latency_pcts = {f"p{p}": round(pct(latencies_sorted, p), 4) for p in eval_cfg.metrics.latency_percentiles}
    latency_pcts["mean"] = round(mean(latencies), 4)

    recall_at_k = _retrieval_recall_at_k(records, eval_cfg.metrics.retrieval_recall_at_k)
    full_recall_at_k = _retrieval_full_recall_at_k(records, eval_cfg.metrics.full_retrieval_recall_at_k)

    ci_low, ci_high = _bootstrap_ci(
        [1.0 if r.execution_correct else 0.0 for r in records],
        confidence=eval_cfg.bootstrap.confidence_level,
        resamples=eval_cfg.bootstrap.resamples,
        seed=eval_cfg.bootstrap.seed,
    )

    cost_per_correct = _estimate_cost_per_correct(records, wall_time_s, settings, gpu_cfg)
    cost_per_question = _estimate_cost_per_question(records, wall_time_s, settings, gpu_cfg)

    return EvalSummary(
        slice=slice_name,
        n=n,
        execution_accuracy=sum(exe) / n,
        program_accuracy=(sum(prog) / len(prog)) if prog else 0.0,
        unit_correctness=sum(units) / n,
        groundedness_rate=sum(ground) / n,
        refusal_rate=sum(refused) / n,
        retrieval_recall=recall_at_k,
        retrieval_full_recall=full_recall_at_k,
        latency_ms={k: v * 1000.0 for k, v in latency_pcts.items()},
        tokens={},
        cost_per_correct_usd=cost_per_correct,
        cost_per_question_usd=cost_per_question,
        wall_time_s=wall_time_s,
        config_id=f"{gpu_cfg.gpu.name}:{gpu_cfg.generator.model}",
        ablation_id=ablation_id,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _retrieval_recall_at_k(records: list[EvalRecord], ks: list[int]) -> dict[int, float]:
    # We only store the final top rerank_k chunks; recall@k here is computed by
    # truncating retrieved_ids. For smaller k, we report the same (or lower) recall.
    out: dict[int, float] = {}
    for k in ks:
        vals = []
        for r in records:
            if not r.gold_inds:
                continue
            topk = set(r.retrieved_ids[:k])
            vals.append(len(topk & set(r.gold_inds)) / len(set(r.gold_inds)))
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def _retrieval_full_recall_at_k(records: list[EvalRecord], ks: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in ks:
        vals = []
        for r in records:
            if not r.gold_inds:
                continue
            topk = set(r.retrieved_ids[:k])
            vals.append(1.0 if set(r.gold_inds).issubset(topk) else 0.0)
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def _bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 13,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    import random

    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(resamples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    alpha = (1 - confidence) / 2
    lo = samples[int(alpha * resamples)]
    hi = samples[int((1 - alpha) * resamples) - 1]
    return (round(lo, 6), round(hi, 6))


def _estimate_cost_per_correct(
    records: list[EvalRecord],
    wall_time_s: float,
    settings: Settings,
    gpu_cfg: GpuConfig,
) -> float:
    correct = sum(1 for r in records if r.execution_correct)
    if correct == 0:
        return 0.0
    per_q = _estimate_cost_per_question(records, wall_time_s, settings, gpu_cfg)
    total = per_q * len(records)
    return round(total / correct, 6)


def _estimate_cost_per_question(
    records: list[EvalRecord],
    wall_time_s: float,
    settings: Settings,
    gpu_cfg: GpuConfig,
) -> float:
    if not records:
        return 0.0
    model = (settings.llm_model or "").lower()
    if model in OPENAI_PRICING and any(r.prompt_tokens + r.completion_tokens for r in records):
        in_px, out_px = OPENAI_PRICING[model]
        in_tok = sum(r.prompt_tokens for r in records)
        out_tok = sum(r.completion_tokens for r in records)
        return round((in_tok * in_px + out_tok * out_px) / 1_000_000 / max(1, len(records)), 6)
    hourly = GPU_HOURLY_USD.get(gpu_cfg.gpu.name, 0.70)
    total_cost = hourly * (wall_time_s / 3600.0)
    return round(total_cost / max(1, len(records)), 6)


def _persist(
    out_path: Path | str,
    records: list[EvalRecord],
    summary: EvalSummary,
    ablation_id: str | None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary.model_dump(),
        "records": [r.model_dump() for r in records],
        "ablation_id": ablation_id,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote eval results to %s", out_path)
