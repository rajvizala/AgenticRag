"""LangGraph node implementations.

Each public function here is a LangGraph node: it takes ``state`` and returns
a partial ``state`` update. Nodes are intentionally small and idempotent -
if a node's output is already populated on state (e.g. by a test or a
previous invocation), it short-circuits.

Pipeline order::

    ensure_document -> route -> retrieve -> extract_scale -> generate
    -> execute -> verify -> self_consistency -> (format | refuse)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from finqa_bot.config import GpuConfig, Settings
from finqa_bot.execution.dsl import dump_program
from finqa_bot.execution.executor import DSLExecutor
from finqa_bot.execution.numbers import format_number, normalize_number
from finqa_bot.graph.prompts import SYSTEM_PROMPT, build_user_message
from finqa_bot.graph.router import Router
from finqa_bot.logging import get_logger
from finqa_bot.retrieval.hybrid import HybridRetriever
from finqa_bot.retrieval.indexer import CorpusIndex
from finqa_bot.retrieval.scale_extractor import ScaleExtractor
from finqa_bot.serving.openai_client import build_structured_client
from finqa_bot.types import (
    AnswerEnvelope,
    DocumentContext,
    ExecutionResult,
    GraphState,
    GroundednessResult,
    RetrievalHit,
    RoutingDecision,
)
from finqa_bot.verification.groundedness import GroundednessChecker
from finqa_bot.verification.units import check_units

log = get_logger(__name__)


CONSISTENCY_REFUSAL_THRESHOLD = 0.20


# ----- Dependency container ---------------------------------------------------


@dataclass
class PipelineDeps:
    """Shared, process-lived dependencies wired into every node via closure.

    Some fields are optional so the graph runs in reduced form (e.g. without a
    retriever for unit tests where hits are provided in state).
    """

    cfg: GpuConfig
    settings: Settings
    index: CorpusIndex | None = None
    retriever: HybridRetriever | None = None
    scale_extractor: ScaleExtractor | None = None
    router: Router | None = None
    executor: DSLExecutor | None = None
    groundedness: GroundednessChecker | None = None
    structured_client_generalist: Any = None
    structured_client_specialist: Any = None
    feature_flags: dict[str, bool] | None = None

    def flag(self, name: str, default: bool = True) -> bool:
        flags = self.feature_flags or {}
        return bool(flags.get(name, default))


# ----- Utility helpers --------------------------------------------------------


def _append_trace(state: GraphState, node: str, **kv: Any) -> list[dict[str, Any]]:
    trace = list(state.get("trace") or [])
    trace.append({"node": node, **kv})
    return trace


def _ensure_executor(deps: PipelineDeps) -> DSLExecutor:
    if deps.executor is None:
        deps.executor = DSLExecutor()
    return deps.executor


def _ensure_groundedness(deps: PipelineDeps) -> GroundednessChecker:
    if deps.groundedness is None:
        deps.groundedness = GroundednessChecker()
    return deps.groundedness


def _ensure_scale_extractor(deps: PipelineDeps) -> ScaleExtractor:
    if deps.scale_extractor is None:
        deps.scale_extractor = ScaleExtractor()
    return deps.scale_extractor


def _ensure_router(deps: PipelineDeps) -> Router:
    if deps.router is None:
        deps.router = Router(deps.cfg, deps.settings)
    return deps.router


def _get_structured_client(deps: PipelineDeps, specialist: bool) -> Any:
    attr = "structured_client_specialist" if specialist else "structured_client_generalist"
    existing = getattr(deps, attr)
    if existing is not None:
        return existing
    model = deps.settings.llm_specialist_model if specialist else deps.settings.llm_model
    try:
        client = build_structured_client(
            settings=deps.settings,
            schema=AnswerEnvelope,
            model=model,
            temperature=deps.cfg.generation.temperature,
            top_p=deps.cfg.generation.top_p,
            max_tokens=1200,
        )
    except Exception as exc:
        log.warning("Failed to build %s structured client: %s", "specialist" if specialist else "generalist", exc)
        return None
    setattr(deps, attr, client)
    return client


# ----- Nodes ------------------------------------------------------------------


def node_ensure_document(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Populate ``state['document']`` from the corpus index if not already set."""

    async def _run(state: GraphState) -> GraphState:
        if state.get("document") is not None:
            return {}  # already populated
        doc_id = state.get("doc_id")
        if doc_id and deps.index is not None:
            doc = deps.index.document(doc_id)
            if doc is not None:
                return {
                    "document": doc,
                    "trace": _append_trace(state, "ensure_document", doc_id=doc_id, ok=True),
                }
        return {
            "trace": _append_trace(state, "ensure_document", doc_id=doc_id, ok=False),
        }

    return _run


def node_route(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Classify the question and decide whether to defer to the specialist."""

    async def _run(state: GraphState) -> GraphState:
        if state.get("routing") is not None:
            return {}
        if not deps.flag("router", default=True):
            decision = RoutingDecision(
                category="multi_step",
                route="generalist",
                confidence=1.0,
                reason="router_disabled",
            )
            return {
                "routing": decision,
                "trace": _append_trace(state, "route", decision=decision.model_dump(), skipped=True),
            }
        router = _ensure_router(deps)
        decision = await router.classify(state["question"])
        return {
            "routing": decision,
            "trace": _append_trace(state, "route", decision=decision.model_dump()),
        }

    return _run


def node_retrieve(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Populate ``state['hits']`` by running hybrid retrieval."""

    async def _run(state: GraphState) -> GraphState:
        if state.get("hits"):
            return {}
        if deps.retriever is None:
            return {
                "hits": [],
                "trace": _append_trace(state, "retrieve", skipped=True, reason="no_retriever"),
            }
        t0 = perf_counter()
        hits = deps.retriever.retrieve(
            query=state["question"],
            doc_id=state.get("doc_id"),
        )
        dt = perf_counter() - t0
        return {
            "hits": hits,
            "trace": _append_trace(
                state,
                "retrieve",
                n_hits=len(hits),
                hit_ids=[h.chunk.id for h in hits],
                latency_s=round(dt, 4),
            ),
        }

    return _run


def node_extract_scale(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Set ``state['scale']`` from the document (or default to 1.0)."""

    async def _run(state: GraphState) -> GraphState:
        doc = state.get("document")
        if doc is None:
            return {
                "scale": 1.0,
                "trace": _append_trace(state, "extract_scale", scale=1.0, reason="no_document"),
            }
        if not deps.flag("scale_extraction", default=True):
            return {
                "scale": 1.0,
                "trace": _append_trace(state, "extract_scale", scale=1.0, reason="disabled"),
            }
        result = _ensure_scale_extractor(deps).extract(doc)
        return {
            "scale": result.factor,
            "trace": _append_trace(
                state,
                "extract_scale",
                scale=result.factor,
                label=result.label,
                source=result.source,
            ),
        }

    return _run


def _scale_label(factor: float) -> str:
    if factor >= 1e9:
        return "billions"
    if factor >= 1e6:
        return "millions"
    if factor >= 1e3:
        return "thousands"
    return "units"


def _is_ratio_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ("portion", "ratio", "percent of", "what percentage", "share"))


def _is_change_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ("change", "difference", "increase", "decrease", "grew", "decline"))


def _expects_positive_change(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ("increase", "grew", "growth"))


def _expects_negative_change(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ("decrease", "decline", "drop"))


def _is_literal_number_arg(arg: float | str) -> bool:
    if isinstance(arg, int | float):
        return True
    if not isinstance(arg, str):
        return False
    s = arg.strip()
    if s.startswith("#") or s.startswith("const_"):
        return False
    return normalize_number(s) is not None


def _flip_first_literal_subtract(env: AnswerEnvelope) -> AnswerEnvelope | None:
    patched = env.model_copy(deep=True)
    for step in patched.program:
        if step.op != "subtract" or len(step.args) != 2:
            continue
        a, b = step.args
        if not (_is_literal_number_arg(a) and _is_literal_number_arg(b)):
            continue
        step.args = [b, a]
        return patched
    return None


def _convert_single_subtract_to_divide(env: AnswerEnvelope) -> AnswerEnvelope | None:
    if len(env.program) != 1:
        return None
    step = env.program[0]
    if step.op != "subtract" or len(step.args) != 2:
        return None
    a, b = step.args
    if not (_is_literal_number_arg(a) and _is_literal_number_arg(b)):
        return None
    patched = env.model_copy(deep=True)
    patched.program[0].op = "divide"
    return patched


def _repair_candidate_for_question(
    question: str,
    env: AnswerEnvelope,
    result: ExecutionResult,
    executor: DSLExecutor,
    document: DocumentContext,
) -> tuple[AnswerEnvelope, ExecutionResult, str | None]:
    """Apply lightweight deterministic repairs for frequent program mistakes."""
    if not result.ok:
        return env, result, None

    # Ratio/portion questions are often mis-generated as subtract(a, b).
    if _is_ratio_question(question):
        ratio_env = _convert_single_subtract_to_divide(env)
        if ratio_env is not None:
            ratio_result = executor.run(ratio_env.program, document)
            if ratio_result.ok and isinstance(ratio_result.value, int | float):
                ratio_val = float(ratio_result.value)
                if -0.25 <= ratio_val <= 2.5:
                    if "percent" in question.lower():
                        ratio_env.answer_form = "percent"
                    else:
                        ratio_env.answer_form = "ratio"
                    ratio_env.answer_value = ratio_result.value
                    return ratio_env, ratio_result, "ratio_subtract_to_divide"

    # Directional sign correction for change/increase/decrease questions.
    is_change = _is_change_question(question)
    is_diff = "difference" in question.lower()
    
    if (is_change or is_diff) and isinstance(result.value, int | float):
        value = float(result.value)
        wrong_sign = False
        if is_change:
            wrong_sign = (_expects_positive_change(question) and value < 0.0) or (
                _expects_negative_change(question) and value > 0.0
            )
        if is_diff and value < 0.0:
            wrong_sign = True

        if wrong_sign:
            flipped = _flip_first_literal_subtract(env)
            if flipped is not None:
                flipped_result = executor.run(flipped.program, document)
                if flipped_result.ok and isinstance(flipped_result.value, int | float):
                    flipped.answer_value = flipped_result.value
                    tag = "flip_subtract_for_difference" if is_diff and not is_change else "flip_subtract_direction"
                    return flipped, flipped_result, tag

    return env, result, None


def node_generate(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Generate ``K`` typed envelopes via structured LLM calls."""

    async def _run(state: GraphState) -> GraphState:
        if state.get("envelope_candidates"):
            return {}
        routing = state.get("routing") or RoutingDecision(
            category="multi_step", route="generalist", confidence=0.5, reason=""
        )
        forced_attempt = routing.route == "refuse"
        specialist = routing.route == "specialist" and not forced_attempt
        client = _get_structured_client(deps, specialist=specialist)
        if client is None:
            # Fall back to generalist if specialist unavailable.
            client = _get_structured_client(deps, specialist=False)
        if client is None:
            return {
                "envelope_candidates": [],
                "trace": _append_trace(state, "generate", skipped=True, reason="no_llm"),
            }

        hits = state.get("hits") or []
        scale = state.get("scale", 1.0)
        user_msg = build_user_message(
            question=state["question"],
            hits=hits,
            scale_label=_scale_label(scale),
            scale_source=str(state.get("document").scale_source if state.get("document") else "n/a"),
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        sc = deps.cfg.generation.self_consistency
        use_sc = sc.enabled and deps.flag("self_consistency", default=True)
        if not use_sc:
            k = 1
        elif routing.category in {"direct_lookup", "single_ratio"}:
            # Fast path for easy questions to reduce p90/p99 latency.
            k = 1
        else:
            k = sc.samples
        temps = [deps.cfg.generation.temperature]
        while len(temps) < k:
            temps.append(sc.sampling_temperature)

        use_grounding = deps.flag("groundedness", default=True)
        checker = _ensure_groundedness(deps) if use_grounding else None

        t0 = perf_counter()
        envelopes = await _sample_envelopes(client, messages, temps, hits=hits, checker=checker)

        # If the specialist model is not served (404) all samples fail and
        # envelopes is empty. Retry immediately with the generalist so the
        # pipeline does not silently refuse every question that the rule-based
        # router classifies as "specialist".
        if specialist and len(envelopes) == 0:
            log.warning(
                "[generate] specialist returned 0 envelopes (model not served?); "
                "retrying with generalist"
            )
            fallback = _get_structured_client(deps, specialist=False)
            if fallback is not None:
                envelopes = await _sample_envelopes(fallback, messages, temps, hits=hits, checker=checker)
                specialist = False

        dt = perf_counter() - t0

        log.info(
            "[generate] produced %d/%d envelopes in %.2fs",
            len(envelopes), len(temps), dt,
        )
        for idx, env in enumerate(envelopes):
            prog_ops = [s.op for s in env.program] if env.program else []
            log.info(
                "[generate] sample %d: program=%s args=%s answer_value=%r confidence=%.2f rationale=%r",
                idx,
                prog_ops,
                [[str(a) for a in s.args] for s in env.program],
                env.answer_value,
                env.confidence,
                env.rationale[:120],
            )

        return {
            "envelope_candidates": envelopes,
            "trace": _append_trace(
                state,
                "generate",
                k=len(envelopes),
                temps=temps[: len(envelopes)],
                specialist=specialist,
                forced_attempt=forced_attempt,
                latency_s=round(dt, 4),
            ),
        }

    return _run


async def _sample_envelopes(
    client: Any,
    messages: list[Any],
    temps: list[float],
    hits: list[RetrievalHit] | None = None,
    checker: GroundednessChecker | None = None,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> list[AnswerEnvelope]:
    """Fan out K LangChain structured-output calls at the given temperatures.

    Each sample independently retries up to ``max_retries`` times with
    exponential backoff so that transient vLLM connection errors (common
    during model warm-up on Colab T4) don't silently discard every
    candidate and force a refusal.
    """

    async def one(t: float) -> AnswerEnvelope | None:
        runnable = client.bind(temperature=t) if hasattr(client, "bind") else client
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await runnable.ainvoke(messages)
                if isinstance(result, dict):
                    result = AnswerEnvelope(**result)
                if isinstance(result, AnswerEnvelope):
                    if checker is not None and hits is not None:
                        ground = checker.check(result.program, hits)
                        if not ground.ok:
                            raise ValueError(f"Ungrounded numeric literals generated: {ground.missing}")
                    return result
                log.warning("Unexpected envelope type: %s", type(result).__name__)
                return None
            except Exception as exc:
                last_exc = exc
                delay = min(base_delay * (2 ** (attempt - 1)), 8.0)
                log.info(
                    "LLM call attempt %d/%d failed (%s); retrying in %.1fs …",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        log.warning("LLM call failed after %d retries (%s); skipping sample.", max_retries, last_exc)
        return None

    results = await asyncio.gather(*(one(t) for t in temps), return_exceptions=False)
    return [r for r in results if r is not None]


def node_execute(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Execute each candidate program."""

    async def _run(state: GraphState) -> GraphState:
        envelopes = state.get("envelope_candidates") or []
        doc: DocumentContext | None = state.get("document")
        hits: list[RetrievalHit] = state.get("hits") or []

        if not envelopes:
            log.warning("[execute] skipped: no envelopes")
            return {
                "executions": [],
                "trace": _append_trace(state, "execute", skipped=True, reason="no_envelope"),
            }

        # When no doc_id was pinned the document is None, but programs that only
        # use literal arithmetic (subtract, divide, etc.) don't need a real
        # document. Infer the most-likely document from the retrieval hits so
        # table_* ops also work; fall back to a synthetic empty document so that
        # literal programs at least execute rather than being silently dropped.
        doc_source = "state"
        if doc is None:
            if hits and deps.index is not None:
                doc_id_counts = Counter(h.chunk.doc_id for h in hits)
                best_doc_id = doc_id_counts.most_common(1)[0][0]
                doc = deps.index.document(best_doc_id)
                doc_source = f"inferred_from_hits:{best_doc_id}"
            if doc is None:
                fallback_id = hits[0].chunk.doc_id if hits else "unknown"
                doc = DocumentContext(doc_id=fallback_id)
                doc_source = f"synthetic_empty:{fallback_id}"

        log.info("[execute] using doc_id=%r source=%s", doc.doc_id, doc_source)

        executor = _ensure_executor(deps)
        executions: list[ExecutionResult] = []
        repaired_envelopes: list[AnswerEnvelope] = []
        for i, env in enumerate(envelopes):
            result = executor.run(env.program, doc)
            chosen_env, chosen_result, repair_tag = _repair_candidate_for_question(
                question=state["question"],
                env=env,
                result=result,
                executor=executor,
                document=doc,
            )
            if repair_tag:
                log.info("[execute] sample %d repaired via %s", i, repair_tag)
            repaired_envelopes.append(chosen_env)
            executions.append(chosen_result)
            log.info(
                "[execute] sample %d: ok=%s value=%r error=%r program_steps=%d",
                i, chosen_result.ok, chosen_result.value, chosen_result.error, len(chosen_env.program),
            )
        return {
            "envelope_candidates": repaired_envelopes,
            "executions": executions,
            "trace": _append_trace(
                state,
                "execute",
                n=len(executions),
                oks=[e.ok for e in executions],
                values=[e.value for e in executions],
            ),
        }

    return _run


def node_verify(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Run groundedness on each candidate."""

    async def _run(state: GraphState) -> GraphState:
        envelopes = state.get("envelope_candidates") or []
        hits = state.get("hits") or []
        if not envelopes:
            return {
                "trace": _append_trace(state, "verify", skipped=True, reason="no_envelope"),
            }
        if not deps.flag("groundedness", default=True):
            trivial = GroundednessResult(ok=True, missing=[], normalized_context_numbers=[])
            return {
                "groundedness": trivial,
                "trace": _append_trace(state, "verify", skipped=True, reason="groundedness_off"),
            }
        checker = _ensure_groundedness(deps)
        results = [checker.check(e.program, hits) for e in envelopes]
        for i, r in enumerate(results):
            log.info(
                "[verify] sample %d: grounded=%s missing=%s context_numbers=%s",
                i, r.ok, r.missing, r.normalized_context_numbers[:10],
            )
        # Aggregate: state.groundedness tracks the winner's; the vector goes into trace.
        return {
            "trace": _append_trace(
                state,
                "verify",
                per_candidate=[{"ok": r.ok, "missing": r.missing} for r in results],
            ),
            "groundedness": results[0],
        }

    return _run


def node_self_consistency(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Majority-vote over executed values, respecting groundedness."""

    async def _run(state: GraphState) -> GraphState:
        envelopes = state.get("envelope_candidates") or []
        executions = state.get("executions") or []
        hits = state.get("hits") or []

        log.info(
            "[self_consistency] envelopes=%d executions=%d",
            len(envelopes), len(executions),
        )
        for i, exe in enumerate(executions):
            log.info("[self_consistency] exe[%d]: ok=%s value=%r error=%r", i, exe.ok, exe.value, exe.error)

        if not envelopes or not executions:
            log.warning("[self_consistency] skipped: envelopes=%d executions=%d", len(envelopes), len(executions))
            return {
                "envelope": None,
                "execution": None,
                "consistency_agreement": 0.0,
                "trace": _append_trace(state, "consistency", skipped=True, reason="no_candidates"),
            }

        checker = _ensure_groundedness(deps)
        use_grounding = deps.flag("groundedness", default=True)

        indices: list[int] = []
        for i, (env, exe) in enumerate(zip(envelopes, executions, strict=False)):
            if not exe.ok:
                log.info("[self_consistency] dropping sample %d: exe.ok=False error=%r", i, exe.error)
                continue
            ground = checker.check(env.program, hits)
            if use_grounding and not ground.ok:
                log.info("[self_consistency] dropping sample %d: not grounded missing=%s", i, ground.missing)
                continue
            indices.append(i)
        if not indices and executions:
            # Nothing grounded: fall back to any successful execution.
            log.info("[self_consistency] groundedness fallback: trying any ok execution")
            for i, exe in enumerate(executions):
                if exe.ok:
                    indices.append(i)

        log.info("[self_consistency] valid indices after filtering: %s", indices)

        if not indices:
            log.warning("[self_consistency] no valid candidates — will refuse")
            return {
                "envelope": None,
                "execution": executions[0] if executions else None,
                "consistency_agreement": 0.0,
                "trace": _append_trace(state, "consistency", ok=False, reason="no_valid_candidate"),
            }

        values = [_key_for_value(executions[i].value) for i in indices]
        counter = Counter(values)
        winning_key, winning_count = counter.most_common(1)[0]
        total = len(indices)
        agreement = winning_count / float(total)
        log.info(
            "[self_consistency] vote: values=%s counter=%s winning=%r agreement=%.3f (threshold=%.2f)",
            values, dict(counter), winning_key, agreement,
            CONSISTENCY_REFUSAL_THRESHOLD,
        )
        winner_idx = next(i for i in indices if _key_for_value(executions[i].value) == winning_key)
        winner_env = envelopes[winner_idx]
        winner_exec = executions[winner_idx]
        winner_ground = checker.check(winner_env.program, hits) if use_grounding else None

        return {
            "envelope": winner_env,
            "execution": winner_exec,
            "groundedness": winner_ground,
            "consistency_agreement": agreement,
            "trace": _append_trace(
                state,
                "consistency",
                n=total,
                winner=winner_idx,
                value=winner_exec.value,
                agreement=round(agreement, 4),
            ),
        }

    return _run


def _key_for_value(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{round(float(value), 5)}"
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, sort_keys=True)


def node_decide(state: GraphState) -> str:
    """LangGraph conditional edge: which of ``format`` or ``refuse`` to call."""
    envelope = state.get("envelope")
    execution = state.get("execution")
    ground = state.get("groundedness")
    agreement = state.get("consistency_agreement", 0.0)
    if envelope is None or execution is None or not execution.ok:
        return "refuse"
    # Prefer answering in-domain FinQA questions even when grounding is imperfect,
    # but still refuse when both agreement and groundedness are weak.
    if agreement < CONSISTENCY_REFUSAL_THRESHOLD and (ground is not None and not ground.ok):
        return "refuse"
    return "format"


def node_format(deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    """Render a natural-language answer + citations."""

    async def _run(state: GraphState) -> GraphState:
        envelope = state.get("envelope")
        execution = state.get("execution")
        hits = state.get("hits") or []
        if envelope is None or execution is None:
            return {"refused": True, "refusal_reason": "missing_envelope"}
        value = execution.value if execution.value is not None else envelope.answer_value
        display = format_number(value, envelope.answer_form) if isinstance(value, int | float) else str(value)
        citations = ", ".join(sorted({h.chunk.id for h in hits}))
        unit_report = check_units(envelope, state.get("document"))
        warnings = ""
        if not unit_report.ok:
            warnings = " Notes: " + "; ".join(unit_report.warnings) + "."
        prog_str = dump_program(envelope.program)
        rationale = envelope.rationale.strip()
        answer_text = (
            f"Answer: {display}. Program: {prog_str}. Citations: {citations}.{warnings}"
            + (f" Reason: {rationale}" if rationale else "")
        )
        return {
            "answer_text": answer_text,
            "refused": False,
            "trace": _append_trace(
                state,
                "format",
                value=value,
                form=envelope.answer_form,
                unit_ok=unit_report.ok,
                unit_warnings=unit_report.warnings,
            ),
        }

    return _run


def node_refuse(_deps: PipelineDeps) -> Callable[[GraphState], GraphState]:
    async def _run(state: GraphState) -> GraphState:
        routing = state.get("routing")
        ground = state.get("groundedness")
        agreement = state.get("consistency_agreement", 0.0)
        reason_parts: list[str] = []
        if routing is not None and routing.route == "refuse":
            reason_parts.append("question_out_of_scope")
        if ground is not None and not ground.ok:
            reason_parts.append(f"ungrounded_literals={ground.missing}")
        if agreement < CONSISTENCY_REFUSAL_THRESHOLD:
            reason_parts.append(f"no_majority_agreement={agreement:.2f}")
        if not reason_parts:
            reason_parts.append("unknown")
        reason = "; ".join(reason_parts)
        return {
            "refused": True,
            "refusal_reason": reason,
            "answer_text": "I cannot answer this confidently from the provided filing.",
            "trace": _append_trace(state, "refuse", reason=reason),
        }

    return _run


def retrieved_hits_as_texts(hits: list[RetrievalHit]) -> list[str]:
    """Public helper used by non-graph callers (eval, tests)."""
    return [h.chunk.text for h in hits]
