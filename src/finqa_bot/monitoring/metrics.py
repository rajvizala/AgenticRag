"""Prometheus metrics for the FinQA bot.

We expose a minimal, FinQA-specific set of metrics alongside vLLM's own
``/metrics`` endpoint. These are the counters / gauges / histograms the
Grafana dashboard in ``ops/grafana/`` reads from.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.registry import CollectorRegistry


class MetricsRegistry:
    """Wraps a :class:`prometheus_client.CollectorRegistry` with our named metrics."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or REGISTRY

        self.questions_total = Counter(
            "finqa_questions_total",
            "Total questions processed by the graph.",
            registry=self.registry,
        )
        self.refusals_total = Counter(
            "finqa_refusals_total",
            "Total calibrated refusals.",
            ["reason"],
            registry=self.registry,
        )
        self.ungrounded_total = Counter(
            "finqa_ungrounded_programs_total",
            "Total programs where at least one literal was ungrounded.",
            registry=self.registry,
        )
        self.executions_total = Counter(
            "finqa_executions_total",
            "Total DSL-executor invocations.",
            ["ok"],
            registry=self.registry,
        )
        self.routing_total = Counter(
            "finqa_routing_total",
            "Router classifications.",
            ["category", "route"],
            registry=self.registry,
        )
        self.retrieval_recall = Gauge(
            "finqa_retrieval_recall_rolling",
            "Rolling retrieval recall over the golden set.",
            registry=self.registry,
        )
        self.psi = Gauge(
            "finqa_input_psi",
            "Population stability index on input embeddings.",
            registry=self.registry,
        )
        self.refusal_rate = Gauge(
            "finqa_refusal_rate_rolling",
            "Rolling refusal rate.",
            registry=self.registry,
        )
        self.consistency_agreement = Histogram(
            "finqa_consistency_agreement",
            "Self-consistency agreement of answered questions.",
            buckets=(0.0, 0.34, 0.5, 0.67, 0.8, 0.9, 1.0),
            registry=self.registry,
        )
        self.latency_seconds = Histogram(
            "finqa_graph_latency_seconds",
            "End-to-end graph latency per question.",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
            registry=self.registry,
        )
        self.tokens_prompt = Counter(
            "finqa_prompt_tokens_total",
            "Total prompt tokens consumed.",
            registry=self.registry,
        )
        self.tokens_completion = Counter(
            "finqa_completion_tokens_total",
            "Total completion tokens produced.",
            registry=self.registry,
        )

    def export(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def record_graph_outcome(self, state: dict[str, Any], elapsed_s: float) -> None:
        """Update the in-registry metrics from a finished graph state."""
        self.questions_total.inc()
        self.latency_seconds.observe(max(0.0, float(elapsed_s)))

        routing = state.get("routing")
        if routing is not None:
            self.routing_total.labels(
                category=getattr(routing, "category", "unknown"),
                route=getattr(routing, "route", "unknown"),
            ).inc()

        refused = bool(state.get("refused"))
        if refused:
            self.refusals_total.labels(reason=state.get("refusal_reason") or "unknown").inc()

        ground = state.get("groundedness")
        if ground is not None and not getattr(ground, "ok", True):
            self.ungrounded_total.inc()

        execution = state.get("execution")
        if execution is not None:
            self.executions_total.labels(ok=str(bool(getattr(execution, "ok", False))).lower()).inc()

        agreement = float(state.get("consistency_agreement", 0.0) or 0.0)
        self.consistency_agreement.observe(agreement)


registry = MetricsRegistry()
