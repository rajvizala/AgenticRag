"""End-to-end smoke test.

Builds a graph with a mocked structured LLM client and runs a single question
through every node. No GPU, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from finqa_bot.config import (
    EmbeddingConfig,
    GenerationConfig,
    GpuConfig,
    GpuSpec,
    RerankerConfig,
    RetrievalConfig,
    RouterConfig,
    SelfConsistencyConfig,
    ServingConfig,
    Settings,
    VllmGeneratorConfig,
    VllmSpecialistConfig,
)
from finqa_bot.graph.graph import BuildOptions, build_graph
from finqa_bot.graph.nodes import PipelineDeps
from finqa_bot.graph.router import Router
from finqa_bot.types import AnswerEnvelope, DocumentContext, RoutingDecision, Step


def _fake_gpu_cfg() -> GpuConfig:
    return GpuConfig(
        gpu=GpuSpec(name="t4", arch="turing", vram_gb=16),
        generator=VllmGeneratorConfig(model="mock/gen"),
        specialist=VllmSpecialistConfig(model="mock/spec", enabled=False),
        router=RouterConfig(enabled=False, backend="transformers"),
        embedding=EmbeddingConfig(model="mock/embed", device="cpu"),
        reranker=RerankerConfig(model="mock/rerank", device="cpu"),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(self_consistency=SelfConsistencyConfig(enabled=False)),
        serving=ServingConfig(),
    )


class _FakeStructuredClient:
    """Drop-in replacement for the LangChain structured-output runnable."""

    def __init__(self, envelope: AnswerEnvelope) -> None:
        self._env = envelope

    def bind(self, **_: Any) -> _FakeStructuredClient:
        return self

    async def ainvoke(self, _messages: Any) -> AnswerEnvelope:
        return self._env


class _FakeRouter(Router):
    """Router that always returns a generalist decision."""

    def __init__(self) -> None:
        pass

    async def classify(self, question: str) -> RoutingDecision:  # type: ignore[override]
        return RoutingDecision(
            category="multi_step", route="generalist", confidence=0.9, reason="test"
        )


@pytest.mark.asyncio
async def test_end_to_end_smoke(sample_doc: DocumentContext) -> None:
    """Build the graph with mocked deps and run it on a canned question."""
    env = AnswerEnvelope(
        program=[Step(op="subtract", args=[120.0, 100.0], source="table_1")],
        answer_value=20.0,
        answer_form="decimal",
        scale="units",
        grounded_numbers=[120.0, 100.0],
        confidence=0.95,
        rationale="120 - 100 = 20",
    )
    gpu_cfg = _fake_gpu_cfg()
    settings = Settings()

    runner = build_graph(
        gpu_cfg=gpu_cfg,
        settings=settings,
        options=BuildOptions(
            index=None,
            feature_flags={"router": False, "groundedness": False, "self_consistency": False},
            enable_retrieval=False,
        ),
    )
    deps: PipelineDeps = runner.deps
    deps.structured_client_generalist = _FakeStructuredClient(env)
    deps.structured_client_specialist = _FakeStructuredClient(env)
    deps.router = _FakeRouter()

    from finqa_bot.graph.state import initial_state

    seeded = initial_state("What was the change in revenue from 2023 to 2024?")
    seeded["document"] = sample_doc
    state = await runner.ainvoke(seeded)

    assert state.get("envelope") is not None
    execution = state.get("execution")
    assert execution is not None and execution.ok
    assert float(execution.value) == 20.0
    assert not state.get("refused")
