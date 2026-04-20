"""Tests for API streaming behavior and runtime config overrides."""

from __future__ import annotations

import asyncio
import json

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
    VllmGeneratorConfig,
    VllmSpecialistConfig,
)
from finqa_bot.ui import api as api_mod


def _fake_gpu_cfg(arch: str = "turing") -> GpuConfig:
    return GpuConfig(
        gpu=GpuSpec(name=arch, arch=arch, vram_gb=16 if arch == "turing" else 24),
        generator=VllmGeneratorConfig(model="mock/gen"),
        specialist=VllmSpecialistConfig(model="mock/spec", enabled=False),
        router=RouterConfig(enabled=True, backend="transformers", device="cuda"),
        embedding=EmbeddingConfig(model="mock/embed", device="cuda"),
        reranker=RerankerConfig(model="mock/rerank", device="cuda"),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(self_consistency=SelfConsistencyConfig(enabled=False)),
        serving=ServingConfig(),
    )


def test_prepare_runtime_gpu_config_moves_t4_aux_models_to_cpu() -> None:
    cfg = _fake_gpu_cfg("turing")

    adjusted = api_mod._prepare_runtime_gpu_config(cfg)

    assert adjusted.embedding.device == "cpu"
    assert adjusted.reranker.device == "cpu"
    assert adjusted.router.device == "cpu"
    assert cfg.embedding.device == "cuda"


def test_prepare_runtime_gpu_config_leaves_non_turing_unchanged() -> None:
    cfg = _fake_gpu_cfg("ada")

    adjusted = api_mod._prepare_runtime_gpu_config(cfg)

    assert adjusted.embedding.device == "cuda"
    assert adjusted.reranker.device == "cuda"
    assert adjusted.router.device == "cuda"


def test_sse_stream_emits_final_without_second_graph_run(monkeypatch) -> None:
    class _FakeRunner:
        async def astream(self, question=None, doc_id=None):  # type: ignore[no-untyped-def]
            yield {"route": {"routing": {"route": "generalist"}}}
            yield {"format": {"answer_text": "done", "refused": False}}

        async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("ainvoke should not be called by /chat streaming")

    async def _collect() -> list[dict[str, object]]:
        async def _fake_ensure_runner():
            return _FakeRunner()

        monkeypatch.setattr(api_mod, "_ensure_runner", _fake_ensure_runner)
        raw = []
        async for chunk in api_mod._sse_stream("q", None):
            raw.append(json.loads(chunk.decode().removeprefix("data: ").strip()))
        return raw

    events = asyncio.run(_collect())

    assert [event["event"] for event in events] == ["node", "node", "final"]
    assert events[-1]["state"]["answer_text"] == "done"
