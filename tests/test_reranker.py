"""Tests for reranker fallback logging and candidate selection."""

from __future__ import annotations

import sys
import types

import pytest

from finqa_bot.retrieval import reranker as reranker_mod


class _FakeCrossEncoder:
    failures: dict[str, Exception] = {}

    def __init__(self, model_name: str, device: str = "cuda", trust_remote_code: bool = True) -> None:
        exc = self.failures.get(model_name)
        if exc is not None:
            raise exc
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code


def test_reranker_logs_candidate_failure_and_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    infos: list[str] = []

    class _StubLogger:
        def info(self, msg: str, *args: object) -> None:
            infos.append(msg % args if args else msg)

        def warning(self, msg: str, *args: object) -> None:
            warnings.append(msg % args if args else msg)

    _FakeCrossEncoder.failures = {
        "Qwen/Qwen3-Reranker-0.6B": RuntimeError("primary boom"),
        "BAAI/bge-reranker-v2-m3": RuntimeError("secondary boom"),
    }

    fake_module = types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(reranker_mod, "log", _StubLogger())

    reranker = reranker_mod.Reranker(model_name="Qwen/Qwen3-Reranker-0.6B", device="cuda")

    assert reranker.enabled
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert any("Failed to load reranker candidate Qwen/Qwen3-Reranker-0.6B" in line for line in warnings)
    assert any("Failed to load reranker candidate BAAI/bge-reranker-v2-m3" in line for line in warnings)
    assert any("using fallback cross-encoder/ms-marco-MiniLM-L-6-v2" in line for line in warnings)
