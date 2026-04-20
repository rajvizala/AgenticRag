"""Scale extractor tests."""

from __future__ import annotations

from finqa_bot.retrieval.scale_extractor import ScaleExtractor
from finqa_bot.types import DocumentContext


def _doc(**kwargs: object) -> DocumentContext:
    base = {
        "doc_id": "unit/test",
        "pre_text": [],
        "post_text": [],
        "table": [],
        "scale_factor": 1.0,
        "scale_source": "default",
    }
    base.update(kwargs)
    return DocumentContext(**base)  # type: ignore[arg-type]


def test_scale_in_millions_from_pre_text() -> None:
    doc = _doc(pre_text=["Amounts are ( in millions ).", "Revenue..."])
    result = ScaleExtractor().extract(doc)
    assert result.factor == 1_000_000.0
    assert result.label == "millions"
    assert result.source.startswith("pre_text")


def test_scale_in_thousands_from_table_header() -> None:
    doc = _doc(table=[["( dollars in thousands )", "2023", "2024"], ["Revenue", "1,000", "1,200"]])
    result = ScaleExtractor().extract(doc)
    assert result.factor == 1_000.0
    assert result.label == "thousands"


def test_scale_in_billions_from_post_text() -> None:
    doc = _doc(post_text=["All amounts ( in billions ) unless otherwise stated."])
    result = ScaleExtractor().extract(doc)
    assert result.factor == 1_000_000_000.0
    assert result.label == "billions"


def test_scale_defaults_to_units() -> None:
    doc = _doc(pre_text=["Nothing here about scale."])
    result = ScaleExtractor().extract(doc)
    assert result.factor == 1.0
    assert result.label == "units"
    assert result.source == "default"


def test_scale_extractor_model_fallback_triggers_when_regex_fails() -> None:
    calls: list[str] = []

    def fallback(blob: str) -> tuple[float, str]:
        calls.append(blob)
        return 1_000_000.0, "millions"

    doc = _doc(table=[["year", "amount"], ["2023", "100"]])
    result = ScaleExtractor(model_fallback=fallback).extract(doc)
    assert calls, "Fallback should have been invoked when regex failed."
    assert result.factor == 1_000_000.0
    assert result.source == "model_fallback"
