"""ScaleExtractor: per-document scale-factor detection.

The regex-based implementation from :mod:`finqa_bot.data.chunking` is fast and
handles ~95% of FinQA tables out of the box ("( in millions )", "( dollars in
thousands )", etc.). For the remaining tail we expose an optional
model-backed fallback: pass a ``predict`` callable that takes a header-ish
text blob and returns a ``(scale_factor, scale_label)`` tuple.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from finqa_bot.data.chunking import SCALE_PATTERNS, extract_scale_factor
from finqa_bot.types import DocumentContext


@dataclass
class ScaleResult:
    """Output of the extractor."""

    factor: float
    label: str
    source: str
    confidence: float


class ScaleExtractor:
    """Extract the reporting scale of a financial document.

    By default uses the regex table in :mod:`finqa_bot.data.chunking`. Supply
    ``model_fallback`` (a callable that predicts a scale on the header blob)
    to try a tiny classifier for anything the regex misses.
    """

    def __init__(
        self,
        model_fallback: Callable[[str], tuple[float, str]] | None = None,
    ) -> None:
        self._fallback = model_fallback

    def extract(self, doc: DocumentContext) -> ScaleResult:
        factor, source = extract_scale_factor(doc.pre_text, doc.post_text, doc.table)
        if factor != 1.0:
            return ScaleResult(factor=factor, label=_label_for(factor), source=source, confidence=1.0)

        if self._fallback is not None:
            header_blob = _header_blob(doc)
            try:
                f2, label = self._fallback(header_blob)
                if f2 and f2 != 1.0:
                    return ScaleResult(factor=f2, label=label, source="model_fallback", confidence=0.7)
            except Exception:
                pass
        return ScaleResult(factor=1.0, label="units", source="default", confidence=0.5)


def _label_for(factor: float) -> str:
    if factor >= 1e9:
        return "billions"
    if factor >= 1e6:
        return "millions"
    if factor >= 1e3:
        return "thousands"
    return "units"


def _header_blob(doc: DocumentContext) -> str:
    parts: list[str] = []
    if doc.table:
        parts.append(" ".join(doc.table[0]))
    if doc.pre_text:
        parts.extend(doc.pre_text[:2])
    if doc.post_text:
        parts.extend(doc.post_text[:1])
    return " \n ".join(parts)


__all__ = ["SCALE_PATTERNS", "ScaleExtractor", "ScaleResult"]
