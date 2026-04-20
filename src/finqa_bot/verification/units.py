"""Unit + scale sanity checks.

Cheap sanity tests on the ``AnswerEnvelope``:
- ``answer_form`` consistency: percent answers should be 0..1 (we store as
  decimal); counts should be non-negative integers; booleans should be
  ``yes`` / ``no``.
- Scale consistency: if the document reports in millions, a huge absolute
  answer (e.g. 10 billion) is a red flag unless the prompt explicitly asked
  for an absolute dollar amount.

This is intentionally conservative. Failing a unit check does not trigger
refusal by itself; it bumps a "unit_correct" flag that shows up in
:class:`finqa_bot.types.EvalRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass

from finqa_bot.types import AnswerEnvelope, DocumentContext


@dataclass(frozen=True)
class UnitCheckResult:
    """Result of a unit / scale sanity pass."""

    ok: bool
    warnings: list[str]


def check_units(envelope: AnswerEnvelope, document: DocumentContext | None = None) -> UnitCheckResult:
    """Return a list of warnings about ``envelope`` plus an overall OK flag."""
    warnings: list[str] = []
    val = envelope.answer_value

    if envelope.answer_form == "boolean":
        if not isinstance(val, str) or val.strip().lower() not in {"yes", "no"}:
            warnings.append(f"boolean form expects 'yes'/'no', got {val!r}")
    else:
        if not isinstance(val, int | float):
            warnings.append(f"non-boolean form expects numeric value, got {type(val).__name__}")

    if (
        envelope.answer_form == "percent"
        and isinstance(val, int | float)
        and not -10.0 <= float(val) <= 10.0
    ):
        warnings.append(
            f"percent form expects |value| <= 10 (stored as decimal); got {val}"
        )

    if (
        envelope.answer_form == "count"
        and isinstance(val, int | float)
        and (float(val) < 0 or (float(val) - round(float(val))) > 1e-6)
    ):
        warnings.append(f"count form expects non-negative integer, got {val}")

    if document and isinstance(val, int | float):
        doc_scale = document.scale_factor
        declared_scale = {
            "units": 1.0,
            "thousands": 1e3,
            "millions": 1e6,
            "billions": 1e9,
        }.get(envelope.scale, 1.0)
        if doc_scale >= 1e6 and declared_scale < 1.0 and abs(float(val)) > 1e7:
            warnings.append(
                "document is in millions but envelope.scale says units with a very large value; "
                "likely missed const_1000000 divide."
            )

    return UnitCheckResult(ok=not warnings, warnings=warnings)
