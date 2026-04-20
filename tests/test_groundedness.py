"""Groundedness verifier tests."""

from __future__ import annotations

from finqa_bot.types import RetrievalHit, Step, TableChunk
from finqa_bot.verification.groundedness import (
    GroundednessChecker,
    NumberTolerance,
    check_groundedness,
)


def _hit(text: str, cid: str = "table_1") -> RetrievalHit:
    chunk = TableChunk(id=cid, doc_id="doc", chunk_type="table_row", text=text)
    return RetrievalHit(chunk=chunk, score=1.0, source="dense")


def test_grounded_program_passes() -> None:
    program = [Step(op="divide", args=[120.0, 100.0], source="table_1")]
    hits = [_hit("Revenue: 100 in 2023 and 120 in 2024 (millions).")]
    result = GroundednessChecker().check(program, hits)
    assert result.ok, result.missing


def test_ungrounded_literal_is_flagged() -> None:
    program = [Step(op="multiply", args=[999999.0, 1.0], source="table_1")]
    hits = [_hit("Revenue: 100 in 2023 and 120 in 2024.")]
    result = GroundednessChecker().check(program, hits)
    assert not result.ok
    assert 999999.0 in result.missing


def test_back_references_and_constants_are_exempt() -> None:
    program = [
        Step(op="divide", args=[120.0, 100.0], source="table_1"),
        Step(op="multiply", args=["#0", "const_100"], source="const"),
    ]
    hits = [_hit("Revenue: 100 and 120.")]
    result = GroundednessChecker().check(program, hits)
    assert result.ok


def test_row_labels_are_exempt_from_grounding() -> None:
    program = [Step(op="table_sum", args=["Revenue"], source="table_1")]
    hits = [_hit("Revenue row: 100 + 120 + 140.")]
    result = GroundednessChecker().check(program, hits)
    assert result.ok


def test_tolerance_accepts_rounding() -> None:
    # 1234.50 should match 1,234.5 within tolerance; both literals are present
    # in the retrieved context.
    program = [Step(op="multiply", args=[1234.50, 2024.0], source="table_1")]
    hits = [_hit("Item reported as 1,234.5 in 2024 filings.")]
    result = check_groundedness(program, [h.chunk.text for h in hits], NumberTolerance(rel=1e-3))
    assert result.ok, result.missing


def test_accounting_negatives_are_normalized() -> None:
    program = [Step(op="add", args=[-500.0, 100.0], source="table_1")]
    hits = [_hit("Net income loss was ($500) while other income was 100.")]
    result = check_groundedness(program, [h.chunk.text for h in hits])
    assert result.ok, result.missing
