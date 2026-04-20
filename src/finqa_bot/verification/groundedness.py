"""Groundedness verifier.

For every numeric literal used in a generated program, verify that the same
number appears somewhere in the retrieved context (after normalization).
Literals that reference back-steps (``#n``) or FinQA constants (``const_*``)
are exempt, as are string row labels for ``table_*`` aggregate operators
(those are resolved by the executor).

Why this matters: in FinQA's error analysis (Chen et al. 2021, Figures 4-5),
the two dominant failure modes are (a) hallucinated numbers and (b) wrong
row / year selection. Numeric grounding catches (a) outright; the executor's
row-label resolution catches (b).

The :class:`NumberTolerance` knob controls fuzzy matching (e.g. ``"1,234.5"``
vs ``"1234.50"`` vs ``"1.2345e3"``). Default tolerance is 0.1% relative plus a
small absolute floor for rounding.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finqa_bot.execution.numbers import extract_numbers_from_chunks, normalize_number
from finqa_bot.types import GroundednessResult, RetrievalHit, Step


@dataclass(frozen=True)
class NumberTolerance:
    """Fuzzy-match tolerance for numeric grounding."""

    rel: float = 1e-3
    abs_floor: float = 1e-2

    def matches(self, a: float, b: float) -> bool:
        return abs(a - b) <= max(self.abs_floor, self.rel * max(abs(a), abs(b), 1.0))


def _iter_literals(program: Iterable[Step]) -> Iterable[float]:
    """Yield numeric literals used by the program (skipping #n, const_*, labels)."""
    for step in program:
        is_table_agg = step.op.startswith("table_")
        for arg in step.args:
            if isinstance(arg, int | float):
                yield float(arg)
                continue
            if not isinstance(arg, str):
                continue
            s = arg.strip()
            if s.startswith("#") or s.startswith("const_"):
                continue
            if is_table_agg:
                # First-arg for table_* is a row label, not a number.
                continue
            val = normalize_number(s)
            if val is not None:
                yield float(val)


def check_groundedness(
    program: list[Step],
    context_texts: list[str],
    tolerance: NumberTolerance = NumberTolerance(),
) -> GroundednessResult:
    """Verify that every literal in ``program`` appears in ``context_texts``.

    Returns :class:`GroundednessResult`; ``ok`` is True iff every literal is
    matched to some context number within tolerance.
    """
    context_numbers = sorted(set(extract_numbers_from_chunks(context_texts)))
    missing: list[float] = []
    for lit in _iter_literals(program):
        if not any(tolerance.matches(lit, cn) for cn in context_numbers):
            missing.append(lit)
    return GroundednessResult(
        ok=not missing,
        missing=missing,
        normalized_context_numbers=context_numbers,
    )


class GroundednessChecker:
    """Object-oriented convenience wrapper for callers that keep state."""

    def __init__(self, tolerance: NumberTolerance | None = None) -> None:
        self.tolerance = tolerance or NumberTolerance()

    def check(self, program: list[Step], hits: list[RetrievalHit]) -> GroundednessResult:
        context_texts = [h.chunk.text for h in hits]
        return check_groundedness(program, context_texts, self.tolerance)
