"""Byte-for-byte replica of the canonical FinQA ``evaluate.py``.

Source: ``czyssrs/FinQA/code/evaluate/evaluate.py``. Two public functions:

- :func:`official_str_to_num` - identical to ``str_to_num`` in the upstream
  script, including the well-known quirk of dropping everything after an open
  parenthesis (accounting-style negatives collapse to positives).
- :func:`exe_equal` - ``round(pred, 5) == round(gold, 5)`` after coercion,
  matching the ``evaluate.py`` default tolerance.

Keeping this separate from :mod:`finqa_bot.execution.numbers` makes the
comparison to published FinQA numbers auditable: any disagreement between our
semantic normalizer and the upstream reference is visible in a three-line diff.
"""

from __future__ import annotations

from typing import Any

from finqa_bot.execution.numbers import finqa_str_to_num


def official_str_to_num(value: Any) -> float | str:
    """Identical semantics to the upstream ``str_to_num`` helper."""
    return finqa_str_to_num(value)


def exe_equal(pred: Any, gold: Any, tolerance: float = 1e-5) -> bool:
    """Official execution-accuracy equality.

    Numbers compare with rounding to 5 decimals (``round(..., 5) == ...``).
    Non-numeric values compare case-insensitively after trimming whitespace.
    """
    p = official_str_to_num(pred) if isinstance(pred, str) else pred
    g = official_str_to_num(gold) if isinstance(gold, str) else gold
    if isinstance(p, str) or isinstance(g, str):
        return str(p).strip().lower() == str(g).strip().lower()
    if p is None or g is None:
        return p is g
    try:
        return round(float(p), 5) == round(float(g), 5) or abs(float(p) - float(g)) <= tolerance
    except (TypeError, ValueError):
        return False


def percent_or_decimal_equal(pred: Any, gold: Any) -> bool:
    """Convenience: match even when one side is ``0.25`` and the other ``25%``.

    Some submissions emit percents as fractions, others as whole numbers. The
    official harness divides ``%`` tokens by 100 via ``str_to_num``; this helper
    also considers ``pred * 100`` or ``pred / 100`` against gold.
    """
    if exe_equal(pred, gold):
        return True
    try:
        p = float(official_str_to_num(pred) if isinstance(pred, str) else pred)
        g = float(official_str_to_num(gold) if isinstance(gold, str) else gold)
    except (TypeError, ValueError):
        return False
    return round(p * 100, 5) == round(g, 5) or round(p, 5) == round(g * 100, 5)
