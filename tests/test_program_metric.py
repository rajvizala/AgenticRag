"""Symbolic program equivalence tests."""

from __future__ import annotations

from finqa_bot.eval.finqa_metric import exe_equal, official_str_to_num
from finqa_bot.eval.program_metric import program_match_symbolic


def test_program_match_identical() -> None:
    assert program_match_symbolic("subtract(120, 100)", "subtract(120, 100)")


def test_program_match_equivalent_via_commutativity() -> None:
    assert program_match_symbolic("add(100, 20)", "add(20, 100)")


def test_program_match_rejects_different_result() -> None:
    assert not program_match_symbolic("subtract(120, 100)", "subtract(100, 120)")


def test_program_match_handles_rearranged_steps() -> None:
    # Both compute (120 - 100) * 100 / 100 = 20 but in a different step order.
    pred = "subtract(120, 100), multiply(#0, 100), divide(#1, 100)"
    gold = "subtract(120, 100), divide(#0, 100), multiply(#1, 100)"
    assert program_match_symbolic(pred, gold)


def test_exe_equal_respects_official_rounding() -> None:
    assert exe_equal(0.333333, 1 / 3)
    assert not exe_equal(0.5, 0.501)


def test_official_str_to_num_percent_divides_by_100() -> None:
    assert official_str_to_num("50%") == 0.5
    assert official_str_to_num("const_50") == 50.0
    assert official_str_to_num("const_m1") == -1.0
    assert official_str_to_num("$1,234") == 1234.0
