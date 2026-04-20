"""Tests for the DSL parser and deterministic executor."""

from __future__ import annotations

import math

import pytest

from finqa_bot.execution.dsl import ProgramParseError, dump_program, parse_program
from finqa_bot.execution.executor import DSLExecutor
from finqa_bot.execution.numbers import (
    extract_numbers_from_text,
    finqa_str_to_num,
    format_number,
    normalize_number,
)
from finqa_bot.types import DocumentContext, Step


def test_parse_program_roundtrip() -> None:
    text = "divide(9413, 20.01), divide(8249, 9.48), subtract(#0, #1)"
    steps = parse_program(text)
    assert [s.op for s in steps] == ["divide", "divide", "subtract"]
    assert steps[2].args == ["#0", "#1"]
    assert dump_program(steps) == text


def test_parse_program_strips_eof_sentinel() -> None:
    steps = parse_program("add(1, 2), EOF")
    assert len(steps) == 1
    assert steps[0].op == "add"


def test_parse_program_rejects_unknown_op() -> None:
    with pytest.raises(ProgramParseError):
        parse_program("magic_op(1, 2)")


def test_executor_handles_all_binary_ops(sample_doc: DocumentContext) -> None:
    executor = DSLExecutor()
    cases: list[tuple[str, list[float | str], float | str]] = [
        ("add", [1.0, 2.0], 3.0),
        ("subtract", [5.0, 2.0], 3.0),
        ("multiply", [3.0, 4.0], 12.0),
        ("divide", [10.0, 4.0], 2.5),
        ("exp", [2.0, 3.0], 8.0),
        ("greater", [5.0, 2.0], "yes"),
        ("greater", [1.0, 2.0], "no"),
    ]
    for op, args, expected in cases:
        result = executor.run([Step(op=op, args=args, source="table_1")], sample_doc)  # type: ignore[arg-type]
        assert result.ok, result.error
        assert result.value == expected


def test_executor_chains_backrefs(sample_doc: DocumentContext) -> None:
    program = [
        Step(op="divide", args=[100.0, 120.0], source="table_1"),
        Step(op="multiply", args=["#0", 100.0], source="table_1"),
    ]
    result = DSLExecutor().run(program, sample_doc)
    assert result.ok
    assert math.isclose(float(result.value), (100.0 / 120.0) * 100.0, rel_tol=1e-6)


def test_executor_resolves_constants(sample_doc: DocumentContext) -> None:
    # (120 - 100) / 100 * 100 to get percent change.
    program = [
        Step(op="subtract", args=[120.0, 100.0], source="table_1"),
        Step(op="divide", args=["#0", 100.0], source="table_1"),
        Step(op="multiply", args=["#1", "const_100"], source="const"),
    ]
    result = DSLExecutor().run(program, sample_doc)
    assert result.ok
    assert math.isclose(float(result.value), 20.0, rel_tol=1e-6)


def test_executor_divide_by_zero(sample_doc: DocumentContext) -> None:
    program = [Step(op="divide", args=[1.0, 0.0], source="table_1")]
    result = DSLExecutor().run(program, sample_doc)
    assert not result.ok
    assert result.error is not None


def test_executor_table_aggregates(sample_doc: DocumentContext) -> None:
    ex = DSLExecutor()
    out = ex.run([Step(op="table_sum", args=["Revenue"], source="table_1")], sample_doc)
    assert out.ok and float(out.value) == 220.0

    out = ex.run([Step(op="table_average", args=["Revenue"], source="table_1")], sample_doc)
    assert out.ok and float(out.value) == 110.0

    out = ex.run([Step(op="table_max", args=["Revenue"], source="table_1")], sample_doc)
    assert out.ok and float(out.value) == 120.0

    out = ex.run([Step(op="table_min", args=["Revenue"], source="table_1")], sample_doc)
    assert out.ok and float(out.value) == 100.0


def test_executor_missing_row_label(sample_doc: DocumentContext) -> None:
    program = [Step(op="table_sum", args=["Frobozz"], source="table_1")]
    result = DSLExecutor().run(program, sample_doc)
    assert not result.ok


def test_executor_empty_program(sample_doc: DocumentContext) -> None:
    result = DSLExecutor().run([], sample_doc)
    assert not result.ok
    assert "Empty program" in (result.error or "")


def test_normalize_number_accounting_negative() -> None:
    assert normalize_number("(123)") == -123.0
    assert normalize_number("$1,234.56") == pytest.approx(1234.56)
    # Percent is normalized to its fractional form, matching the upstream FinQA
    # convention (so that a program answer of 0.15 compares equal to gold "15%").
    assert normalize_number("15%") == pytest.approx(0.15)
    assert normalize_number("2.5 million") == pytest.approx(2_500_000.0)
    assert normalize_number("1.2 billion") == pytest.approx(1_200_000_000.0)
    assert normalize_number("const_100") == 100.0
    assert normalize_number("const_m1") == -1.0
    assert normalize_number("n/a") is None


def test_finqa_str_to_num_matches_upstream() -> None:
    assert finqa_str_to_num("n/a") == "n/a"
    assert finqa_str_to_num("$1,234") == 1234.0
    assert finqa_str_to_num("50%") == 0.5
    assert finqa_str_to_num("const_100") == 100.0


def test_extract_numbers_from_text_picks_up_all() -> None:
    nums = extract_numbers_from_text("Revenue rose 12.5% from $1,200 million to $1,350 million.")
    assert 12.5 in nums or pytest.approx(12.5) in nums
    assert 1200.0 in nums
    assert 1350.0 in nums


def test_format_number_percent_and_decimal() -> None:
    assert format_number(12.5, "percent").endswith("%")
    assert "12.50" in format_number(12.5, "decimal") or "12.5" in format_number(12.5, "decimal")
