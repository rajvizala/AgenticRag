"""Deterministic FinQA DSL executor.

Given a typed program (``list[Step]``) and a document (tables + text), this
module:
- Resolves ``#n`` back-references to prior step results.
- Resolves ``const_*`` constants.
- Resolves row labels for ``table_*`` aggregate operations.
- Executes each operator with strict numeric handling.
- Returns a :class:`~finqa_bot.types.ExecutionResult`.

This is intentionally not Python ``exec``; there is no arbitrary code path.
Every operator is a named function with a known signature, which is what
allows us to quantify "groundedness" rigorously: every numeric literal used
must come from the retrieved context, and every aggregate must be over a
resolvable row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from finqa_bot.execution.numbers import normalize_number
from finqa_bot.types import DocumentContext, ExecutionResult, Step


class ExecutionError(RuntimeError):
    """Raised for any deterministic executor failure."""


@dataclass(frozen=True)
class ExecutionContext:
    """Minimal read-only context provided to the executor."""

    document: DocumentContext


def _as_float(v: float | str, step_results: list[float | str]) -> float:
    if isinstance(v, int | float):
        return float(v)
    if not isinstance(v, str):
        raise ExecutionError(f"Unsupported arg type {type(v).__name__}: {v!r}")
    s = v.strip()
    if s.startswith("#"):
        idx = int(s[1:])
        if not (0 <= idx < len(step_results)):
            raise ExecutionError(f"Back-reference {s} out of range; have {len(step_results)} steps.")
        prev = step_results[idx]
        if isinstance(prev, str):
            raise ExecutionError(f"Step {idx} is non-numeric ({prev!r}); cannot chain.")
        return float(prev)
    if s.startswith("const_"):
        val = normalize_number(s)
        if val is None:
            raise ExecutionError(f"Unknown constant: {s}")
        return float(val)
    val = normalize_number(s)
    if val is not None:
        return float(val)
    # The model sometimes emits "row_label: number" (table cell context leaking
    # into the arg).  Try to recover by taking the numeric part after the colon.
    if ": " in s:
        _, _, numeric_part = s.partition(": ")
        val = normalize_number(numeric_part.strip())
        if val is not None:
            return float(val)
    raise ExecutionError(f"Cannot parse numeric arg: {v!r}")


def _find_row_numbers(context: ExecutionContext, row_label: str) -> list[float]:
    """Return numeric cells of the table row whose label matches ``row_label``.

    Matching is case-insensitive substring on ``row[0]``. If multiple rows
    match, the first is used. Header row is skipped.
    """
    label = row_label.strip().lower()
    if not label:
        raise ExecutionError("Empty row label for table aggregate op.")
    for row in context.document.table[1:]:
        if not row:
            continue
        rl = (row[0] or "").strip().lower()
        if label == rl or label in rl or rl in label:
            nums: list[float] = []
            for cell in row[1:]:
                v = normalize_number(cell)
                if v is not None:
                    nums.append(v)
            return nums
    raise ExecutionError(f"Row label not found in table: {row_label!r}")


def _op_add(a: float, b: float) -> float:
    return a + b


def _op_subtract(a: float, b: float) -> float:
    return a - b


def _op_multiply(a: float, b: float) -> float:
    return a * b


def _op_divide(a: float, b: float) -> float:
    if b == 0.0:
        raise ExecutionError("Division by zero.")
    return a / b


def _op_exp(a: float, b: float) -> float:
    return a ** b


def _op_greater(a: float, b: float) -> str:
    return "yes" if a > b else "no"


BINARY_OPS: dict[str, Callable[[float, float], float | str]] = {
    "add": _op_add,
    "subtract": _op_subtract,
    "multiply": _op_multiply,
    "divide": _op_divide,
    "exp": _op_exp,
    "greater": _op_greater,
}


TABLE_AGGREGATES: dict[str, Callable[[list[float]], float]] = {
    "table_sum": lambda xs: float(sum(xs)),
    "table_average": lambda xs: float(sum(xs) / len(xs)) if xs else 0.0,
    "table_max": lambda xs: float(max(xs)),
    "table_min": lambda xs: float(min(xs)),
}


class DSLExecutor:
    """Stateless executor. Create one, call :meth:`run` as many times as needed."""

    def run(self, program: list[Step], document: DocumentContext) -> ExecutionResult:
        """Execute ``program`` against ``document``.

        Returns a :class:`~finqa_bot.types.ExecutionResult`; on failure, ``ok``
        is ``False`` and ``error`` contains a short diagnostic.
        """
        ctx = ExecutionContext(document=document)
        step_results: list[float | str] = []
        try:
            for i, step in enumerate(program):
                if step.op in BINARY_OPS:
                    if len(step.args) != 2:
                        raise ExecutionError(
                            f"Step {i} op {step.op} needs 2 args, got {len(step.args)}"
                        )
                    a = _as_float(step.args[0], step_results)
                    b = _as_float(step.args[1], step_results)
                    res = BINARY_OPS[step.op](a, b)
                    step_results.append(res)
                elif step.op in TABLE_AGGREGATES:
                    if len(step.args) != 1:
                        raise ExecutionError(
                            f"Step {i} op {step.op} needs 1 arg (row label), got {len(step.args)}"
                        )
                    label_arg = step.args[0]
                    if not isinstance(label_arg, str):
                        raise ExecutionError(f"Table aggregate arg must be a row label string, got {label_arg!r}")
                    nums = _find_row_numbers(ctx, label_arg)
                    if not nums:
                        raise ExecutionError(f"No numeric cells in row {label_arg!r}")
                    step_results.append(TABLE_AGGREGATES[step.op](nums))
                else:
                    raise ExecutionError(f"Unknown op: {step.op}")
        except ExecutionError as exc:
            return ExecutionResult(ok=False, value=None, steps=step_results, error=str(exc))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            return ExecutionResult(ok=False, value=None, steps=step_results, error=str(exc))

        if not step_results:
            return ExecutionResult(ok=False, value=None, steps=[], error="Empty program.")
        return ExecutionResult(ok=True, value=step_results[-1], steps=step_results, error=None)
