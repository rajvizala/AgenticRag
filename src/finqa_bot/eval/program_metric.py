"""Program-accuracy metric via symbolic equivalence.

Two FinQA programs are *program-equivalent* if they compute the same value for
every legal input. We approximate this by:

1. Parsing both programs into ``list[Step]``.
2. Rebuilding each as a nested SymPy expression, substituting ``#n``
   back-references recursively and treating ``table_*`` aggregate results
   over the same row label as the same opaque ``Symbol``.
3. Comparing with ``sympy.simplify(a - b) == 0``. For boolean outputs
   (``greater``) we compare structurally.

SymPy handles commutativity (``add(a, b) == add(b, a)``) and distributivity
(``divide(subtract(a, b), c) == subtract(divide(a, c), divide(b, c))``) for
free, which is exactly what "program accuracy" is supposed to forgive.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from finqa_bot.execution.dsl import parse_program
from finqa_bot.execution.numbers import normalize_number
from finqa_bot.types import DSL_OPS, Step

_BINARY: dict[str, Any] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "exp": lambda a, b: a ** b,
}


def _to_expr(steps: list[Step]) -> Any:
    """Build a SymPy expression (or tuple for boolean programs) from steps."""
    if not steps:
        return sp.S.Zero
    step_exprs: list[Any] = []

    def resolve(arg: Any) -> Any:
        if isinstance(arg, int | float):
            return sp.Rational(str(arg)) if float(arg).is_integer() else sp.Float(float(arg))
        if isinstance(arg, str):
            s = arg.strip()
            if s.startswith("#"):
                idx = int(s[1:])
                if not (0 <= idx < len(step_exprs)):
                    raise ValueError(f"Bad back-reference: {s}")
                return step_exprs[idx]
            if s.startswith("const_"):
                v = normalize_number(s)
                if v is None:
                    return sp.Symbol(s)
                return sp.Float(v)
            v = normalize_number(s)
            if v is not None:
                return sp.Float(v)
            return sp.Symbol(s)
        raise TypeError(f"Unsupported arg type {type(arg).__name__}")

    for s in steps:
        if s.op in _BINARY:
            if len(s.args) != 2:
                raise ValueError(f"Op {s.op} requires 2 args; got {len(s.args)}")
            a, b = (resolve(x) for x in s.args)
            step_exprs.append(_BINARY[s.op](a, b))
        elif s.op == "greater":
            if len(s.args) != 2:
                raise ValueError(f"greater requires 2 args; got {len(s.args)}")
            a, b = (resolve(x) for x in s.args)
            step_exprs.append(sp.Gt(a, b))
        elif s.op.startswith("table_"):
            if len(s.args) != 1:
                raise ValueError(f"{s.op} requires 1 arg; got {len(s.args)}")
            label = str(s.args[0]).strip().lower().replace(" ", "_")
            sym = sp.Symbol(f"{s.op}::{label}")
            step_exprs.append(sym)
        else:
            raise ValueError(f"Unknown op: {s.op}")
    return step_exprs[-1]


def program_match_symbolic(pred: str | list[Step], gold: str | list[Step]) -> bool:
    """Return True iff ``pred`` and ``gold`` are symbolically equivalent."""
    try:
        p_steps = pred if isinstance(pred, list) else parse_program(pred)
        g_steps = gold if isinstance(gold, list) else parse_program(gold)
    except Exception:
        return False

    if not _validate_ops(p_steps) or not _validate_ops(g_steps):
        return False
    if not p_steps or not g_steps:
        return False

    try:
        a = _to_expr(p_steps)
        b = _to_expr(g_steps)
    except Exception:
        return False

    if isinstance(a, sp.Basic) and isinstance(b, sp.Basic) and a.is_Boolean and b.is_Boolean:
        return bool(sp.simplify(sp.Equivalent(a, b)))
    try:
        diff = sp.simplify(sp.expand(a - b))
        return diff == 0
    except Exception:
        return False


def _validate_ops(steps: list[Step]) -> bool:
    return all(s.op in DSL_OPS for s in steps)
