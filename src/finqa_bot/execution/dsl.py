"""FinQA DSL parser.

The FinQA DSL from Chen et al. 2021 is a flat sequence of operations joined by
commas, e.g.::

    divide(9413, 20.01), divide(8249, 9.48), subtract(#0, #1)

Supported ops: the 10 listed in :data:`finqa_bot.types.DSL_OPS`. Args are
either numeric literals, ``#n`` step back-references, or ``const_*`` constants
(including ``const_m1`` for -1).

We provide two-way conversion between this flat string and our typed
:class:`~finqa_bot.types.Step` list. The typed form is what the generator
emits (grammar-constrained); the flat form is what the canonical
``evaluate.py`` expects.
"""

from __future__ import annotations

import re

from finqa_bot.types import DSL_OPS, Step


class ProgramParseError(ValueError):
    """Raised when a FinQA program string cannot be parsed."""


_PROG_RE = re.compile(r"(?P<op>\w+)\s*\(\s*(?P<args>[^()]*)\s*\)")


def parse_program(text: str) -> list[Step]:
    """Parse a flat-form FinQA program string into a list of ``Step``.

    Unknown ops are rejected with :class:`ProgramParseError`. The ``EOF``
    sentinel found at the end of some annotation files is silently stripped.
    """
    if not isinstance(text, str):
        raise ProgramParseError(f"Expected str, got {type(text).__name__}")
    cleaned = text.strip().removesuffix(", EOF").removesuffix(",EOF").removesuffix("EOF").strip(", ")
    if not cleaned:
        return []

    ops = _PROG_RE.findall(cleaned)
    if not ops:
        raise ProgramParseError(f"No ops parsed from: {text!r}")

    steps: list[Step] = []
    for op, arg_blob in ops:
        if op not in DSL_OPS:
            raise ProgramParseError(f"Unknown op '{op}' in program: {text!r}")
        args = _parse_args(arg_blob)
        steps.append(Step(op=op, args=args, source=""))
    return steps


def _parse_args(blob: str) -> list[float | str]:
    """Parse the comma-separated arg list for a single op."""
    parts = [p.strip() for p in blob.split(",") if p.strip()]
    out: list[float | str] = []
    for part in parts:
        if part.startswith("#") or part.startswith("const_"):
            out.append(part)
            continue
        try:
            out.append(float(part))
        except ValueError:
            out.append(part)
    return out


def dump_program(steps: list[Step]) -> str:
    """Serialize a list of ``Step`` back into the flat FinQA DSL string form."""
    parts: list[str] = []
    for s in steps:
        arg_strs = [_arg_to_str(a) for a in s.args]
        parts.append(f"{s.op}({', '.join(arg_strs)})")
    return ", ".join(parts)


def _arg_to_str(arg: float | str) -> str:
    if isinstance(arg, str):
        return arg
    if float(arg).is_integer():
        return str(int(arg))
    return f"{arg:g}"


class DSLParser:
    """Thin object wrapper around :func:`parse_program` and :func:`dump_program`."""

    def parse(self, text: str) -> list[Step]:
        return parse_program(text)

    def dump(self, steps: list[Step]) -> str:
        return dump_program(steps)
