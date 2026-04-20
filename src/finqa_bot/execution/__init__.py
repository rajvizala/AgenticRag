"""FinQA DSL parser, deterministic executor, and number normalization.

Public API:
- :class:`DSLParser`
- :class:`DSLExecutor`
- :func:`normalize_number`
"""

from __future__ import annotations

from finqa_bot.execution.dsl import DSLParser, ProgramParseError
from finqa_bot.execution.executor import DSLExecutor, ExecutionError
from finqa_bot.execution.numbers import (
    extract_numbers_from_text,
    finqa_str_to_num,
    format_number,
    normalize_number,
)

__all__ = [
    "DSLExecutor",
    "DSLParser",
    "ExecutionError",
    "ProgramParseError",
    "extract_numbers_from_text",
    "finqa_str_to_num",
    "format_number",
    "normalize_number",
]
