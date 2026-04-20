"""Evaluation harness: FinQA metrics, program accuracy, ablations, benchmarks."""

from __future__ import annotations

from finqa_bot.eval.finqa_metric import exe_equal, official_str_to_num
from finqa_bot.eval.harness import run_eval
from finqa_bot.eval.program_metric import program_match_symbolic
from finqa_bot.eval.slices import load_slice

__all__ = [
    "exe_equal",
    "load_slice",
    "official_str_to_num",
    "program_match_symbolic",
    "run_eval",
]
