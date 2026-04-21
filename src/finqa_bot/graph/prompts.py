"""Prompt templates.

All prompts are kept in one place so the prefix is stable across runs. This
matters for vLLM prefix-caching: a stable system + few-shot prefix means the
KV cache hits immediately after the first request, cutting TTFT ~2x.

Structure::

    [system]       # role, task, DSL grammar, output contract
    [few-shots]    # 2-3 canonical FinQA examples
    [context]      # retrieved chunks + scale metadata
    [question]     # the user's question

Only ``[context]`` and ``[question]`` are per-request; everything above is
static, so vLLM prefix-caching can memoize it.
"""

from __future__ import annotations

import re

from finqa_bot.types import AnswerEnvelope, RetrievalHit

SYSTEM_PROMPT = """You are a careful financial analyst who answers SEC filing questions in FinQA DSL.

Context: 1) Tagged text (`text_i`) and tables (`table_i`); 2) Reporting-scale (units/thousands/etc); 3) A question.
Respond with JSON matching this schema:
- `extracted_evidence`: Step 1: Quote exact values and labels from context.
- `program`: list of steps with `op`, `args`, and `source` chunk. Ops: add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_max, table_min.
- `args` for math: plain numbers (1234, 56.7), references "#0"/"#1", or FinQA constants "const_100", "const_1000", "const_1000000", "const_m1". For table_* ops: use one string row label.
- `answer_value`: final number or "yes"/"no". `answer_form`: percent/decimal/ratio/currency/count/boolean. `scale`: reporting-scale.
- `grounded_numbers`: literal numbers used. `confidence`: 0.0-1.0. `rationale`: 1-2 sentences.

Rules:
1. ONLY use numbers verbatim from the context.
2. Provide your best explicitly-grounded answer; do not refuse if the context contains relevant numbers.
3. Constants (const_100) are for literal math. DO NOT use const_100 to convert to percentage format.
4. "#n" is the result of step n (0-index). Step 0 cannot use "#n".
5. Numeric args must be pure numbers (e.g. 1663, not "label: 1663").
6. DECIMALS (Critical): FinQA needs percent answers as decimals (15.1% = 0.151). DO NOT ADD multiply by const_100.
   For % change: step 0: subtract(new, old). step 1: divide(#0, old). answer_form="percent", answer_value=0.151.
7. "portion/ratio/percent of" -> divide(num, den). "change from X to Y" -> subtract(Y, X).
8. Use precise rows rather than table_sum/average if possible.
9. "Difference": Subtraction direction must follow language: "from X to Y" means Y - X.


--- FEW-SHOT EXAMPLES (study these carefully before answering) ---

EXAMPLE 1 -- single subtraction; step 0 uses literal numbers only (no #n)
Context:
[text_0] Operating income for 2007 was $2009 thousand.
[text_1] Operating income for 2008 was $1663 thousand.
Question: What was the change in operating income from 2007 to 2008?
Answer:
{"extracted_evidence": "[text_0] Operating income for 2007 was $2009 thousand. [text_1] Operating income for 2008 was $1663 thousand.", \
"program": [{"op": "subtract", "args": [1663, 2009], "source": "text_0"}], \
"answer_value": -346.0, "answer_form": "currency", "scale": "thousands", \
"grounded_numbers": [1663.0, 2009.0], "confidence": 0.98, \
"rationale": "2008 income minus 2007 income: 1663 - 2009 = -346 thousand."}

EXAMPLE 2 -- percentage change: TWO steps only, answer is a decimal fraction.
DO NOT add a third multiply step. 0.151 means 15.1%.
Context:
[text_3] Net revenue was $94.0 million in 2015 and $108.2 million in 2016.
Question: What was the percentage increase in net revenue from 2015 to 2016?
Answer:
{"extracted_evidence": "[text_3] Net revenue was $94.0 million in 2015 and $108.2 million in 2016.", \
"program": [\
{"op": "subtract", "args": [108.2, 94.0], "source": "text_3"}, \
{"op": "divide",   "args": ["#0", 94.0],  "source": "text_3"}], \
"answer_value": 0.151, "answer_form": "percent", "scale": "units", \
"grounded_numbers": [108.2, 94.0], "confidence": 0.97, \
"rationale": "Percentage change = (108.2 - 94.0) / 94.0 = 0.151, i.e. 15.1%. \
Two steps: subtract then divide. No multiply step."}

EXAMPLE 3 -- stock total return: const_100 is the $100 initial investment basis,
answer is still a decimal fraction (0.3781 = 37.81% return).
Context:
[table_1] The following graph compares a $100 investment in our common stock
on January 1, 2009. Value at December 31, 2014: 137.81.
Question: What was the total return percentage for the five years ended December 2014?
Answer:
{"extracted_evidence": "[table_1] Value at December 31, 2014: 137.81. initial basis 100", \
"program": [\
{"op": "subtract", "args": [137.81, "const_100"], "source": "table_1"}, \
{"op": "divide",   "args": ["#0",   "const_100"], "source": "table_1"}], \
"answer_value": 0.3781, "answer_form": "percent", "scale": "units", \
"grounded_numbers": [137.81], "confidence": 0.97, \
"rationale": "Return = (final_value - initial_basis) / initial_basis = \
(137.81 - 100) / 100 = 0.3781 = 37.81%. const_100 is the $100 initial investment, \
not a percentage converter. Answer is a decimal fraction, not multiplied by 100."}

--- END FEW-SHOT EXAMPLES ---
"""


FEW_SHOT_EXAMPLES: list[tuple[str, AnswerEnvelope]] = [
    # Intentionally defined in code (not prose) so the JSON schema stays
    # authoritative; the prompt builder serializes these verbatim.
]


def format_context(hits: list[RetrievalHit]) -> str:
    """Render retrieved hits as a compact, citation-friendly block."""
    lines: list[str] = []
    for h in hits:
        lines.append(f"[{h.chunk.id}] {h.chunk.text}")
    return "\n".join(lines) if lines else "(no context retrieved)"


def _question_hints(question: str) -> str:
    q = question.lower().strip()
    hints: list[str] = []

    if any(k in q for k in ("portion", "ratio", "percent of", "what percentage", "share")):
        hints.append("Intent=ratio_or_portion: use divide(numerator, denominator), not subtract.")

    if any(k in q for k in ("change", "difference", "increase", "decrease", "grew", "decline")):
        hints.append("Intent=change: from X to Y means subtract(Y, X).")

    if any(k in q for k in ("percentage change", "percent change", "growth rate", "increase rate")):
        hints.append("Intent=percentage_change: subtract(new, old) then divide(#0, old).")

    if any(k in q for k in ("average", "mean")):
        hints.append("Intent=average: use add+divide or table_average over the intended row only.")

    if any(k in q for k in ("total", "sum")):
        hints.append("Intent=total: aggregation ops are allowed when the question explicitly asks for totals.")

    if not hints:
        hints.append("Intent=generic_financial_calc: prefer explicit extracted values and minimal steps.")

    return "\n".join(f"- {h}" for h in hints)


def build_user_message(
    question: str,
    hits: list[RetrievalHit],
    scale_label: str,
    scale_source: str,
) -> str:
    """Render the per-request portion of the prompt."""
    ctx = format_context(hits)
    hints = _question_hints(question)
    return (
        f"Reporting scale (from {scale_source}): {scale_label}.\n"
        f"Question hints:\n{hints}\n"
        f"Context:\n{ctx}\n"
        f"Question: {question}\n"
        "Return the JSON envelope."
    )


ROUTER_SYSTEM_PROMPT = """You classify FinQA questions into one of four types. \
Respond with a single JSON object `{"category": "...", "route": "...", \
"confidence": 0.0-1.0, "reason": "..."}`.

Categories:
- `direct_lookup`: answer is a single number stated verbatim in the filing.
- `single_ratio`: answer requires exactly one ratio / percent / growth rate \
computation between two numbers.
- `multi_step`: answer requires two or more arithmetic operations or a chain \
across table rows.
- `composition`: answer requires combining prose and a table (e.g. plug a \
prose number into a table-derived formula).
- `out_of_scope`: the question is not about the filing or cannot be answered \
with arithmetic over it.

Route rules:
- `direct_lookup` and `single_ratio` -> `specialist`
- `multi_step` and `composition` -> `generalist`
- `out_of_scope` -> `refuse`
"""


ROUTER_USER_TEMPLATE = "Question: {question}\nReturn the JSON classification."
