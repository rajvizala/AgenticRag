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

SYSTEM_PROMPT = """You are a careful financial analyst who answers questions \
about a single SEC filing by emitting a structured program in the FinQA DSL.

You will receive:
1. Excerpts from the filing (pre-table prose, table rows, post-table prose). Each \
excerpt is tagged with an ID of the form `text_i` or `table_i` so you can cite it.
2. A reporting-scale hint derived from the table header (units, thousands, \
millions, or billions). Numbers in the table are in that scale unless \
stated otherwise.
3. A question.

You must respond with a single JSON object matching this schema exactly:
- `program`: a list of step objects. Each step has `op`, `args`, and `source` \
(the chunk id the numbers came from). The ten operators are:
  add, subtract, multiply, divide, exp, greater,
  table_sum, table_average, table_max, table_min.
- `args` for arithmetic ops are plain numeric literals (e.g. 1234, 56.7) or \
step back-references "#0"/"#1"/... or FinQA constants "const_100", \
"const_1000", "const_1000000", "const_m1".
- `args` for table_* ops contain exactly one string: the row label.
- `answer_value`: the final numeric answer or "yes"/"no" for comparisons.
- `answer_form`: one of percent, decimal, ratio, currency, count, boolean.
- `scale`: one of units, thousands, millions, billions.
- `grounded_numbers`: every numeric literal used in the program args.
- `confidence`: 0.0 to 1.0.
- `rationale`: one or two concise sentences.

Hard rules:
1. Only use numbers verbatim from the supplied context. Do not fabricate figures.
2. Always return your best grounded program attempt from the retrieved context. \
Do not refuse and do not emit an empty program unless the context block is truly empty.
3. Use the FinQA symbolic constants only when the literal value they represent \
actually appears in the computation: const_100 = the number 100, const_1000 = 1000, \
const_1000000 = 1000000, const_m1 = -1. For example, const_100 is correct when the \
question involves a $100 initial investment basis, NOT as a percentage converter.
4. BACK-REFERENCE RULE: "#n" means the numeric RESULT of step n (0-indexed). \
Step 0 has no prior results, so its args MUST be plain numbers, never "#n". \
"#0" is only valid in step 1 or later. "#1" is only valid in step 2 or later.
5. NUMERIC ARG FORMAT: args must be plain numbers such as 1663, 94.0, or 3.14. \
Never include a row label, table key, or any other prefix. \
If a context chunk shows "some label: 1663", the arg is just 1663.
6. DECIMAL FRACTION CONVENTION (critical): FinQA stores ALL percentage-change, \
growth-rate, and ratio answers as plain decimal fractions. A 15.1% increase is \
stored as 0.151, not 15.1. Therefore for "percentage change / growth rate / percent of": \
  step 0: subtract(new_value, old_value) \
  step 1: divide(#0, old_value) \
  answer_value = 0.151, answer_form = "percent" \
NEVER add multiply(#N, const_100) to a percentage-change program. That produces 15.1 \
instead of 0.151, which is the WRONG answer in the FinQA evaluation.
7. OPERATOR POLICY:
    - "portion", "ratio", "percent of", "what percentage" -> use divide(numerator, denominator).
    - "change from X to Y", "increase/decrease" -> use subtract(Y, X). If percentage change, then divide(#0, X).
    - "average/mean" -> use add and divide or table_average.
8. Prefer explicit row-cell values over broad table_sum/table_average when the question references specific years/segments.
9. Subtraction direction must follow language direction: from X to Y means Y - X.

--- FEW-SHOT EXAMPLES (study these carefully before answering) ---

EXAMPLE 1 -- single subtraction; step 0 uses literal numbers only (no #n)
Context:
[text_0] Operating income for 2007 was $2009 thousand.
[text_1] Operating income for 2008 was $1663 thousand.
Question: What was the change in operating income from 2007 to 2008?
Answer:
{"program": [{"op": "subtract", "args": [1663, 2009], "source": "text_0"}], \
"answer_value": -346.0, "answer_form": "currency", "scale": "thousands", \
"grounded_numbers": [1663.0, 2009.0], "confidence": 0.98, \
"rationale": "2008 income minus 2007 income: 1663 - 2009 = -346 thousand."}

EXAMPLE 2 -- percentage change: TWO steps only, answer is a decimal fraction.
DO NOT add a third multiply step. 0.151 means 15.1%.
Context:
[text_3] Net revenue was $94.0 million in 2015 and $108.2 million in 2016.
Question: What was the percentage increase in net revenue from 2015 to 2016?
Answer:
{"program": [\
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
{"program": [\
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
