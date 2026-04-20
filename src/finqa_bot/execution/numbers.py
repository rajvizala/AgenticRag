"""Number normalization for FinQA.

Two flavors:
- :func:`finqa_str_to_num` - byte-for-byte replica of ``code/evaluate/str_to_num``
  from the official `czyssrs/FinQA` repo. Used exclusively by
  :mod:`finqa_bot.eval.finqa_metric` so that our execution-accuracy numbers are
  directly comparable to every published FinQA result. Known quirk: this
  function strips everything after ``(`` which silently treats
  accounting-style parentheses ``(1,234)`` as ``1234`` (not ``-1234``).
- :func:`normalize_number` - semantic number parsing for production use.
  Recognizes accounting-style negatives, currency symbols, scale suffixes
  (``million``, ``billion``, ``k``, ``M``, ``B``), percent, and FinQA
  ``const_*`` constants. Used everywhere except the official metric.

Keeping the two separate makes the "we match the canonical metric exactly"
guarantee auditable while still shipping correct behavior everywhere else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

SCALE_SUFFIXES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(?i)\bbillions?\b"), 1_000_000_000.0),
    (re.compile(r"(?i)\bmillions?\b"), 1_000_000.0),
    (re.compile(r"(?i)\bthousands?\b"), 1_000.0),
    (re.compile(r"(?i)\bbn\b"), 1_000_000_000.0),
    (re.compile(r"(?i)\bmn\b"), 1_000_000.0),
]

NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    -?\(?\$?                        # optional sign, open-paren, currency
    \d{1,3}(?:,\d{3})+(?:\.\d+)?    # grouped thousands
    |
    -?\(?\$?\d+(?:\.\d+)?           # plain number
    \)?                             # optional close-paren (accounting negative)
    """,
    re.VERBOSE,
)


def finqa_str_to_num(s: str | int | float) -> float | str:
    """Exact replica of the official ``str_to_num`` used by ``evaluate.py``.

    Behavior preserved even where it is arguably wrong; see module docstring.
    Returns ``"n/a"`` on parse failure, matching the upstream sentinel.
    """
    if isinstance(s, int | float):
        return float(s)
    text = s.strip()

    if text.startswith("const_"):
        body = text[len("const_"):]
        if body.startswith("m"):
            try:
                return -float(body[1:])
            except ValueError:
                return "n/a"
        try:
            return float(body)
        except ValueError:
            return "n/a"

    if "(" in text:
        text = text.split("(", 1)[0].strip()

    text = text.replace("$", "").replace(",", "").strip()
    percent = False
    if text.endswith("%"):
        percent = True
        text = text[:-1].strip()
    try:
        v = float(text)
    except ValueError:
        return "n/a"
    return v / 100.0 if percent else v


def normalize_number(token: str | int | float) -> float | None:
    """Parse a single number token into a float. Semantically correct.

    Handles accounting negatives ``(1,234)`` -> ``-1234``, scale suffixes
    ``" million"`` / ``" billion"`` / ``"k"``, currency symbols, percent,
    trailing sign flips, and ``const_*`` constants from FinQA. Returns
    ``None`` on failure.
    """
    if isinstance(token, int | float):
        return float(token)
    text = str(token).strip()
    if not text:
        return None

    if text.startswith("const_"):
        body = text[len("const_"):]
        if body.startswith("m"):
            try:
                return -float(body[1:])
            except ValueError:
                return None
        try:
            return float(body)
        except ValueError:
            return None

    sign = 1.0
    if text.startswith("(") and text.endswith(")"):
        sign *= -1.0
        text = text[1:-1].strip()

    scale = 1.0
    for pat, factor in SCALE_SUFFIXES:
        if pat.search(text):
            scale *= factor
            text = pat.sub("", text).strip()
            break

    if text.endswith("%"):
        try:
            return sign * float(text[:-1].replace(",", "").replace("$", "")) / 100.0 * scale
        except ValueError:
            return None

    cleaned = text.replace(",", "").replace("$", "").strip()
    if cleaned in {"", "-", "--", "n/a", "na", "none", "null"}:
        return None
    if cleaned.startswith("-"):
        sign *= -1.0
        cleaned = cleaned[1:]
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    try:
        return sign * float(cleaned) * scale
    except ValueError:
        return None


def extract_numbers_from_text(text: str) -> list[float]:
    """Pull every number out of a blob of text (tables, sentences, JSON blobs)."""
    seen: list[float] = []
    for match in NUMBER_RE.findall(text):
        val = normalize_number(match)
        if val is not None:
            seen.append(val)
    return seen


def extract_numbers_from_chunks(texts: Iterable[str]) -> list[float]:
    """Batch over multiple chunks of text, deduplicating to 4-decimal precision."""
    seen: dict[float, None] = {}
    for t in texts:
        for v in extract_numbers_from_text(t):
            seen[round(v, 4)] = None
    return list(seen.keys())


def format_number(value: float, form: str = "decimal") -> str:
    """Render a numeric answer in the canonical FinQA output form.

    ``form`` is one of ``percent``, ``decimal``, ``ratio``, ``currency``,
    ``count``, ``boolean``. ``boolean`` expects the caller to have already
    produced ``"yes"`` / ``"no"`` and is passed through.
    """
    if form == "boolean":
        return str(value)
    if form == "percent":
        return f"{value * 100:.2f}%"
    if form == "currency":
        return f"${value:,.2f}"
    if form == "count":
        return f"{round(value)}"
    return f"{value:.6g}"
