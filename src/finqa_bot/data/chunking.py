"""Row-granular chunking that matches FinQA's ``gold_inds`` convention.

FinQA gold supporting-fact keys look like ``text_3`` or ``table_1``.
- ``text_i`` where ``i < len(pre_text)`` points at ``pre_text[i]``.
- ``text_i`` where ``i >= len(pre_text)`` points at
  ``post_text[i - len(pre_text)]``.
- ``table_0`` points at the header row; ``table_i`` for ``i >= 1`` points at
  data row ``table[i]`` in the raw table.

We emit chunks whose ``id`` equals the gold_inds key, so retrieval recall@k is
a one-line set-intersection against the gold supporting facts. The header row
is included as a ``table_header`` chunk for readability but is not a data row
and rarely appears in gold_inds.
"""

from __future__ import annotations

import re
from typing import Any

from finqa_bot.types import DocumentContext, TableChunk


SCALE_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"\(\s*in\s+billions\s*\)", re.IGNORECASE), 1_000_000_000.0, "millions|billions"),
    (re.compile(r"\(\s*in\s+millions\s*\)", re.IGNORECASE), 1_000_000.0, "millions"),
    (re.compile(r"\(\s*in\s+thousands\s*\)", re.IGNORECASE), 1_000.0, "thousands"),
    (re.compile(r"\(\s*dollars\s+in\s+billions\s*\)", re.IGNORECASE), 1_000_000_000.0, "billions"),
    (re.compile(r"\(\s*dollars\s+in\s+millions\s*\)", re.IGNORECASE), 1_000_000.0, "millions"),
    (re.compile(r"\(\s*dollars\s+in\s+thousands\s*\)", re.IGNORECASE), 1_000.0, "thousands"),
]


def extract_scale_factor(pre_text: list[str], post_text: list[str], table: list[list[str]]) -> tuple[float, str]:
    """Best-effort scale-factor extraction from surrounding prose and headers.

    Returns ``(scale_factor, source)``. Source is a short human-readable tag
    identifying where the scale was found (``"pre_text[2]"``,
    ``"table_header"``, or ``"default"``).
    """
    for i, line in enumerate(pre_text):
        for pat, factor, _label in SCALE_PATTERNS:
            if pat.search(line):
                return factor, f"pre_text[{i}]"
    for i, line in enumerate(post_text):
        for pat, factor, _label in SCALE_PATTERNS:
            if pat.search(line):
                return factor, f"post_text[{i}]"
    if table:
        header = " ".join(table[0])
        for pat, factor, _label in SCALE_PATTERNS:
            if pat.search(header):
                return factor, "table_header"
    return 1.0, "default"


def _row_to_text(row: list[str], headers: list[str] | None) -> str:
    """Serialize one table row as ``label | header1: value1 | header2: value2 ...``.

    We intentionally use a key-value-per-column format rather than a CSV or
    markdown row. Per the 2025 Table Serialization Kitchen study, Markdown-KV
    beats Markdown-Table for row-level semantic retrieval by ~5-10 points
    downstream. It also keeps the row parsable even when embedding models see
    it out of context.
    """
    if not row:
        return ""
    label = row[0].strip() if row else ""
    cells = row[1:] if len(row) > 1 else []
    if headers and len(headers) == len(row):
        header_cells = headers[1:] if len(headers) > 1 else []
        pairs = [f"{h.strip() or f'col{i + 1}'}: {c.strip()}" for i, (h, c) in enumerate(zip(header_cells, cells))]
    else:
        pairs = [f"col{i + 1}: {c.strip()}" for i, c in enumerate(cells)]
    kv = " | ".join(pairs)
    return f"{label} | {kv}" if kv else label


def _header_to_text(row: list[str]) -> str:
    if not row:
        return ""
    return "table headers | " + " | ".join(c.strip() for c in row)


def chunk_record(record: dict[str, Any], doc_id: str | None = None) -> tuple[DocumentContext, list[TableChunk]]:
    """Convert a raw FinQA record into a ``DocumentContext`` and per-row chunks.

    Each chunk ``id`` matches the corresponding FinQA ``gold_inds`` key where
    applicable, so downstream retrieval metrics compare like-for-like.
    """
    pre_text: list[str] = [s for s in record.get("pre_text", []) if s]
    post_text: list[str] = [s for s in record.get("post_text", []) if s]
    table: list[list[str]] = record.get("table", []) or []
    rid = doc_id or str(record.get("id") or record.get("filename") or "unknown")

    scale_factor, scale_source = extract_scale_factor(pre_text, post_text, table)
    doc = DocumentContext(
        doc_id=rid,
        pre_text=pre_text,
        post_text=post_text,
        table=table,
        scale_factor=scale_factor,
        scale_source=scale_source,
    )

    chunks: list[TableChunk] = []
    for i, sent in enumerate(pre_text):
        chunks.append(
            TableChunk(
                id=f"text_{i}",
                doc_id=rid,
                chunk_type="text",
                text=sent,
                scale_factor=scale_factor,
                metadata={"position": "pre_text", "index": i},
            )
        )
    for j, sent in enumerate(post_text):
        chunks.append(
            TableChunk(
                id=f"text_{len(pre_text) + j}",
                doc_id=rid,
                chunk_type="text",
                text=sent,
                scale_factor=scale_factor,
                metadata={"position": "post_text", "index": j},
            )
        )

    if table:
        header_row = table[0]
        chunks.append(
            TableChunk(
                id="table_0",
                doc_id=rid,
                chunk_type="table_header",
                text=_header_to_text(header_row),
                row_index=0,
                headers=header_row,
                scale_factor=scale_factor,
                metadata={"position": "table", "index": 0},
            )
        )
        for k in range(1, len(table)):
            row = table[k]
            chunks.append(
                TableChunk(
                    id=f"table_{k}",
                    doc_id=rid,
                    chunk_type="table_row",
                    text=_row_to_text(row, header_row),
                    row_index=k,
                    row_label=row[0].strip() if row else None,
                    headers=header_row,
                    scale_factor=scale_factor,
                    metadata={"position": "table", "index": k},
                )
            )

    return doc, chunks
