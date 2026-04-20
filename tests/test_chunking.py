"""Tests for row-granular chunking vs FinQA gold_inds convention."""

from __future__ import annotations

from finqa_bot.data.chunking import chunk_record


def test_chunk_ids_match_finqa_gold_inds() -> None:
    record = {
        "id": "ExampleCo/2024/page_1.pdf-1",
        "pre_text": [
            "This is paragraph zero.",
            "This is paragraph one.",
        ],
        "post_text": [
            "Footnote one.",
        ],
        "table": [
            ["", "2023", "2024"],
            ["Revenue", "100", "120"],
            ["Expenses", "60", "72"],
        ],
        "qa": {
            "question": "what was the change in revenue?",
            "answer": "20",
            "program": "subtract(120, 100)",
            "gold_inds": {"table_1": "Revenue: ...", "text_0": "..."},
        },
    }
    doc, chunks = chunk_record(record)
    ids = {c.id for c in chunks}

    assert "text_0" in ids
    assert "text_1" in ids
    assert "text_2" in ids  # first post-text follows pre_text
    assert "table_0" in ids  # header
    assert "table_1" in ids
    assert "table_2" in ids
    assert doc.doc_id == record["id"]
    assert doc.scale_factor == 1.0


def test_chunking_detects_millions_scale() -> None:
    record = {
        "id": "ExampleCo/2024/page_1.pdf-1",
        "pre_text": ["Amounts are ( in millions )."],
        "post_text": [],
        "table": [["", "2023"], ["Revenue", "100"]],
    }
    doc, chunks = chunk_record(record)
    assert doc.scale_factor == 1_000_000.0
    assert all(c.scale_factor == 1_000_000.0 for c in chunks)


def test_chunk_row_text_uses_markdown_kv() -> None:
    record = {
        "id": "ExampleCo/2024/page_1.pdf-1",
        "pre_text": [],
        "post_text": [],
        "table": [["", "2023", "2024"], ["Revenue", "100", "120"]],
    }
    _, chunks = chunk_record(record)
    row = next(c for c in chunks if c.id == "table_1")
    assert "Revenue" in row.text
    assert "2023: 100" in row.text
    assert "2024: 120" in row.text
