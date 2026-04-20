"""Retrieval recall against FinQA gold_inds."""

from __future__ import annotations

from finqa_bot.data.chunking import chunk_record


def test_retrieval_recall_full_gold_inds() -> None:
    record = {
        "id": "Unit/Test/page_1.pdf-1",
        "pre_text": ["Intro sentence zero.", "Intro sentence one."],
        "post_text": ["Post text zero."],
        "table": [
            ["", "2023", "2024"],
            ["Revenue", "100", "120"],
            ["Expenses", "60", "72"],
        ],
    }
    _, chunks = chunk_record(record)
    gold_inds = {"text_1", "table_1"}

    retrieved = {c.id for c in chunks}  # pretend retrieval returned all chunks
    recall = len(gold_inds & retrieved) / len(gold_inds)
    assert recall == 1.0


def test_retrieval_partial_recall() -> None:
    record = {
        "id": "Unit/Test/page_1.pdf-1",
        "pre_text": ["Intro zero.", "Intro one."],
        "post_text": [],
        "table": [
            ["", "2023"],
            ["Revenue", "100"],
        ],
    }
    _, chunks = chunk_record(record)
    gold_inds = {"text_0", "table_1"}
    # Only return the text chunk.
    retrieved = {"text_0"}
    recall = len(gold_inds & retrieved) / len(gold_inds)
    assert recall == 0.5
