"""Shared fixtures for the FinQA test suite."""

from __future__ import annotations

import pytest

from finqa_bot.types import DocumentContext, Step, TableChunk


@pytest.fixture
def sample_doc() -> DocumentContext:
    """A minimal two-row document used by unit tests."""
    return DocumentContext(
        doc_id="UnitTest/2024/page_1.pdf-1",
        pre_text=["Revenues and costs are in millions of U.S. dollars."],
        post_text=["The effective tax rate was 21% in 2024."],
        table=[
            ["", "2023", "2024"],
            ["Revenue", "100", "120"],
            ["Cost of revenue", "60", "72"],
        ],
        scale_factor=1_000_000.0,
        scale_source="pre_text",
    )


@pytest.fixture
def sample_chunks() -> list[TableChunk]:
    """Row-granular chunks matching FinQA's gold_inds convention."""
    return [
        TableChunk(
            id="table_1",
            doc_id="UnitTest/2024/page_1.pdf-1",
            chunk_type="table_row",
            text="| row | 2023 | 2024 |\n| --- | --- | --- |\n| Revenue | 100 | 120 |",
            row_index=1,
            row_label="Revenue",
            headers=["", "2023", "2024"],
            scale_factor=1_000_000.0,
        ),
        TableChunk(
            id="table_2",
            doc_id="UnitTest/2024/page_1.pdf-1",
            chunk_type="table_row",
            text="| row | 2023 | 2024 |\n| --- | --- | --- |\n| Cost of revenue | 60 | 72 |",
            row_index=2,
            row_label="Cost of revenue",
            headers=["", "2023", "2024"],
            scale_factor=1_000_000.0,
        ),
        TableChunk(
            id="text_0",
            doc_id="UnitTest/2024/page_1.pdf-1",
            chunk_type="text",
            text="Revenues and costs are in millions of U.S. dollars.",
        ),
    ]


@pytest.fixture
def simple_program() -> list[Step]:
    """add 100 to 120 = 220."""
    return [Step(op="add", args=[100.0, 120.0], source="table_1")]
