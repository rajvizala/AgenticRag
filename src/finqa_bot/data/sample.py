"""Load FinQA raw JSON records into typed ``EvalSample`` objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from finqa_bot.data.chunking import chunk_record
from finqa_bot.logging import get_logger
from finqa_bot.types import DocumentContext, EvalSample

log = get_logger(__name__)


def iter_raw_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records out of a FinQA split JSON file.

    The official files are a single JSON array; we load the whole array for
    simplicity because each file fits comfortably in memory (test.json is
    ~2.5 MB).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {p}, got {type(data).__name__}")
    yield from data


def load_samples(
    path: Path | str,
    n: int | None = None,
    offset: int = 0,
    seed: int | None = None,
) -> list[EvalSample]:
    """Load FinQA records and return a list of typed ``EvalSample`` objects.

    When ``seed`` is provided, records are shuffled deterministically before the
    offset/n slice is applied. Default behavior (seed=None) preserves the file
    order so that slices are reproducible against the canonical dataset.
    """
    records = list(iter_raw_records(path))
    if seed is not None:
        import random

        rng = random.Random(seed)
        rng.shuffle(records)
    if offset:
        records = records[offset:]
    if n is not None:
        records = records[:n]

    samples: list[EvalSample] = []
    for r in records:
        try:
            samples.append(record_to_sample(r))
        except Exception as exc:  # noqa: BLE001
            rid = r.get("id", "<unknown>")
            log.warning("Skipping malformed FinQA record %s: %s", rid, exc)
    log.info("Loaded %d samples from %s", len(samples), path)
    return samples


def record_to_sample(record: dict[str, Any]) -> EvalSample:
    """Convert one FinQA raw record to an ``EvalSample``."""
    qa = record.get("qa", {}) or {}
    question = qa.get("question", "").strip()
    if not question:
        raise ValueError("Record missing 'qa.question'")

    rid = str(record.get("id") or record.get("filename") or "unknown")
    doc, _ = chunk_record(record, doc_id=rid)

    gold_inds = list((qa.get("gold_inds") or {}).keys())
    gold_program = qa.get("program") or ""
    gold_answer = qa.get("exe_ans") if "exe_ans" in qa else qa.get("answer")

    return EvalSample(
        id=rid,
        question=question,
        document=doc,
        gold_program=gold_program,
        gold_answer=gold_answer,
        gold_inds=gold_inds,
        raw=record,
    )


def dump_context_summary(doc: DocumentContext) -> str:
    """Render a compact string summary of a document for logging/debugging."""
    return (
        f"doc_id={doc.doc_id} pre={len(doc.pre_text)}s post={len(doc.post_text)}s "
        f"table={len(doc.table)}r scale={doc.scale_factor} src={doc.scale_source}"
    )
