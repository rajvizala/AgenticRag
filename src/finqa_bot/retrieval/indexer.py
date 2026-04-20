"""Corpus indexer.

Builds and persists a global retrieval index over the FinQA corpus. Each
FinQA record becomes one "document" whose chunks are indexed with ``doc_id``
metadata so queries can filter to a single 10-K filing while the underlying
index is shared across the whole corpus.

Index artifacts (all under ``<data_dir>/indices/<split>/``):
- ``chunks.json``: serialized :class:`TableChunk` records (the source of truth).
- ``embeddings.npy``: ``(n_chunks, dim)`` float32, L2-normalized.
- ``faiss.index``: FAISS ``IndexFlatIP`` over the embeddings (IP on normalized
  vectors == cosine similarity).
- ``bm25.pkl``: pickled ``rank_bm25.BM25Okapi`` + the tokenized corpus.
- ``meta.json``: embedder name, dim, chunk count, timestamp.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finqa_bot.config import GpuConfig
from finqa_bot.data.chunking import chunk_record
from finqa_bot.data.downloader import SPLITS, download_finqa
from finqa_bot.data.sample import iter_raw_records
from finqa_bot.logging import get_logger
from finqa_bot.retrieval.embedder import Embedder
from finqa_bot.types import DocumentContext, TableChunk

log = get_logger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Cheap, deterministic tokenizer for BM25. Lowercases + alphanumeric split."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class CorpusIndex:
    """Materialized retrieval index."""

    chunks: list[TableChunk]
    embeddings: np.ndarray
    faiss_index: object  # faiss.Index
    bm25: object         # rank_bm25.BM25Okapi
    tokenized_corpus: list[list[str]]
    embedder_name: str
    dim: int
    path: Path
    documents: dict[str, DocumentContext]

    def chunks_for_doc(self, doc_id: str) -> list[int]:
        return [i for i, c in enumerate(self.chunks) if c.doc_id == doc_id]

    def chunk_by_id(self, chunk_id: str, doc_id: str | None = None) -> int | None:
        for i, c in enumerate(self.chunks):
            if c.id == chunk_id and (doc_id is None or c.doc_id == doc_id):
                return i
        return None

    def document(self, doc_id: str) -> DocumentContext | None:
        return self.documents.get(doc_id)

    def doc_ids(self) -> list[str]:
        return sorted(self.documents.keys())


def _index_root(data_dir: Path | str, split: str) -> Path:
    return Path(data_dir) / "indices" / split


def build_index(
    data_dir: Path | str,
    cfg: GpuConfig,
    split: str = "dev",
    rebuild: bool = False,
) -> CorpusIndex:
    """Build (or load) the retrieval index for a given FinQA split.

    Idempotent: if ``rebuild`` is False and the index artifacts exist under
    ``data_dir/indices/<split>``, the index is loaded from disk.
    """
    data_dir = Path(data_dir)
    index_dir = _index_root(data_dir, split)
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = index_dir / "chunks.json"
    docs_path = index_dir / "documents.json"
    emb_path = index_dir / "embeddings.npy"
    faiss_path = index_dir / "faiss.index"
    bm25_path = index_dir / "bm25.pkl"
    meta_path = index_dir / "meta.json"

    artifacts_exist = all(
        p.exists() for p in (chunks_path, docs_path, emb_path, faiss_path, bm25_path, meta_path)
    )
    if artifacts_exist and not rebuild:
        log.info("Index already present at %s (use --rebuild-index to regenerate).", index_dir)
        return load_index(index_dir)

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    split_path = data_dir / "raw" / "finqa" / f"{split}.json"
    if not split_path.exists():
        download_finqa(data_dir, split=split)

    log.info("Chunking FinQA %s ...", split)
    chunks: list[TableChunk] = []
    documents: dict[str, DocumentContext] = {}
    n_records = 0
    for record in iter_raw_records(split_path):
        doc, chunks_r = chunk_record(record)
        chunks.extend(chunks_r)
        documents[doc.doc_id] = doc
        n_records += 1
    log.info("Produced %d chunks from %d records.", len(chunks), n_records)

    log.info("Loading embedder %s ...", cfg.embedding.model)
    embedder = Embedder(
        model_name=cfg.embedding.model,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
    )
    texts = [c.text for c in chunks]
    log.info("Encoding %d chunks ...", len(texts))
    embeddings = embedder.encode_documents(texts)

    log.info("Building FAISS IndexFlatIP ...")
    import faiss

    dim = int(embeddings.shape[1]) if embeddings.size else embedder.dim
    index = faiss.IndexFlatIP(dim)
    if embeddings.size:
        index.add(embeddings.astype(np.float32))

    log.info("Building BM25 ...")
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None

    log.info("Persisting index to %s ...", index_dir)
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f)
    with docs_path.open("w", encoding="utf-8") as f:
        json.dump({did: d.model_dump() for did, d in documents.items()}, f)
    np.save(emb_path, embeddings.astype(np.float32))
    faiss.write_index(index, str(faiss_path))
    with bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedder": embedder.model_name,
                "dim": dim,
                "n_chunks": len(chunks),
                "n_records": n_records,
                "split": split,
                "built_at": int(time.time()),
            },
            f,
            indent=2,
        )

    return CorpusIndex(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=index,
        bm25=bm25,
        tokenized_corpus=tokenized,
        embedder_name=embedder.model_name,
        dim=dim,
        path=index_dir,
        documents=documents,
    )


def load_index(index_dir: Path | str) -> CorpusIndex:
    """Load a pre-built index from disk."""
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"Index dir missing: {index_dir}")

    import faiss

    with (index_dir / "chunks.json").open("r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunks = [TableChunk(**c) for c in raw_chunks]

    documents: dict[str, DocumentContext] = {}
    docs_path = index_dir / "documents.json"
    if docs_path.exists():
        with docs_path.open("r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        documents = {did: DocumentContext(**d) for did, d in raw_docs.items()}

    embeddings = np.load(index_dir / "embeddings.npy")
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    with (index_dir / "bm25.pkl").open("rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]
    tokenized = payload["tokenized"]

    with (index_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return CorpusIndex(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=faiss_index,
        bm25=bm25,
        tokenized_corpus=tokenized,
        embedder_name=str(meta.get("embedder", "")),
        dim=int(meta.get("dim", embeddings.shape[1] if embeddings.size else 0)),
        path=index_dir,
        documents=documents,
    )


def chunks_for_samples(sample_ids: Sequence[str], index: CorpusIndex) -> dict[str, list[int]]:
    """Helper: map each sample id to its chunk indices in the index."""
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(index.chunks):
        by_doc.setdefault(c.doc_id, []).append(i)
    return {sid: by_doc.get(sid, []) for sid in sample_ids}
