"""Hybrid dense + BM25 retrieval with reciprocal-rank fusion and reranking.

Flow::

    query
     |- dense:  FAISS IP @ K1
     |- sparse: BM25     @ K1
     --> RRF fuse        @ K2
     --> cross-encoder rerank @ retrieval_k
     --> top rerank_k hits returned to the graph

When ``doc_id`` is passed, both retrievers over-fetch from the global index
and then filter; this keeps the index shared across the whole corpus while
guaranteeing the final hits belong to the asked-about document.
"""

from __future__ import annotations

import numpy as np

from finqa_bot.config import RetrievalConfig
from finqa_bot.logging import get_logger
from finqa_bot.retrieval.embedder import Embedder
from finqa_bot.retrieval.indexer import CorpusIndex, tokenize
from finqa_bot.retrieval.reranker import Reranker
from finqa_bot.types import RetrievalHit

log = get_logger(__name__)


class HybridRetriever:
    """Dense + BM25 + RRF + optional cross-encoder rerank."""

    def __init__(
        self,
        index: CorpusIndex,
        embedder: Embedder,
        cfg: RetrievalConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.cfg = cfg
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        doc_id: str | None = None,
        k: int | None = None,
        rerank_k: int | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve top-k hits. When ``doc_id`` is given, results are filtered to that document."""
        k = k or self.cfg.retrieval_k
        rerank_k = rerank_k or self.cfg.rerank_k
        if not self.index.chunks:
            return []

        allowed: set[int] | None = None
        overfetch = k * 4
        if doc_id is not None:
            allowed = set(self.index.chunks_for_doc(doc_id))
            if not allowed:
                log.warning("doc_id %s not found in index; falling back to global search.", doc_id)
                allowed = None
            else:
                overfetch = max(k * 2, len(allowed))

        dense_hits = self._dense_search(query, k=max(k * 3, overfetch))
        sparse_hits = self._sparse_search(query, k=max(k * 3, overfetch))

        if allowed is not None:
            dense_hits = [(i, s) for i, s in dense_hits if i in allowed]
            sparse_hits = [(i, s) for i, s in sparse_hits if i in allowed]

        fused = self._rrf(
            dense_hits,
            sparse_hits,
            k=self.cfg.rrf_k,
            dense_weight=self.cfg.dense_weight,
            sparse_weight=self.cfg.bm25_weight,
        )
        candidates = fused[:max(k, rerank_k * 2)]
        hits = [self._to_hit(i, score, "rrf") for i, score in candidates]

        if self.reranker is not None and self.reranker.enabled and hits:
            passages = [self.index.chunks[i].text for i, _ in candidates]
            scores = self.reranker.score(query, passages)
            if scores is not None:
                ordered = sorted(zip(hits, scores, strict=False), key=lambda t: t[1], reverse=True)
                hits = [h.model_copy(update={"score": float(s), "source": "reranker"}) for h, s in ordered]

        return hits[:rerank_k]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.index.faiss_index.ntotal == 0:
            return []
        vec = self.embedder.encode_query(query).astype(np.float32)
        vec = np.expand_dims(vec, 0)
        k = min(k, self.index.faiss_index.ntotal)
        scores, idxs = self.index.faiss_index.search(vec, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0], strict=False) if i >= 0]

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.index.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.index.bm25.get_scores(tokens)
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        ordered = sorted(((int(i), float(scores[i])) for i in top_idx), key=lambda t: t[1], reverse=True)
        return ordered

    @staticmethod
    def _rrf(
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(dense, start=1):
            fused[idx] = fused.get(idx, 0.0) + dense_weight / (k + rank)
        for rank, (idx, _score) in enumerate(sparse, start=1):
            fused[idx] = fused.get(idx, 0.0) + sparse_weight / (k + rank)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def _to_hit(self, idx: int, score: float, source: str) -> RetrievalHit:
        chunk = self.index.chunks[idx]
        return RetrievalHit(chunk=chunk, score=score, source=source)
