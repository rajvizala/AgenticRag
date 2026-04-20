"""Cross-encoder reranker.

The plan specifies ``Qwen/Qwen3-Reranker-0.6B`` as the preferred reranker but
it requires custom yes/no-token scoring because it is a causal-LM, not a
traditional cross-encoder. To keep the default path robust on Colab, we:

1. Try the configured model via ``sentence_transformers.CrossEncoder`` with
   ``trust_remote_code=True`` (works for BGE-reranker-v2-m3 and for recent
   versions of sentence-transformers that support Qwen3-Reranker).
2. Fall back to ``BAAI/bge-reranker-v2-m3`` if the configured model fails.
3. Fall back to ``cross-encoder/ms-marco-MiniLM-L-6-v2`` if that also fails.
4. As a last resort, return ``None`` from :meth:`Reranker.score` so the
   pipeline continues without reranking.
"""

from __future__ import annotations

from collections.abc import Sequence

from finqa_bot.logging import get_logger

log = get_logger(__name__)


DEFAULT_FALLBACKS = (
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


class Reranker:
    """Thin wrapper around ``CrossEncoder`` with a bypass mode."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
        trust_remote_code: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.model = None
        self.model_name: str | None = None

        candidates = (model_name, *tuple(m for m in DEFAULT_FALLBACKS if m != model_name))
        last_err: Exception | None = None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            log.warning("sentence-transformers CrossEncoder unavailable: %s", exc)
            return

        for cand in candidates:
            try:
                log.info("Loading reranker %s on %s ...", cand, device)
                self.model = CrossEncoder(cand, device=device, trust_remote_code=trust_remote_code)
                self.model_name = cand
                if cand != model_name:
                    log.warning("Primary reranker %s failed; using fallback %s.", model_name, cand)
                return
            except Exception as exc:
                last_err = exc
                log.warning("Failed to load reranker candidate %s on %s: %s", cand, device, exc)
        log.warning(
            "All reranker candidates failed; reranking disabled. Last error: %s",
            last_err,
        )

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def score(self, query: str, passages: Sequence[str]) -> list[float] | None:
        """Return relevance scores for ``(query, passage_i)`` pairs.

        Returns ``None`` if no reranker could be loaded, so callers fall back
        to the pre-rerank ranking.
        """
        if self.model is None or not passages:
            return None
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        return [float(s) for s in scores]
