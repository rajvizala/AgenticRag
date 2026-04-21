"""Dense embedding wrapper.

Target model: ``Qwen/Qwen3-Embedding-0.6B`` (MTEB multilingual top-tier, 32K
context, instruction-aware). Falls back to ``BAAI/bge-small-en-v1.5`` and
then ``sentence-transformers/all-MiniLM-L6-v2`` if loading fails, so the repo
clones cleanly on machines where ``trust_remote_code=True`` is not enabled
or the Qwen3-Embedding weights are unavailable.

We keep the interface minimal:
- :meth:`Embedder.encode_documents`
- :meth:`Embedder.encode_query`

Qwen3-Embedding is instruction-aware; queries get a short instruction prefix
(the exact wording recommended by the Qwen3-Embedding model card) while
documents are encoded plain. Non-Qwen backbones ignore the instruction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from finqa_bot.logging import get_logger

log = get_logger(__name__)


DEFAULT_FALLBACKS = (
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def _qwen_query_instruction(task: str = "Given a financial filing question, retrieve the sentence or table row that contains the answer") -> str:
    return f"Instruct: {task}\nQuery: "


class Embedder:
    """Wraps sentence-transformers with graceful fallback."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.normalize = normalize
        self._is_qwen = "qwen3-embedding" in model_name.lower()

        tried: list[tuple[str, str]] = []
        candidates = (model_name, *tuple(m for m in DEFAULT_FALLBACKS if m != model_name))
        last_err: Exception | None = None
        for cand in candidates:
            try:
                log.info("Loading embedder %s on %s ...", cand, device)
                self.model = SentenceTransformer(cand, device=device, trust_remote_code=trust_remote_code)
                self.model_name = cand
                if cand != model_name:
                    log.warning(
                        "Primary embedder %s failed; using fallback %s.",
                        model_name,
                        cand,
                    )
                if hasattr(self.model, "get_embedding_dimension"):
                    self.dim = int(self.model.get_embedding_dimension() or 0)
                else:
                    self.dim = int(self.model.get_sentence_embedding_dimension() or 0)
                return
            except Exception as exc:
                tried.append((cand, str(exc)))
                last_err = exc
        msg = "; ".join(f"{c}: {e}" for c, e in tried)
        raise RuntimeError(f"Failed to load any embedder candidate ({msg})") from last_err

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of documents. Returns an ``(n, dim)`` float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Applies Qwen3-Embedding instruction prefix when appropriate."""
        text = _qwen_query_instruction() + query if self._is_qwen else query
        vec = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vec, dtype=np.float32)[0]
