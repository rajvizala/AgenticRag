"""Assemble the full LangGraph ``StateGraph``.

The graph is straight-line with a single conditional edge at the end:

    ensure_document -> route -> retrieve -> extract_scale -> generate
    -> execute -> verify -> self_consistency -> {format, refuse} -> END

The checkpointer defaults to an in-memory saver; callers asking for
persistence should pass ``use_sqlite_checkpointer=True`` to
:func:`build_graph`, which switches to ``SqliteSaver`` backed by
``settings.finqa_checkpoint_db``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from finqa_bot.config import GpuConfig, Settings
from finqa_bot.graph.nodes import (
    PipelineDeps,
    node_decide,
    node_ensure_document,
    node_execute,
    node_extract_scale,
    node_format,
    node_generate,
    node_refuse,
    node_retrieve,
    node_route,
    node_self_consistency,
    node_verify,
)
from finqa_bot.graph.state import initial_state
from finqa_bot.logging import get_logger
from finqa_bot.retrieval.embedder import Embedder
from finqa_bot.retrieval.hybrid import HybridRetriever
from finqa_bot.retrieval.indexer import CorpusIndex, load_index
from finqa_bot.retrieval.reranker import Reranker
from finqa_bot.retrieval.scale_extractor import ScaleExtractor
from finqa_bot.types import GraphState

log = get_logger(__name__)


@dataclass
class BuildOptions:
    """Knobs for ``build_graph``."""

    index: CorpusIndex | None = None
    retriever: HybridRetriever | None = None
    embedder: Embedder | None = None
    reranker: Reranker | None = None
    feature_flags: dict[str, bool] | None = None
    use_sqlite_checkpointer: bool = False
    enable_retrieval: bool = True


class GraphRunner:
    """Convenience bundle so callers get the compiled graph plus its deps.

    The overloaded ``ainvoke`` / ``invoke`` methods accept either a pre-built
    :class:`GraphState` (low-level) or ``question=..., doc_id=...`` keyword
    arguments (high-level), making the UI and CLI call sites terse without
    losing access to raw-state use-cases in tests and evaluation.
    """

    def __init__(self, graph: Any, deps: PipelineDeps) -> None:
        self.graph = graph
        self.deps = deps

    @staticmethod
    def _coerce_state(
        state_or_question: GraphState | str | None,
        question: str | None,
        doc_id: str | None,
    ) -> GraphState:
        if isinstance(state_or_question, dict):
            return state_or_question  # type: ignore[return-value]
        q = state_or_question if isinstance(state_or_question, str) else question
        if not q:
            raise ValueError("GraphRunner.ainvoke requires either a state or a question.")
        return initial_state(q, doc_id=doc_id)

    async def ainvoke(
        self,
        state: GraphState | str | None = None,
        *,
        question: str | None = None,
        doc_id: str | None = None,
    ) -> GraphState:
        return await self.graph.ainvoke(self._coerce_state(state, question, doc_id))

    def invoke(
        self,
        state: GraphState | str | None = None,
        *,
        question: str | None = None,
        doc_id: str | None = None,
    ) -> GraphState:
        return self.graph.invoke(self._coerce_state(state, question, doc_id))

    async def astream(
        self,
        state: GraphState | str | None = None,
        *,
        question: str | None = None,
        doc_id: str | None = None,
        stream_mode: str = "updates",
    ) -> Any:
        """Stream per-node updates as the graph executes."""
        resolved = self._coerce_state(state, question, doc_id)
        async for update in self.graph.astream(resolved, stream_mode=stream_mode):
            yield update


def build_graph(
    gpu_cfg: GpuConfig,
    settings: Settings,
    options: BuildOptions | None = None,
    *,
    index: CorpusIndex | None = None,
    feature_flags: dict[str, bool] | None = None,
) -> GraphRunner:
    """Construct and compile the LangGraph with wired dependencies.

    The ``index`` and ``feature_flags`` kwargs are convenience shortcuts that
    mirror fields on :class:`BuildOptions`; they win over anything already set
    on the passed-in ``options`` object, so callers can do either::

        build_graph(gpu_cfg, settings, BuildOptions(index=idx))
        build_graph(gpu_cfg, settings, index=idx)
    """
    options = options or BuildOptions()
    if index is not None:
        options.index = index
    if feature_flags is not None:
        options.feature_flags = feature_flags

    deps = _build_deps(gpu_cfg, settings, options)

    g: StateGraph = StateGraph(GraphState)
    g.add_node("ensure_document", node_ensure_document(deps))
    g.add_node("route", node_route(deps))
    g.add_node("retrieve", node_retrieve(deps))
    g.add_node("extract_scale", node_extract_scale(deps))
    g.add_node("generate", node_generate(deps))
    g.add_node("execute", node_execute(deps))
    g.add_node("verify", node_verify(deps))
    g.add_node("self_consistency", node_self_consistency(deps))
    g.add_node("format", node_format(deps))
    g.add_node("refuse", node_refuse(deps))

    g.set_entry_point("ensure_document")
    g.add_edge("ensure_document", "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "extract_scale")
    g.add_edge("extract_scale", "generate")
    g.add_edge("generate", "execute")
    g.add_edge("execute", "verify")
    g.add_edge("verify", "self_consistency")
    g.add_conditional_edges(
        "self_consistency",
        node_decide,
        {"format": "format", "refuse": "refuse"},
    )
    g.add_edge("format", END)
    g.add_edge("refuse", END)

    checkpointer = _build_checkpointer(settings) if options.use_sqlite_checkpointer else None
    compiled = g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
    return GraphRunner(compiled, deps)


def _build_deps(gpu_cfg: GpuConfig, settings: Settings, options: BuildOptions) -> PipelineDeps:
    index = options.index
    retriever = options.retriever
    if retriever is None and options.enable_retrieval:
        if index is None:
            index_dir = settings.finqa_data_dir / "indices" / "dev"
            if index_dir.exists():
                try:
                    index = load_index(index_dir)
                except Exception as exc:
                    log.warning("Failed to auto-load index at %s: %s", index_dir, exc)
        if index is not None:
            embedder = options.embedder or Embedder(
                model_name=gpu_cfg.embedding.model,
                device=gpu_cfg.embedding.device,
                batch_size=gpu_cfg.embedding.batch_size,
            )
            reranker = options.reranker or Reranker(
                model_name=gpu_cfg.reranker.model,
                device=gpu_cfg.reranker.device,
                batch_size=gpu_cfg.reranker.batch_size,
            )
            retriever = HybridRetriever(
                index=index,
                embedder=embedder,
                cfg=gpu_cfg.retrieval,
                reranker=reranker,
            )

    return PipelineDeps(
        cfg=gpu_cfg,
        settings=settings,
        index=index,
        retriever=retriever,
        scale_extractor=ScaleExtractor(),
        feature_flags=options.feature_flags,
    )


def _build_checkpointer(settings: Settings) -> Any:
    """Return a SQLite-backed LangGraph checkpointer, or ``None`` if unavailable."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = settings.finqa_checkpoint_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(db_path))
    except Exception as exc:
        log.warning("SqliteSaver unavailable (%s); using in-memory checkpointing.", exc)
        return None
