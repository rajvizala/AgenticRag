"""FastAPI server for the FinQA bot.

Exposes:

* ``GET /health`` - readiness probe.
* ``POST /ask`` - synchronous JSON endpoint returning the full envelope.
* ``POST /chat`` - Server-Sent Events stream of graph-node updates plus the
  final answer envelope, for the Gradio UI to render a progress trace.
* ``GET /metrics`` - Prometheus scrape endpoint (our custom metrics; vLLM
  exposes its own ``/metrics`` on the inference port).

The server builds the :class:`GraphRunner` lazily on first request to keep
startup fast, then reuses it across requests.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from finqa_bot.config import GpuConfig, get_settings, load_eval_config, load_gpu_config
from finqa_bot.graph.graph import GraphRunner, build_graph
from finqa_bot.logging import configure_logging, get_logger
from finqa_bot.monitoring.metrics import registry
from finqa_bot.retrieval.indexer import build_index
from finqa_bot.types import AnswerEnvelope, RetrievalHit

configure_logging()
log = get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: str | None = None


class AskResponse(BaseModel):
    answer: Any
    envelope: Any
    program: Any
    citations: list[dict[str, Any]]
    refused: bool
    refusal_reason: str | None
    latency_ms: float
    trace: list[dict[str, Any]] | None = None


_runner: GraphRunner | None = None


def _prepare_runtime_gpu_config(gpu_cfg: GpuConfig) -> GpuConfig:
    """Enforce model placement rules that cannot be expressed purely in YAML.

    T4 (Turing, 16 GB):
        vLLM owns the entire GPU.  All auxiliary models must run on CPU to
        avoid the ~1.5 GiB parasitic CUDA context that would be claimed by
        a second PyTorch process, leaving no room for activation buffers.

    L4/Ada (Ada, 24 GB):
        vLLM (14B AWQ) + embedder + reranker consume ~20.6 GiB, leaving only
        ~1.4 GiB free — not enough for the 1.7B router (~3.2 GiB in fp16).
        Embedder and reranker (0.6B each, ~0.5 GiB each) fit; the router
        must stay on CPU.
    """
    arch = gpu_cfg.gpu.arch
    if arch == "turing":
        adjusted = gpu_cfg.model_copy(deep=True)
        adjusted.embedding.device = "cpu"
        adjusted.reranker.device = "cpu"
        adjusted.router.device = "cpu"
        return adjusted
    if arch == "ada":
        adjusted = gpu_cfg.model_copy(deep=True)
        adjusted.router.device = "cpu"
        return adjusted
    return gpu_cfg


async def _ensure_runner() -> GraphRunner:
    global _runner
    if _runner is not None:
        return _runner
    settings = get_settings()
    gpu_cfg = load_gpu_config(settings.finqa_gpu_config)
    runtime_gpu_cfg = _prepare_runtime_gpu_config(gpu_cfg)
    _ = load_eval_config(settings.finqa_eval_config)
    index = build_index(settings.finqa_data_dir, gpu_cfg, split="dev")

    # Note: we do NOT block here waiting for vLLM.  setup.sh already
    # waits for the model to be loaded before starting us, and the
    # retry logic in _sample_envelopes (nodes.py) handles any remaining
    # race between the API and vLLM.
    _runner = build_graph(settings=settings, gpu_cfg=runtime_gpu_cfg, index=index)
    return _runner


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        await _ensure_runner()
    except Exception as exc:
        log.warning("Deferred graph init: %s", exc)
    yield


app = FastAPI(title="FinQA Bot", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "runner": _runner is not None}


@app.get("/metrics")
async def metrics() -> Response:
    data, content_type = registry.export()
    return Response(content=data, media_type=content_type)


def _summarise_state(result: dict[str, Any], elapsed_ms: float) -> AskResponse:
    envelope: AnswerEnvelope | None = result.get("envelope")
    execution = result.get("execution")
    answer: Any = None
    if execution is not None and getattr(execution, "ok", False):
        answer = execution.value
    elif envelope is not None:
        answer = envelope.answer_value
    program_dump = [s.model_dump() for s in envelope.program] if envelope is not None else []
    citations = []
    for hit in result.get("hits") or []:  # type: ignore[assignment]
        if isinstance(hit, RetrievalHit):
            citations.append({
                "chunk_id": hit.chunk.id,
                "text": hit.chunk.text,
                "score": hit.score,
                "source": hit.source,
                "doc_id": hit.chunk.doc_id,
                "chunk_type": hit.chunk.chunk_type,
            })
    return AskResponse(
        answer=answer,
        envelope=envelope.model_dump() if envelope is not None else None,
        program=program_dump,
        citations=citations,
        refused=bool(result.get("refused")),
        refusal_reason=result.get("refusal_reason"),
        latency_ms=elapsed_ms,
        trace=result.get("trace"),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    runner = await _ensure_runner()
    started = time.perf_counter()
    try:
        result = await runner.ainvoke(question=req.question, doc_id=req.doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000
    registry.record_graph_outcome(result, elapsed / 1000)
    return _summarise_state(result, elapsed)


async def _sse_stream(question: str, doc_id: str | None) -> AsyncIterator[bytes]:
    runner = await _ensure_runner()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def _producer() -> None:
        started = time.perf_counter()
        final_state: dict[str, Any] = {"question": question, "doc_id": doc_id}
        try:
            async for update in runner.astream(question=question, doc_id=doc_id):
                for node_name, payload in update.items():
                    if isinstance(payload, dict):
                        final_state.update(payload)
                    await queue.put({"event": "node", "node": node_name, "data": _serialise(payload)})
        except Exception as exc:
            await queue.put({"event": "error", "message": str(exc)})
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            registry.record_graph_outcome(final_state, elapsed / 1000)
            await queue.put({"event": "final", "state": _serialise(final_state), "latency_ms": elapsed})
            await queue.put(None)

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n".encode()
    finally:
        producer_task.cancel()


def _serialise(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_serialise(v) for v in obj]
    return obj


@app.post("/chat")
async def chat(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(_sse_stream(req.question, req.doc_id), media_type="text/event-stream")


def create_app() -> FastAPI:
    """Entry point for ``uvicorn finqa_bot.ui.api:create_app``."""
    return app


__all__ = ["app", "create_app"]
