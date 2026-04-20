"""Gradio ChatInterface client that streams from the FastAPI SSE endpoint.

We stream per-node updates as a side-panel log so reviewers can *watch* the
LangGraph progression (router -> retrieve -> generate -> execute -> verify)
without digging through logs, and the final answer renders the typed program
plus citations in collapsible sections.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import gradio as gr
import httpx

from finqa_bot.config import get_settings
from finqa_bot.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def _is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    import os
    return bool(os.environ.get("COLAB_RELEASE_TAG"))


@dataclass
class StreamEvent:
    kind: str
    node: str | None
    payload: Any


def _emit_launch_diagnostic(message: str) -> None:
    """Write launch diagnostics to stdout and the configured logger."""
    print(message, flush=True)
    log.info(message)


def _summarize_launch_result(result: Any) -> tuple[str | None, str | None]:
    """Best-effort extraction of local/share URLs from Gradio launch()."""
    local_url: str | None = None
    share_url: str | None = None
    if isinstance(result, tuple):
        if len(result) >= 2 and isinstance(result[1], str):
            local_url = result[1]
        if len(result) >= 3 and isinstance(result[2], str):
            share_url = result[2]
    return local_url, share_url


async def _stream_from_api(api_base: str, question: str, doc_id: str | None) -> Iterable[StreamEvent]:
    events: list[StreamEvent] = []
    url = api_base.rstrip("/") + "/chat"
    payload = {"question": question, "doc_id": doc_id}
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)) as client,
        client.stream("POST", url, json=payload) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:]
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                continue
            kind = parsed.get("event", "node")
            events.append(StreamEvent(kind=kind, node=parsed.get("node"), payload=parsed))
    return events


def _render_program(program: list[dict[str, Any]] | None) -> str:
    if not program:
        return "_(empty program)_"
    lines = []
    for i, step in enumerate(program):
        args = ", ".join(str(a) for a in step.get("args", []))
        src = step.get("source", "")
        lines.append(f"`#{i}` **{step.get('op')}**({args})  _(src: {src})_")
    return "\n\n".join(lines)


def _render_citations(hits: list[dict[str, Any]] | None) -> str:
    if not hits:
        return "_(no retrieved citations)_"
    lines = []
    for h in hits[:8]:
        if "chunk" in h and isinstance(h["chunk"], dict):
            cid = h["chunk"].get("id", "?")
            text = (h["chunk"].get("text") or "")[:240]
            chunk_type = h["chunk"].get("chunk_type", "")
        else:
            cid = h.get("chunk_id", "?")
            text = (h.get("text") or "")[:240]
            chunk_type = h.get("chunk_type", "")
        score = h.get("score", 0.0)
        source = h.get("source", "")
        lines.append(
            f"- **{cid}** `{chunk_type}` via `{source}` (score={float(score):.3f})\n  > {text}"
        )
    return "\n".join(lines)


_REFUSAL_MESSAGES: dict[str, str] = {
    "question_out_of_scope": (
        "This question appears to be outside the scope of the financial filings corpus. "
        "Please ask about specific financial figures from SEC 10-K filings."
    ),
    "no_majority_agreement": (
        "I generated multiple candidate answers but they didn't agree with each other. "
        "I need consistent results before I'm confident enough to show a number."
    ),
    "ungrounded_literals": (
        "The numbers in the generated calculation don't appear in the retrieved financial "
        "context, which suggests the model may have invented figures rather than read them."
    ),
    "missing_envelope": (
        "No valid answer structure was produced for this question."
    ),
}

_REFUSAL_DEFAULT = (
    "I couldn't find the specific financial data needed to answer this question "
    "in the available filings. The relevant figures may use different terminology "
    "or may not be present in the indexed documents. Try rephrasing the question "
    "or specifying the exact line item name as it appears in the report."
)


def _user_facing_refusal(reason: str) -> str:
    for key, msg in _REFUSAL_MESSAGES.items():
        if key in reason:
            return msg
    return _REFUSAL_DEFAULT


def _render_final(state: dict[str, Any]) -> tuple[str, str, str]:
    env = state.get("envelope") or {}
    if isinstance(env, dict):
        program = env.get("program", [])
        form = env.get("answer_form", "")
        scale = env.get("scale", "")
        rationale = env.get("rationale", "")
        conf = float(env.get("confidence", 0.0) or 0.0)
        answer = env.get("answer_value", "")
    else:
        program = []
        form = scale = rationale = ""
        conf = 0.0
        answer = ""
    execution = state.get("execution") or {}
    if isinstance(execution, dict) and execution.get("ok"):
        answer = execution.get("value", answer)
    refused = state.get("refused")
    reason = state.get("refusal_reason") or ""

    if refused:
        friendly = _user_facing_refusal(reason)
        main = f"**Answer not available.**\n\n{friendly}"
        # Show the rationale from the model if it provided one (e.g. "context
        # does not contain operating income") — it adds useful context for
        # the user without exposing raw pipeline internals.
        if rationale and conf > 0.0:
            main += f"\n\n_Model note: {rationale}_"
        program_md = _render_program(program)
        citations_md = _render_citations(state.get("hits"))
        return main, program_md, citations_md

    main = (
        f"**Answer:** `{answer}`  _(form={form}, scale={scale}, confidence={conf:.2f})_"
        f"\n\n{rationale}"
    )
    program_md = _render_program(program)
    citations_md = _render_citations(state.get("hits"))
    return main, program_md, citations_md


def _build_ui(api_base: str) -> gr.Blocks:
    with gr.Blocks(title="FinQA Bot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# FinQA Bot\n"
            "Ask a question over the FinQA financial filings corpus. The graph "
            "retrieves the relevant rows, generates a typed DSL program, "
            "executes it deterministically, and refuses when confidence is low."
        )
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", lines=3, placeholder="What was the change in operating income from 2006 to 2007?")
                doc_id = gr.Textbox(label="doc_id (optional)", placeholder="Single_JKHY/2009/page_28.pdf-3")
                submit = gr.Button("Ask", variant="primary")
                status = gr.Markdown("")
            with gr.Column(scale=3):
                with gr.Accordion("Answer", open=True):
                    answer_md = gr.Markdown("_(no answer yet)_")
                with gr.Accordion("Typed program", open=False):
                    program_md = gr.Markdown("")
                with gr.Accordion("Citations", open=False):
                    citations_md = gr.Markdown("")
                with gr.Accordion("Graph trace", open=False):
                    trace_md = gr.Markdown("")

        async def _ask(q: str, d: str | None) -> Any:
            events = await _stream_from_api(api_base, q, d)
            trace_lines = []
            final_state: dict[str, Any] | None = None
            for ev in events:
                if ev.kind == "node":
                    node = ev.node or "node"
                    trace_lines.append(f"- `{node}`")
                elif ev.kind == "final":
                    final_state = ev.payload.get("state")
                elif ev.kind == "error":
                    return "error", f"**Error:** {ev.payload.get('message')}", "", "", "\n".join(trace_lines)
            if final_state is None:
                return "error", "_(no final state received)_", "", "", "\n".join(trace_lines)
            answer, program, citations = _render_final(final_state)
            return "done", answer, program, citations, "\n".join(trace_lines)

        def _on_submit(q: str, d: str) -> Any:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_ask(q, d or None))
            finally:
                loop.close()

        submit.click(
            _on_submit,
            inputs=[question, doc_id],
            outputs=[status, answer_md, program_md, citations_md, trace_md],
        )
    return demo


def launch(api_base: str | None = None, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio app.

    In Google Colab, ``share`` is forced to ``True`` so that a public
    ``*.gradio.live`` URL is printed. Localhost is unreachable in Colab,
    and this removes the need for manual proxy workarounds.
    """
    settings = get_settings()
    api = api_base or f"http://127.0.0.1:{settings.finqa_api_port}"
    demo = _build_ui(api)

    # Auto-enable share in Colab so users always get a reachable URL.
    if _is_colab() and not share:
        _emit_launch_diagnostic("Colab detected - enabling Gradio share tunnel automatically.")
        share = True

    _emit_launch_diagnostic(
        f"Launching Gradio: api_base={api} server_name={server_name} server_port={server_port} share={share}"
    )
    try:
        launch_result = demo.queue().launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_api=False,
        )
    except Exception as exc:
        _emit_launch_diagnostic(f"Gradio launch failed: {exc}")
        _emit_launch_diagnostic(traceback.format_exc())
        raise

    local_url, share_url = _summarize_launch_result(launch_result)
    _emit_launch_diagnostic(
        f"Gradio launch returned: local_url={local_url or 'n/a'} share_url={share_url or 'n/a'}"
    )


__all__ = ["launch"]
