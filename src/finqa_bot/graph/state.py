"""Graph state helpers.

The concrete :class:`GraphState` type lives in :mod:`finqa_bot.types` so that
every module importing it avoids circular imports. This module only exposes
constructors.
"""

from __future__ import annotations

from finqa_bot.types import GraphState


def initial_state(question: str, doc_id: str | None = None) -> GraphState:
    """Return a fresh :class:`GraphState` seeded with the user question."""
    return GraphState(
        question=question,
        doc_id=doc_id,
        document=None,
        routing=None,
        hits=[],
        scale=1.0,
        envelope_candidates=[],
        envelope=None,
        execution=None,
        executions=[],
        groundedness=None,
        consistency_agreement=0.0,
        answer_text="",
        refused=False,
        refusal_reason="",
        trace=[],
        error=None,
    )
