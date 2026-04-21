"""Typed data contracts shared across the pipeline.

These types are deliberately narrow:
- ``Step`` and ``AnswerEnvelope`` are the grammar-constrained output schema the
  generator emits. They map onto the FinQA DSL from Chen et al. 2021 plus a
  small number of production-only fields (``answer_form``, ``scale``,
  ``grounded_numbers``, ``confidence``, ``rationale``).
- ``TableChunk`` is a retrieval unit. Its ``id`` is designed to match FinQA's
  ``gold_inds`` keys (``table_i``, ``text_i``) so that retrieval recall against
  the gold set is a one-line join.
- ``GraphState`` is the LangGraph ``TypedDict``. Every node reads / writes a
  typed slice of it.

All Pydantic models use ``model_config = ConfigDict(extra="forbid")`` so that
schema violations surface loudly rather than drifting silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

DSL_OPS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_sum",
    "table_average",
    "table_max",
    "table_min",
)
DslOp = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_sum",
    "table_average",
    "table_max",
    "table_min",
]
AnswerForm = Literal["percent", "decimal", "ratio", "currency", "count", "boolean"]
Scale = Literal["units", "thousands", "millions", "billions"]
RoutingCategory = Literal[
    "direct_lookup",
    "single_ratio",
    "multi_step",
    "composition",
    "out_of_scope",
]
Route = Literal["specialist", "generalist", "refuse"]
ChunkType = Literal["text", "table_row", "table_summary", "table_header"]


class Step(BaseModel):
    """One operation in a FinQA DSL program.

    ``args`` may contain numeric literals (``42``, ``3.14``), ``"#n"``
    back-references to a prior step result (``"#0"``, ``"#1"``), ``const_*``
    constants from the FinQA paper (``"const_100"``, ``"const_1000"``,
    ``"const_m1"``), or row labels for the ``table_*`` aggregate ops.
    """

    model_config = ConfigDict(extra="forbid")

    op: DslOp
    args: list[float | str] = Field(default_factory=list)
    source: str = Field(default="", description="Chunk id this step draws its numbers from.")

    @field_validator("args", mode="before")
    @classmethod
    def _normalize_args(cls, v: Any) -> list[float | str]:
        if v is None:
            return []
        if not isinstance(v, Sequence) or isinstance(v, str | bytes):
            raise TypeError("Step.args must be a list")
        out: list[float | str] = []
        for item in v:
            if isinstance(item, int | float):
                out.append(float(item))
            elif isinstance(item, str):
                out.append(item)
            else:
                raise TypeError(f"Unsupported arg type: {type(item).__name__}")
        return out


class AnswerEnvelope(BaseModel):
    """Full structured answer emitted by the generator.

    This is what the vLLM xgrammar backend constrains the model to produce.
    """

    model_config = ConfigDict(extra="forbid")

    extracted_evidence: str = Field(
        default="",
        description="First step: quote the exact table rows or text you will use, including values and years.",
    )
    program: list[Step] = Field(default_factory=list)
    answer_value: float | str = Field(description="Numeric answer or 'yes'/'no' for greater.")
    answer_form: AnswerForm = Field(default="decimal")
    scale: Scale = Field(default="units")
    grounded_numbers: list[float] = Field(
        default_factory=list,
        description="Every numeric literal used in the program (for grounding audit).",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=2000)


class TableChunk(BaseModel):
    """A retrieval unit. The ``id`` matches FinQA's ``gold_inds`` when possible.

    FinQA's gold_inds keys look like ``text_0``, ``text_3``, ``table_1``, where
    ``text_i`` points to ``pre_text[i]`` or ``post_text[i - len(pre_text)]``
    and ``table_i`` is a 1-indexed row (header row is row 0).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    chunk_type: ChunkType
    text: str
    row_index: int | None = None
    row_label: str | None = None
    headers: list[str] | None = None
    scale_factor: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentContext(BaseModel):
    """A FinQA example wrapped for retrieval and generation."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    pre_text: list[str] = Field(default_factory=list)
    post_text: list[str] = Field(default_factory=list)
    table: list[list[str]] = Field(default_factory=list)
    scale_factor: float = 1.0
    scale_source: str = "default"


class RetrievalHit(BaseModel):
    """One retrieved chunk plus its score and provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk: TableChunk
    score: float
    source: str = Field(description="dense | bm25 | rrf | reranker")


class RoutingDecision(BaseModel):
    """Output of the router node."""

    model_config = ConfigDict(extra="forbid")

    category: RoutingCategory
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class GroundednessResult(BaseModel):
    """Result of the groundedness check."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    missing: list[float] = Field(default_factory=list)
    normalized_context_numbers: list[float] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Result of running the deterministic DSL executor."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    value: float | str | None = None
    steps: list[float | str] = Field(default_factory=list)
    error: str | None = None


class GraphState(TypedDict, total=False):
    """LangGraph shared state.

    Use ``TypedDict`` because LangGraph copies state across node boundaries and
    accepts partial updates. Not every key is populated on every node.
    """

    question: str
    doc_id: str | None
    document: DocumentContext | None

    routing: RoutingDecision | None

    hits: list[RetrievalHit]
    scale: float

    envelope_candidates: list[AnswerEnvelope]
    envelope: AnswerEnvelope | None

    execution: ExecutionResult | None
    executions: list[ExecutionResult]

    groundedness: GroundednessResult | None
    consistency_agreement: float

    answer_text: str
    refused: bool
    refusal_reason: str

    trace: list[dict[str, Any]]
    error: str | None


class EvalSample(BaseModel):
    """A single FinQA example for the eval harness."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    document: DocumentContext
    gold_program: str = ""
    gold_answer: float | str | None = None
    gold_inds: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(BaseModel):
    """One row of eval output per sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question: str
    predicted_answer: float | str | None
    gold_answer: float | str | None
    execution_correct: bool
    program_correct: bool | None
    predicted_program: str = ""
    gold_program: str = ""
    retrieval_recall: float = 0.0
    retrieval_full_recall: bool = False
    retrieved_ids: list[str] = Field(default_factory=list)
    gold_inds: list[str] = Field(default_factory=list)
    grounded: bool = False
    refused: bool = False
    unit_correct: bool = True
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class EvalSummary(BaseModel):
    """Aggregate statistics across an eval run."""

    model_config = ConfigDict(extra="forbid")

    slice: str
    n: int
    execution_accuracy: float
    program_accuracy: float
    unit_correctness: float
    groundedness_rate: float
    refusal_rate: float
    retrieval_recall: dict[int, float] = Field(default_factory=dict)
    retrieval_full_recall: dict[int, float] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    tokens: dict[str, float] = Field(default_factory=dict)
    cost_per_correct_usd: float = 0.0
    cost_per_question_usd: float = 0.0
    wall_time_s: float = 0.0
    config_id: str = "default"
    ablation_id: str = ""
    ci_low: float = 0.0
    ci_high: float = 0.0
