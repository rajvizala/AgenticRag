# Architecture

## High-level diagram

```
                   +--------------+
                   |  User / API  |
                   +------+-------+
                          |
                  FastAPI /ask /chat (SSE)
                          |
                          v
                 +--------+---------+
                 |   GraphRunner    |  (finqa_bot.graph.graph)
                 +--------+---------+
                          |
                          v
     +----------+   +-----+------+   +----------+   +----------+
     |  router  |-->|  retrieve  |-->|  scale   |-->| generate |
     | (tier-0) |   | (hybrid)   |   |          |   | (typed)  |
     +----------+   +------------+   +----------+   +----+-----+
           |              ^                              |
           v              | (SqliteSaver checkpointer)   v
       +--------+    +---------+                   +----------+
       | refuse |<---+ verify  +<------------------+ execute  |
       +--------+    +----+----+                   +-----+----+
                          |                              ^
                          v                              |
                    +-----+------+                 +-----+------+
                    | consistency+----------------->| format/end |
                    +------------+                 +------------+
```

## LangGraph nodes

All nodes live in `src/finqa_bot/graph/nodes.py`. They are pure functions over
`GraphState` (`src/finqa_bot/graph/state.py`) and raise on unrecoverable
errors rather than swallowing them, so the graph surfaces failures.

### `router`
Classifies the incoming question into one of `lookup`, `ratio`, `multi_step`,
`out_of_scope` using Qwen3-1.7B. Writes `state.route` and `state.specialist`
so `generate` can pick the specialist or generalist. If the router says
`out_of_scope` it routes directly to `refuse`.

### `retrieve`
Hybrid retrieval: Qwen3-Embedding FAISS (top 40) merged with BM25 (top 40)
via Reciprocal-Rank Fusion (k=60). Top 20 are reranked by Qwen3-Reranker, and
the top 5 are kept. Retrieval is doc-filtered when the request provides a
`doc_id` (FinQA evaluation always does). Writes `state.hits`.

### `scale`
Runs `ScaleExtractor` over the retrieved context (pre-text + table header +
post-text). Regex handles `(in millions)`, `(dollars in thousands)`, and
similar; a tiny LM confirms ambiguous cases. Writes `state.scale_factor`.

### `generate`
Calls the generator model via `with_structured_output(AnswerEnvelope)` on
vLLM. The LLM emits a JSON conforming to:

```python
class DSLStep(BaseModel):
    op: Literal[...10 ops...]
    args: list[str]          # "#n" back-ref or literal

class AnswerEnvelope(BaseModel):
    program: list[DSLStep]
    answer: str
    answer_form: Literal["number", "percent", "boolean"]
    scale: Literal["unit", "thousand", "million", "billion"]
    evidence_ids: list[str]
    rationale: str
```

Grammar constraint is enforced by vLLM's xgrammar backend; parse failures
are impossible. Writes `state.envelope`.

### `execute`
Runs the DSL deterministically. Numbers are normalized (currency / commas /
percents / parentheses / "million" / "billion" suffixes) by
`execution.numbers._parse_number`. Writes `state.execution` containing the
numeric answer and every intermediate value.

### `verify`
Two gates:
1. **Groundedness**: every literal in the program must appear in the
   retrieved context (after normalization).
2. **Unit sanity**: `envelope.scale` must match the document scale factor
   within one order of magnitude of the answer.

On failure, writes `state.verification.failed = True` and `reasons`.

### `consistency`
If `feature_flags.self_consistency` is on, draws K=3 samples from `generate`
(via `n=3` sampling in vLLM), executes each, and votes on the *numeric*
results (after rounding to 5 decimals). Ties are broken by groundedness
score. Writes `state.consistency_agreement`.

### `format` / `refuse`
`format` emits the final `AnswerEnvelope` and attaches citations
(`hit.chunk.id` for every chunk whose text appears in a program literal).
`refuse` emits a typed refusal with a reason drawn from
`state.verification.reasons` or the router's out-of-scope verdict.

## Persistence

- SqliteSaver checkpointer stores each state transition under the
  `thread_id` passed to the graph. This enables replay for debugging and
  the `demo` UI's "inspect state" feature.
- Retrieval index is persisted to `data/index/` (FAISS + BM25 + row
  metadata). Rebuild with `make ingest`.

## Streaming contract

`GraphRunner.astream` exposes two modes:
- `updates` - one event per node completion with the state delta.
- `messages` - token-level streaming from any LLM-emitting node.

FastAPI `/chat` forwards both as a single SSE stream with `event:` tags.
Gradio consumes the SSE stream and renders:
- A live node-by-node trace.
- The final typed program as a tree.
- Citations as expandable chunks (pre/post/table).

## Feature flags

`configs/eval.yaml` and the CLI expose:

- `retrieve` (default on)
- `rerank` (default on)
- `scale_extract` (default on)
- `router` (default on)
- `self_consistency` (default on; K=3)
- `groundedness_gate` (default on)
- `units_gate` (default on)

Flags turn off *nodes*; they never silently change answers. The eval
harness records the flag vector with every run so ablation tables are
traceable to commits.
