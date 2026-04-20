"""Typer CLI.

All commands are thin wrappers over library calls. Heavy lifting lives under
``finqa_bot.{data,retrieval,graph,eval}``.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from finqa_bot.types import EvalRecord

# LangChain's with_structured_output returns an AIMessage whose `parsed`
# attribute holds the deserialized Pydantic object.  When Pydantic later tries
# to serialize that message it sees the unexpected value and emits a UserWarning.
# The warning is cosmetic — the parsed object is returned correctly — but it
# floods the terminal during eval runs.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)

from finqa_bot.config import Settings, get_settings, load_eval_config, load_gpu_config
from finqa_bot.logging import configure_logging, get_logger

configure_logging()

app = typer.Typer(
    name="finqa-bot",
    help="FinQA chatbot CLI: ingest data, serve the graph, run eval + ablations.",
    no_args_is_help=True,
    add_completion=False,
)

log = get_logger(__name__)


@app.command()
def version() -> None:
    """Print the installed version."""
    from finqa_bot import __version__

    typer.echo(__version__)


@app.command()
def ingest(
    split: str = typer.Option("all", help="'train', 'dev', 'test', or 'all'."),
    data_dir: Path | None = typer.Option(None, help="Override data dir."),
    rebuild_index: bool = typer.Option(False, help="Rebuild FAISS+BM25 index even if present."),
) -> None:
    """Download FinQA and build the row-granular retrieval index."""
    from finqa_bot.data.downloader import download_finqa
    from finqa_bot.retrieval.indexer import build_index

    settings = Settings()
    target_dir = data_dir or settings.finqa_data_dir
    splits = ["train", "dev", "test"] if split == "all" else [split]
    for s in splits:
        download_finqa(target_dir, split=s)
    cfg = load_gpu_config()
    build_index(target_dir, cfg, rebuild=rebuild_index)


@app.command()
def eval(
    split: str = typer.Option("dev"),
    n: int | None = typer.Option(200, help="Number of samples; None for full split."),
    slice: str | None = typer.Option(None, help="Named slice from configs/eval.yaml."),
    out: Path = typer.Option(Path("runs/eval_latest.json"), help="Where to write JSON output."),
    concurrency: int | None = typer.Option(None, help="Override eval concurrency."),
    ablation: str | None = typer.Option(None, help="Apply named ablation from eval config."),
    log_file: Path = typer.Option(
        Path("runs/eval.log"),
        help="File to receive all verbose logs (INFO+). Terminal shows only progress.",
    ),
) -> None:
    """Run eval over a FinQA slice and emit JSON + per-sample progress to stdout.

    All INFO-level pipeline logs (generate, execute, retrieve ...) are
    redirected to --log-file so the cell output stays readable.
    """
    from finqa_bot.eval.harness import run_eval
    from finqa_bot.logging import add_file_handler

    # Redirect verbose logs to file; console shows WARNING+ only.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    add_file_handler(log_file, file_level=logging.INFO, console_level=logging.WARNING)

    eval_cfg = load_eval_config()
    gpu_cfg = load_gpu_config()
    settings = get_settings()

    # --- per-sample progress state (asyncio is single-threaded; no lock needed) ---
    _n_correct: list[int] = [0]
    _n_done: list[int] = [0]

    def _progress(record: "EvalRecord", completed: int, total: int) -> None:  # noqa: F821
        _n_done[0] = completed
        if record.execution_correct:
            _n_correct[0] += 1
        acc = _n_correct[0] / completed
        status = "OK   " if record.execution_correct else "WRONG"
        sid = record.sample_id
        # Keep sample id to a fixed width (trim left if long)
        sid_display = ("..." + sid[-27:]) if len(sid) > 30 else sid
        pred = record.predicted_answer
        gold = record.gold_answer
        # Format answer values compactly
        pred_s = "None" if pred is None else f"{pred:.5g}"
        gold_s = "None" if gold is None else (
            f"{gold:.5g}" if isinstance(gold, float) else str(gold)
        )
        typer.echo(
            f"[{completed:4d}/{total}]  {sid_display:<30}  "
            f"pred={pred_s:<12}  gold={gold_s:<12}  {status}  "
            f"ok={_n_correct[0]:3d}  acc={acc:.3f}"
        )

    typer.echo(f"Detailed logs -> {log_file}")
    typer.echo("-" * 80)

    summary = asyncio.run(
        run_eval(
            gpu_cfg=gpu_cfg,
            eval_cfg=eval_cfg,
            settings=settings,
            slice_name=slice,
            split=split,
            n=n,
            concurrency=concurrency,
            ablation_id=ablation,
            out_path=out,
            progress_fn=_progress,
        )
    )

    # Clean final summary — key metrics only, no full JSON dump.
    r = summary
    w = 42
    typer.echo("\n" + "=" * w)
    typer.echo(f"  EVAL COMPLETE  |  {r.slice}")
    typer.echo("=" * w)
    typer.echo(f"  Execution Accuracy  : {r.execution_accuracy:.1%}  ({round(r.execution_accuracy * r.n)}/{r.n})")
    typer.echo(f"  Refusal Rate        : {r.refusal_rate:.1%}")
    typer.echo(f"  Groundedness Rate   : {r.groundedness_rate:.1%}")
    typer.echo(f"  Unit Correctness    : {r.unit_correctness:.1%}")
    rec = r.retrieval_recall
    typer.echo(f"  Retrieval Recall@1  : {rec.get(1, 0):.1%}")
    typer.echo(f"  Retrieval Recall@5  : {rec.get(5, 0):.1%}")
    lat = r.latency_ms
    typer.echo(f"  Latency p50 / p90   : {lat.get('p50', 0) / 1000:.1f}s / {lat.get('p90', 0) / 1000:.1f}s")
    typer.echo(f"  Wall time           : {r.wall_time_s:.0f}s")
    typer.echo(f"  JSON output         : {out}")
    typer.echo(f"  Detailed logs       : {log_file}")
    typer.echo("=" * w)


@app.command()
def ablate(
    out: Path = typer.Option(Path("docs/ABLATIONS.md")),
    n: int = typer.Option(200, help="Samples per ablation."),
    slice: str = typer.Option("dev_200"),
) -> None:
    """Run every ablation declared in ``configs/eval.yaml`` and write a single Markdown table."""
    from finqa_bot.eval.ablations import run_ablation_matrix

    eval_cfg = load_eval_config()
    gpu_cfg = load_gpu_config()
    settings = Settings()
    run_ablation_matrix(
        gpu_cfg=gpu_cfg,
        eval_cfg=eval_cfg,
        settings=settings,
        slice_name=slice,
        n=n,
        out_path=out,
    )


@app.command()
def bench(
    out: Path = typer.Option(Path("docs/GPU_BENCHMARK.md")),
    n: int = typer.Option(100, help="Samples for accuracy side of the benchmark."),
) -> None:
    """Run the GPU quantization-vs-throughput benchmark matrix."""
    from finqa_bot.eval.gpu_benchmark import run_gpu_benchmark

    gpu_cfg = load_gpu_config()
    settings = Settings()
    run_gpu_benchmark(gpu_cfg=gpu_cfg, settings=settings, n=n, out_path=out)


@app.command()
def ask(
    question: str,
    doc_id: str | None = typer.Option(None),
) -> None:
    """Ask a one-shot question against the default graph."""
    from finqa_bot.graph.graph import build_graph

    gpu_cfg = load_gpu_config()
    settings = get_settings()
    runner = build_graph(gpu_cfg=gpu_cfg, settings=settings)

    async def _run() -> None:
        result = await runner.ainvoke(question=question, doc_id=doc_id)
        typer.echo(result.get("answer_text", "(no answer)"))

    asyncio.run(_run())


@app.command()
def demo(
    api_base: str | None = typer.Option(None, help="FastAPI base URL (defaults to local)."),
    share: bool = typer.Option(False, help="Enable Gradio share tunnel."),
    port: int | None = typer.Option(None, help="Gradio port."),
) -> None:
    """Launch the Gradio demo UI (talks to the FastAPI /chat endpoint)."""
    from finqa_bot.ui.gradio_app import launch

    settings = get_settings()
    launch(
        api_base=api_base,
        server_port=port or settings.finqa_gradio_port,
        share=share,
    )


@app.command()
def api(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    reload: bool = typer.Option(False, help="Enable uvicorn --reload (dev only)."),
) -> None:
    """Launch the FastAPI /chat SSE server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "finqa_bot.ui.api:app",
        host=host or settings.finqa_api_host,
        port=port or settings.finqa_api_port,
        log_level="info",
        reload=reload,
    )


@app.command()
def doctor() -> None:
    """Print the resolved configuration and diagnostic probes."""
    settings = Settings()
    typer.echo("Settings:")
    typer.echo(settings.model_dump_json(indent=2))
    try:
        cfg = load_gpu_config()
        typer.echo("\nGpuConfig:")
        typer.echo(cfg.model_dump_json(indent=2))
    except Exception as exc:
        typer.echo(f"\nGpuConfig: FAILED to load ({exc})")


if __name__ == "__main__":
    app()
