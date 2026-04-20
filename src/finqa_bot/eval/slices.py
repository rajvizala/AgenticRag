"""Load named eval slices defined in ``configs/eval.yaml``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from finqa_bot.config import EvalConfig, SliceSpec
from finqa_bot.data.sample import load_samples
from finqa_bot.logging import get_logger
from finqa_bot.types import EvalSample

log = get_logger(__name__)


def load_slice(
    eval_cfg: EvalConfig,
    slice_name: str,
    data_dir: Path | str = Path("data"),
    n_override: int | None = None,
) -> list[EvalSample]:
    """Resolve a named slice against the eval config and return samples."""
    if slice_name not in eval_cfg.slices:
        raise KeyError(f"Unknown slice '{slice_name}'. Known: {sorted(eval_cfg.slices)}")
    spec = eval_cfg.slices[slice_name]
    split_path = _resolve_split_path(eval_cfg, spec, Path(data_dir))
    n = n_override if n_override is not None else spec.n
    seed = spec.seed
    samples = load_samples(split_path, n=n, seed=seed)

    if spec.filter:
        samples = _apply_filter(samples, spec.filter)
    if spec.perturbation:
        samples = _apply_perturbation(samples, spec.perturbation)
    log.info("Slice %s: %d samples", slice_name, len(samples))
    return samples


def _resolve_split_path(eval_cfg: EvalConfig, spec: SliceSpec, data_dir: Path) -> Path:
    rel = eval_cfg.splits.get(spec.split)
    if not rel:
        raise KeyError(f"No split path configured for '{spec.split}'")
    candidates = [
        Path(rel),
        data_dir / "raw" / "finqa" / f"{spec.split}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


# ------- filters ----------------------------------------------------------

def _apply_filter(samples: list[EvalSample], filter_spec: str) -> list[EvalSample]:
    """Evaluate a tiny predicate DSL on each sample.

    Supported forms:
    - ``program_steps >= N``
    - ``program_contains_any(['const_100', ...])``
    """
    predicate = _compile_filter(filter_spec)
    return [s for s in samples if predicate(s)]


def _compile_filter(spec: str) -> Callable[[EvalSample], bool]:
    spec = spec.strip()
    if spec.startswith("program_steps"):
        parts = spec.split()
        op = parts[1]
        n = int(parts[2])

        def _f(s: EvalSample) -> bool:
            steps = s.gold_program.count("(")
            return _compare(op, steps, n)

        return _f

    if spec.startswith("program_contains_any("):
        inner = spec[len("program_contains_any("):].rstrip(")")
        import ast

        items = ast.literal_eval(inner)

        def _c(s: EvalSample) -> bool:
            prog = s.gold_program or ""
            return any(tok in prog for tok in items)

        return _c

    raise ValueError(f"Unsupported filter: {spec!r}")


def _compare(op: str, a: int, b: int) -> bool:
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    if op == "==":
        return a == b
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    raise ValueError(f"Unknown comparator: {op}")


# ------- perturbations ----------------------------------------------------


def _apply_perturbation(samples: list[EvalSample], name: str) -> list[EvalSample]:
    if name == "delete_gold_row":
        return [_delete_gold_row(s) for s in samples]
    raise ValueError(f"Unknown perturbation: {name!r}")


def _delete_gold_row(sample: EvalSample) -> EvalSample:
    """Remove the gold table row for a sample; used to test refusal behavior."""
    gold_inds = sample.gold_inds
    table_inds = [g for g in gold_inds if g.startswith("table_")]
    if not table_inds:
        return sample
    new_table: list[list[str]] = []
    drop = {int(t.split("_", 1)[1]) for t in table_inds if t.split("_", 1)[1].isdigit()}
    for i, row in enumerate(sample.document.table):
        if i in drop:
            continue
        new_table.append(row)
    doc = sample.document.model_copy(update={"table": new_table})
    return sample.model_copy(update={"document": doc})
