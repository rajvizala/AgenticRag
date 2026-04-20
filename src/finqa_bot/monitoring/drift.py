"""Drift signals.

Four signals are exported:

1. **Input drift (PSI)**: Population Stability Index on a topic distribution
   of incoming questions versus a reference distribution saved at index time.
   PSI < 0.1 is stable; 0.1-0.2 moderate; > 0.2 alert. Implementation follows
   the Evidently / Arize convention (equal-width bins, 10 buckets).
2. **Retrieval drift**: rolling-window hit-rate@k against a frozen
   50-question golden set. A drop means either the retriever has degraded or
   the content mix of user questions has shifted.
3. **Output drift**: simple distribution summaries over the last window -
   program length, operator mix, refusal rate, groundedness rate. Exposed as
   gauges, alerting on relative changes.
4. **Calibration drift**: self-consistency agreement binned against
   correctness (see :mod:`finqa_bot.monitoring.calibration`).
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class OutputDistribution:
    """Rolling summary of the program-level outputs produced by the graph."""

    n: int
    mean_program_length: float
    operator_mix: dict[str, float]
    refusal_rate: float
    groundedness_rate: float


def summarize_outputs(
    program_lengths: Sequence[int],
    operators: Sequence[str],
    refusal_flags: Sequence[bool],
    grounded_flags: Sequence[bool],
) -> OutputDistribution:
    """Compute a snapshot :class:`OutputDistribution`."""
    n = len(program_lengths)
    if n == 0:
        return OutputDistribution(0, 0.0, {}, 0.0, 0.0)
    mix: dict[str, float] = {}
    counter = Counter(operators)
    total = sum(counter.values()) or 1
    for op, count in counter.items():
        mix[op] = count / total
    return OutputDistribution(
        n=n,
        mean_program_length=sum(program_lengths) / n,
        operator_mix=mix,
        refusal_rate=sum(refusal_flags) / n,
        groundedness_rate=sum(grounded_flags) / n,
    )


def compute_psi(
    baseline: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute PSI between two scalar distributions.

    Uses ``bins`` equal-width buckets determined by the union of the two
    samples. Returns a non-negative float.
    """
    if not baseline or not current:
        return 0.0
    lo = min(min(baseline), min(current))
    hi = max(max(baseline), max(current))
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    b_hist, _ = np.histogram(baseline, bins=edges)
    c_hist, _ = np.histogram(current, bins=edges)
    b_dist = b_hist / max(b_hist.sum(), 1)
    c_dist = c_hist / max(c_hist.sum(), 1)
    b_dist = np.where(b_dist == 0, eps, b_dist)
    c_dist = np.where(c_dist == 0, eps, c_dist)
    psi = float(np.sum((c_dist - b_dist) * np.log(c_dist / b_dist)))
    return max(psi, 0.0)


class RollingHitRate:
    """Fixed-size sliding window of retrieval recalls for the golden set."""

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self._recalls: deque[float] = deque(maxlen=window)

    def observe(self, recall: float) -> None:
        self._recalls.append(recall)

    def current(self) -> float:
        if not self._recalls:
            return 0.0
        return sum(self._recalls) / len(self._recalls)


def rolling_hit_rate(recalls: Iterable[float], window: int = 200) -> float:
    """Convenience: return the mean of the most recent ``window`` recalls."""
    r = RollingHitRate(window=window)
    for v in recalls:
        r.observe(v)
    return r.current()
