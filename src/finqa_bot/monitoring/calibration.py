"""Self-consistency-agreement vs correctness calibration bins.

For each answered question, the graph records a ``consistency_agreement`` in
[0, 1] (fraction of K samples whose executed value matches the winner). If
the model is well-calibrated, samples where agreement is high should be
correct more often than samples where agreement is low. A noticeable gap
(e.g. 0.9 agreement yields 95% correctness while 0.34 agreement yields 40%)
is healthy; a flat line means the model's confidence signal is useless.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class CalibrationBins:
    """Aggregates per-bin correctness counts."""

    n_bins: int = 5
    totals: list[int] = field(default_factory=list)
    corrects: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.totals:
            self.totals = [0] * self.n_bins
        if not self.corrects:
            self.corrects = [0] * self.n_bins

    def observe(self, agreement: float, correct: bool) -> None:
        """Record one sample into the appropriate bin."""
        idx = min(self.n_bins - 1, max(0, round(agreement * (self.n_bins - 1))))
        self.totals[idx] += 1
        self.corrects[idx] += int(correct)

    def rates(self) -> list[float]:
        """Return per-bin accuracy rates."""
        return [
            (c / t) if t else 0.0
            for c, t in zip(self.corrects, self.totals, strict=False)
        ]

    def bin_centers(self) -> list[float]:
        step = 1.0 / max(1, self.n_bins - 1)
        return [round(i * step, 4) for i in range(self.n_bins)]


def accumulate_calibration(
    agreements: Iterable[float],
    correct: Iterable[bool],
    n_bins: int = 5,
) -> CalibrationBins:
    """Build a :class:`CalibrationBins` from paired iterables."""
    bins = CalibrationBins(n_bins=n_bins)
    for a, c in zip(agreements, correct, strict=False):
        bins.observe(a, c)
    return bins
