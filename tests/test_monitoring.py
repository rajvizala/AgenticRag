"""Tests for monitoring: drift, calibration, and Prometheus metrics registry."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from finqa_bot.monitoring.calibration import CalibrationBins, accumulate_calibration
from finqa_bot.monitoring.drift import (
    OutputDistribution,
    compute_psi,
    rolling_hit_rate,
    summarize_outputs,
)
from finqa_bot.monitoring.metrics import MetricsRegistry


def test_psi_zero_for_identical_distributions() -> None:
    base = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    current = list(base)
    assert compute_psi(base, current) == 0.0


def test_psi_positive_for_shifted_distribution() -> None:
    base = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    current = [7.0, 8.0, 9.0, 10.0, 11.0] * 20
    assert compute_psi(base, current) > 0.2


def test_rolling_hit_rate_windows() -> None:
    recalls = [1.0] * 100 + [0.0] * 50
    rr = rolling_hit_rate(recalls, window=25)
    # Last 25 entries are all 0.0.
    assert rr == 0.0


def test_summarize_outputs() -> None:
    dist: OutputDistribution = summarize_outputs(
        program_lengths=[1, 2, 3],
        operators=["add", "add", "divide"],
        refusal_flags=[False, True, False],
        grounded_flags=[True, True, False],
    )
    assert dist.n == 3
    assert dist.mean_program_length == 2.0
    assert dist.operator_mix["add"] > dist.operator_mix["divide"]
    assert dist.refusal_rate == 1 / 3


def test_calibration_bins_accumulate() -> None:
    # `observe` snaps each agreement to the nearest bin centre in {0.0, 0.5, 1.0}
    # for n_bins=3, so we feed it agreements that line up with those centres.
    bins: CalibrationBins = accumulate_calibration(
        agreements=[0.0, 0.5, 1.0, 1.0, 1.0],
        correct=[False, True, True, True, False],
        n_bins=3,
    )
    rates = bins.rates()
    assert rates[0] == 0.0
    assert rates[1] == 1.0
    assert 0.5 < rates[2] <= 1.0


def test_metrics_registry_roundtrip() -> None:
    reg = CollectorRegistry()
    m = MetricsRegistry(registry=reg)
    m.questions_total.inc()
    m.refusals_total.labels(reason="ungrounded").inc()
    m.latency_seconds.observe(0.5)
    data, content_type = m.export()
    body = data.decode("utf-8")
    assert "finqa_questions_total" in body
    assert "finqa_refusals_total" in body
    assert "finqa_graph_latency_seconds" in body
    assert "text/plain" in content_type
