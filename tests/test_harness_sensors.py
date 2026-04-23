"""Unit tests for ObjectiveSensor — the deterministic signal producer.

These tests exercise the state machine directly by feeding measurements
in. No ClickHouse, no compute_fn resolution, no engine — just the
sensor's tick logic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from opentracy.harness.ledger import LedgerStore
from opentracy.harness.objectives.schemas import (
    GuardrailSpec,
    Objective,
    ObjectiveMeasurement,
)
from opentracy.harness.sensors import SENSOR_TAG_REGRESSION, ObjectiveSensor


def _objective(
    *,
    objective_id: str = "cost_per_successful_completion",
    direction: str = "lower_is_better",
    regression_pct: float = 5.0,
) -> Objective:
    return Objective(
        id=objective_id,
        description="test objective",
        compute_fn="does.not:matter",
        unit="USD",
        direction=direction,  # type: ignore[arg-type]
        guardrails=[
            GuardrailSpec(type="no_regression_worse_than_pct", threshold=regression_pct),
        ],
    )


def _measurement(value: float, sample_size: int = 100, dims: dict | None = None) -> ObjectiveMeasurement:
    now = datetime.now(timezone.utc).isoformat()
    return ObjectiveMeasurement(
        objective_id="cost_per_successful_completion",
        value=value,
        unit="USD",
        sample_size=sample_size,
        window_start=now,
        window_end=now,
        dimension_values=dims or {},
        computed_at=now,
    )


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Baseline behavior
# ---------------------------------------------------------------------------


def test_first_tick_establishes_baseline_and_emits_nothing(ledger):
    sensor = ObjectiveSensor(_objective(), ledger)
    out = sensor.tick([_measurement(0.01)])
    assert out is None
    assert ledger.recent() == []


def test_empty_measurements_no_signal_no_baseline_update(ledger):
    sensor = ObjectiveSensor(_objective(), ledger)
    assert sensor.tick([]) is None
    # A real value next must STILL act as first-tick baseline,
    # because the empty tick shouldn't have advanced state.
    assert sensor.tick([_measurement(0.01)]) is None
    assert ledger.recent() == []


def test_measurements_with_no_value_are_ignored(ledger):
    sensor = ObjectiveSensor(_objective(), ledger)
    blank = ObjectiveMeasurement(
        objective_id="cost_per_successful_completion",
        value=None,
        unit="USD",
        sample_size=100,
        window_start="2026-04-01T00:00:00Z",
        window_end="2026-04-01T00:00:00Z",
        computed_at="2026-04-01T00:00:00Z",
    )
    assert sensor.tick([blank]) is None


# ---------------------------------------------------------------------------
# Regression detection — lower_is_better
# ---------------------------------------------------------------------------


def test_regression_beyond_threshold_emits_signal(ledger):
    sensor = ObjectiveSensor(_objective(regression_pct=5.0), ledger)
    sensor.tick([_measurement(0.010)])  # baseline
    signal = sensor.tick([_measurement(0.013)])  # +30% (worse)

    assert signal is not None
    assert signal.type == "signal"
    assert signal.objective_id == "cost_per_successful_completion"
    assert signal.agent == "objective_sensor"
    assert SENSOR_TAG_REGRESSION in signal.tags
    assert signal.data["delta_pct"] == pytest.approx(30, rel=1e-3)
    assert signal.data["threshold_pct"] == 5.0
    assert signal.data["current"] == pytest.approx(0.013, rel=1e-6)
    assert signal.data["baseline"] == pytest.approx(0.010, rel=1e-6)


def test_regression_within_threshold_is_silent(ledger):
    sensor = ObjectiveSensor(_objective(regression_pct=10.0), ledger)
    sensor.tick([_measurement(0.010)])
    out = sensor.tick([_measurement(0.0105)])  # +5%, under threshold
    assert out is None
    assert ledger.recent() == []


def test_improvement_does_not_emit_signal(ledger):
    sensor = ObjectiveSensor(_objective(), ledger)
    sensor.tick([_measurement(0.010)])
    out = sensor.tick([_measurement(0.005)])  # halved — great
    assert out is None


# ---------------------------------------------------------------------------
# Direction sensitivity — higher_is_better inverts regression polarity
# ---------------------------------------------------------------------------


def test_higher_is_better_regression_on_decrease(ledger):
    obj = _objective(
        objective_id="domain_coverage_ratio",
        direction="higher_is_better",
        regression_pct=5.0,
    )
    sensor = ObjectiveSensor(obj, ledger)
    sensor.tick([_measurement(0.90)])
    signal = sensor.tick([_measurement(0.80)])  # dropped 11% — regression

    assert signal is not None
    assert signal.data["delta_pct"] == pytest.approx(11.11, rel=1e-2)


def test_higher_is_better_improvement_on_increase(ledger):
    obj = _objective(
        direction="higher_is_better",
        regression_pct=5.0,
    )
    sensor = ObjectiveSensor(obj, ledger)
    sensor.tick([_measurement(0.80)])
    out = sensor.tick([_measurement(0.90)])  # went up — improvement
    assert out is None


# ---------------------------------------------------------------------------
# Aggregation — multi-dimension measurements collapse by sample-size weight
# ---------------------------------------------------------------------------


def test_aggregation_is_sample_size_weighted(ledger):
    sensor = ObjectiveSensor(_objective(regression_pct=5.0), ledger)

    # Baseline: weighted mean = (0.01*200 + 0.05*50) / 250 = 0.018
    sensor.tick([
        _measurement(0.01, sample_size=200, dims={"model": "a"}),
        _measurement(0.05, sample_size=50, dims={"model": "b"}),
    ])
    # Next tick: weighted mean = (0.015*200 + 0.06*50) / 250 = 0.024 → ~33% worse
    signal = sensor.tick([
        _measurement(0.015, sample_size=200, dims={"model": "a"}),
        _measurement(0.06, sample_size=50, dims={"model": "b"}),
    ])
    assert signal is not None
    assert signal.data["delta_pct"] > 5.0
    assert signal.data["measurement_count"] == 2


def test_measurements_without_samples_do_not_poison_baseline(ledger):
    sensor = ObjectiveSensor(_objective(), ledger)
    # First tick: only sample-size=0 measurements → ignored entirely.
    assert sensor.tick([_measurement(9999.0, sample_size=0)]) is None
    # Next real tick must act as the first baseline.
    assert sensor.tick([_measurement(0.01)]) is None


# ---------------------------------------------------------------------------
# Missing guardrail = silent objective
# ---------------------------------------------------------------------------


def test_objective_without_regression_guardrail_emits_nothing(ledger):
    obj = Objective(
        id="domain_coverage_ratio",
        description="no guardrail",
        compute_fn="x:y",
        unit="ratio",
        direction="higher_is_better",
        guardrails=[],
    )
    sensor = ObjectiveSensor(obj, ledger)
    sensor.tick([_measurement(0.9)])
    out = sensor.tick([_measurement(0.1)])  # massive regression
    assert out is None  # no threshold = silence, by contract
