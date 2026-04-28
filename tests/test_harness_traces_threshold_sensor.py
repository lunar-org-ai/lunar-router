"""Tests for NewTracesThresholdSensor.

Independent of ClickHouse — tests inject a stub `count_fn`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from opentracy.harness.ledger import LedgerStore
from opentracy.harness.objectives.schemas import GuardrailSpec, Objective
from opentracy.harness.sensors import (
    NewTracesThresholdSensor,
    SENSOR_TAG_TRACES_THRESHOLD,
)


def _objective() -> Objective:
    return Objective(
        id="domain_coverage_ratio",
        description="",
        compute_fn="x:y",
        unit="ratio",
        direction="higher_is_better",
        guardrails=[GuardrailSpec(type="no_regression_worse_than_pct", threshold=5.0)],
    )


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Threshold crossing
# ---------------------------------------------------------------------------


def test_fires_when_count_exceeds_threshold(ledger):
    sensor = NewTracesThresholdSensor(
        _objective(), ledger, threshold=100,
        count_fn=lambda since: 150,
    )
    entry = sensor.tick()
    assert entry is not None
    assert entry.type == "signal"
    assert SENSOR_TAG_TRACES_THRESHOLD in entry.tags
    assert entry.data["trace_count"] == 150
    assert entry.data["threshold"] == 100


def test_silent_when_below_threshold(ledger):
    sensor = NewTracesThresholdSensor(
        _objective(), ledger, threshold=100,
        count_fn=lambda since: 10,
    )
    assert sensor.tick() is None
    assert ledger.recent() == []


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


def test_cooldown_prevents_rapid_refire(ledger):
    calls = []

    def count_fn(since):
        calls.append(since)
        return 500

    sensor = NewTracesThresholdSensor(
        _objective(), ledger, threshold=100, cooldown_hours=6.0,
        count_fn=count_fn,
    )
    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    first = sensor.tick(now=t0)
    second = sensor.tick(now=t0 + timedelta(hours=1))
    assert first is not None
    assert second is None


def test_fires_again_after_cooldown_elapses(ledger):
    sensor = NewTracesThresholdSensor(
        _objective(), ledger, threshold=100, cooldown_hours=6.0,
        count_fn=lambda since: 500,
    )
    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    first = sensor.tick(now=t0)
    second = sensor.tick(now=t0 + timedelta(hours=6, minutes=1))
    assert first is not None
    assert second is not None
    assert first.id != second.id


# ---------------------------------------------------------------------------
# Window anchoring — after firing, the next count window starts fresh.
# ---------------------------------------------------------------------------


def test_window_anchor_advances_after_fire(ledger):
    received_since: list = []

    def count_fn(since):
        received_since.append(since)
        # First call: simulate 500 traces since anchor; second: simulate 200
        # that accumulated after the fire moment.
        return 500 if len(received_since) == 1 else 200

    sensor = NewTracesThresholdSensor(
        _objective(), ledger, threshold=100, cooldown_hours=0.1,
        count_fn=count_fn,
    )
    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    sensor.tick(now=t0)
    # After cooldown, the sensor should query CH with `since=t0`, not
    # the original construction time.
    sensor.tick(now=t0 + timedelta(hours=1))
    assert received_since[1] >= t0


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_count_fn_exception_returns_none(ledger):
    def boom(since):
        raise RuntimeError("ch down")

    sensor = NewTracesThresholdSensor(
        _objective(), ledger, threshold=100,
        count_fn=boom,
    )
    assert sensor.tick() is None
    assert ledger.recent() == []
