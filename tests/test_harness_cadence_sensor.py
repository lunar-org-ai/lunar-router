"""Tests for CadenceSensor — the schedule-driven complement to
ObjectiveSensor. Fires on wall-clock elapsed time, not on regression.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from opentracy.harness.ledger import LedgerStore
from opentracy.harness.objectives.schemas import GuardrailSpec, Objective
from opentracy.harness.sensors import (
    SENSOR_TAG_CADENCE,
    CadenceSensor,
)


def _objective(cadence: str = "daily") -> Objective:
    return Objective(
        id="cost_per_successful_completion",
        description="",
        compute_fn="x:y",
        unit="USD",
        direction="lower_is_better",
        update_cadence=cadence,
        guardrails=[GuardrailSpec(type="no_regression_worse_than_pct", threshold=5.0)],
    )


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


def test_first_tick_fires_immediately(ledger):
    sensor = CadenceSensor(_objective(), ledger, interval_hours=24)
    entry = sensor.tick()
    assert entry is not None
    assert entry.type == "signal"
    assert SENSOR_TAG_CADENCE in entry.tags
    assert entry.objective_id == "cost_per_successful_completion"
    assert entry.agent == "cadence_sensor"


def test_second_tick_within_interval_is_silent(ledger):
    sensor = CadenceSensor(_objective(), ledger, interval_hours=24)
    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    first = sensor.tick(now=t0)
    second = sensor.tick(now=t0 + timedelta(hours=1))
    assert first is not None
    assert second is None


def test_tick_after_interval_fires_again(ledger):
    sensor = CadenceSensor(_objective(), ledger, interval_hours=24)
    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    first = sensor.tick(now=t0)
    second = sensor.tick(now=t0 + timedelta(hours=24, minutes=1))
    assert first is not None
    assert second is not None
    assert first.id != second.id


def test_signal_data_carries_interval_metadata(ledger):
    sensor = CadenceSensor(_objective(cadence="hourly"), ledger, interval_hours=6)
    entry = sensor.tick()
    assert entry.data["interval_hours"] == 6
    assert entry.data["cadence"] == "hourly"


def test_sub_hour_interval_respected(ledger):
    sensor = CadenceSensor(_objective(), ledger, interval_hours=0.5)
    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    first = sensor.tick(now=t0)
    assert first is not None
    assert sensor.tick(now=t0 + timedelta(minutes=10)) is None
    assert sensor.tick(now=t0 + timedelta(minutes=31)) is not None
