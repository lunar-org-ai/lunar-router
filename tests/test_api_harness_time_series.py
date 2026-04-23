"""Tests for the objective time-series endpoint.

This endpoint powers the Step 4 dashboard plot: one line per
objective with action markers overlaid on the x-axis. The test
asserts the shape the frontend reads and the split between
measurements (continuous, for the line) and markers (discrete events
worth drilling into).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from opentracy.api.server import app
from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.ledger import _global as ledger_global


OBJ = "cost_per_successful_completion"


@pytest.fixture
def client_with_ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    original = ledger_global._instance
    ledger_global._instance = store
    try:
        yield TestClient(app), store
    finally:
        ledger_global._instance = original
        store.close()


def _now_iso(offset_hours: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).isoformat()


def _seed(store: LedgerStore) -> dict:
    """Plant a realistic day in the ledger: measurements every hour,
    plus one signal and one decision/action chain near the end."""
    measurement_ids = []
    for i in range(6):
        m = LedgerEntry(
            type="observation",
            objective_id=OBJ,
            agent="objective_sensor",
            ts=_now_iso(offset_hours=-6 + i),
            data={"value": 0.010 + i * 0.001, "sample_size": 100},
            tags=["measurement", OBJ],
        )
        store.append(m)
        measurement_ids.append(m.id)

    signal = LedgerEntry(
        type="signal",
        objective_id=OBJ,
        agent="objective_sensor",
        ts=_now_iso(offset_hours=-0.5),
        data={"delta_pct": 22.0, "baseline": 0.010, "current": 0.012},
        tags=["objective_regression", OBJ],
    )
    store.append(signal)
    decision = LedgerEntry(
        type="decision",
        objective_id=OBJ,  # propagated from the signal so time-series indexing works
        parent_id=signal.id,
        agent="training_advisor",
        ts=_now_iso(offset_hours=-0.4),
        data={"recommendation": "train_now", "confidence": 0.85},
        tags=["train_now"],
        outcome="ok",
    )
    store.append(decision)

    return {
        "measurement_ids": measurement_ids,
        "signal_id": signal.id,
        "decision_id": decision.id,
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_time_series_returns_measurements_and_markers_split(client_with_ledger):
    client, store = client_with_ledger
    ids = _seed(store)

    r = client.get(f"/v1/harness/objectives/{OBJ}/time-series?hours=24")
    assert r.status_code == 200
    body = r.json()

    assert body["objective_id"] == OBJ
    assert body["window_hours"] == 24
    # 6 continuous measurements → line
    assert len(body["measurements"]) == 6
    # 1 signal + 1 decision → markers
    marker_types = sorted(m["type"] for m in body["markers"])
    assert marker_types == ["decision", "signal"]


def test_measurements_have_plot_ready_shape(client_with_ledger):
    client, store = client_with_ledger
    _seed(store)

    r = client.get(f"/v1/harness/objectives/{OBJ}/time-series?hours=24")
    measurement = r.json()["measurements"][0]
    assert {"ts", "value", "sample_size", "id"} <= set(measurement.keys())
    assert isinstance(measurement["value"], (int, float))


def test_markers_preserve_full_entry_for_drill_down(client_with_ledger):
    client, store = client_with_ledger
    ids = _seed(store)

    r = client.get(f"/v1/harness/objectives/{OBJ}/time-series?hours=24")
    markers = r.json()["markers"]
    signal = next(m for m in markers if m["type"] == "signal")
    # Full ledger entry shape — the UI uses entry.id to open the chain
    # drawer, so the id + parent_id + tags must all be present.
    assert signal["id"] == ids["signal_id"]
    assert "tags" in signal
    assert signal["data"]["delta_pct"] == 22.0


def test_window_hours_clamps_at_month(client_with_ledger):
    client, _ = client_with_ledger
    r = client.get(f"/v1/harness/objectives/{OBJ}/time-series?hours=999999")
    assert r.status_code == 200
    assert r.json()["window_hours"] == 24 * 30


def test_unknown_objective_returns_empty_data(client_with_ledger):
    """An unknown objective id is not an error — it's an empty
    time-series. Lets the UI show "no data yet" without special-casing
    404 handling across objectives."""
    client, _ = client_with_ledger
    r = client.get("/v1/harness/objectives/does_not_exist/time-series")
    assert r.status_code == 200
    body = r.json()
    assert body["measurements"] == []
    assert body["markers"] == []


def test_markers_exclude_runs_and_non_measurement_observations(client_with_ledger):
    """Runs (agent executions) and non-measurement observations should
    NOT land as markers — they'd drown out the signals/decisions/
    actions that actually changed the objective's trajectory."""
    client, store = client_with_ledger
    store.append(
        LedgerEntry(
            type="run", objective_id=OBJ, agent="trace_scanner",
            tags=["scheduler_tick"],
        )
    )
    store.append(
        LedgerEntry(
            type="observation", objective_id=OBJ, agent="trace_scanner",
            data={"issue_type": "latency_spike"}, tags=["latency_spike", "high"],
        )
    )
    r = client.get(f"/v1/harness/objectives/{OBJ}/time-series?hours=24")
    assert r.json()["markers"] == []
