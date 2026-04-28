"""Round-trip tests for the harness ledger store."""

from __future__ import annotations

import pytest

from opentracy.harness.ledger import LedgerEntry, LedgerStore


@pytest.fixture
def store(tmp_path):
    s = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield s
    s.close()


def test_append_and_get(store):
    entry = LedgerEntry(
        type="signal",
        objective_id="cost_per_successful_completion",
        data={"reason": "cost drift detected", "delta_pct": 22.3},
    )
    store.append(entry)

    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.id == entry.id
    assert fetched.type == "signal"
    assert fetched.objective_id == "cost_per_successful_completion"
    assert fetched.data == {"reason": "cost drift detected", "delta_pct": 22.3}


def test_get_unknown_id_returns_none(store):
    assert store.get("does-not-exist") is None


def test_chain_reconstructs_causation(store):
    signal = LedgerEntry(type="signal", objective_id="p95_latency_ms")
    store.append(signal)

    run = LedgerEntry(type="run", parent_id=signal.id, agent="training_advisor")
    store.append(run)

    decision = LedgerEntry(type="decision", parent_id=run.id, outcome="ok")
    store.append(decision)

    action = LedgerEntry(type="action", parent_id=decision.id, outcome="ok")
    store.append(action)

    chain = store.chain(signal.id)
    assert [e.id for e in chain] == [signal.id, run.id, decision.id, action.id]
    assert [e.type for e in chain] == ["signal", "run", "decision", "action"]


def test_chain_handles_multiple_children(store):
    root = LedgerEntry(type="signal")
    store.append(root)
    child_a = LedgerEntry(type="run", parent_id=root.id, agent="a")
    child_b = LedgerEntry(type="run", parent_id=root.id, agent="b")
    store.append(child_a)
    store.append(child_b)

    chain = store.chain(root.id)
    ids = {e.id for e in chain}
    assert ids == {root.id, child_a.id, child_b.id}


def test_time_series_filters_by_objective(store):
    for _ in range(5):
        store.append(
            LedgerEntry(type="run", objective_id="cost_per_successful_completion")
        )
    store.append(LedgerEntry(type="run", objective_id="p95_latency_ms"))

    series = store.time_series("cost_per_successful_completion")
    assert len(series) == 5
    assert all(e.objective_id == "cost_per_successful_completion" for e in series)


def test_recent_returns_newest_first(store):
    ids = []
    for _ in range(3):
        e = LedgerEntry(type="run")
        store.append(e)
        ids.append(e.id)

    recent = store.recent(limit=2)
    assert len(recent) == 2
    assert recent[0].id == ids[-1]
    assert recent[1].id == ids[-2]


def test_data_and_tags_round_trip_as_structured(store):
    entry = LedgerEntry(
        type="observation",
        agent="trace_scanner",
        data={"nested": {"count": 3, "items": [1, 2, 3]}},
        tags=["drift", "cost"],
    )
    store.append(entry)
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.data == {"nested": {"count": 3, "items": [1, 2, 3]}}
    assert fetched.tags == ["drift", "cost"]
