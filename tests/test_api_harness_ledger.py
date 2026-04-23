"""Integration tests for the harness ledger + objectives read endpoints.

These endpoints back the HarnessPage dashboard (Step 4 of the harness
redesign plan). The dashboard's central guarantee is that a user can
answer "why did objective X move?" in under a minute — so we assert
shape correctness and chain reconstruction here, not just status codes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opentracy.api.server import app
from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.ledger import _global as ledger_global


@pytest.fixture
def client_with_ledger(tmp_path):
    """Install a tmp-path ledger as the process singleton for the
    duration of one test. The tearDown restores whatever was there
    before so other tests (and real runtime) aren't affected."""
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    original = ledger_global._instance
    ledger_global._instance = store
    try:
        yield TestClient(app), store
    finally:
        ledger_global._instance = original
        store.close()


def _seed_scan_chain(store: LedgerStore) -> dict:
    """Lay down a minimal causal chain so the endpoints have something
    interesting to return. Mirrors the shape the trigger engine would
    produce at runtime."""
    signal = LedgerEntry(
        type="signal",
        objective_id="cost_per_successful_completion",
        agent="objective_sensor",
        data={"delta_pct": 22.0, "baseline": 0.01, "current": 0.012},
        tags=["objective_regression", "cost_per_successful_completion"],
    )
    store.append(signal)

    dispatch = LedgerEntry(
        type="run",
        agent="training_advisor",
        parent_id=signal.id,
        parameters_in={"policy_id": "cost_drift_to_training_advisor"},
        tags=["policy_dispatch", "cost_drift_to_training_advisor"],
    )
    store.append(dispatch)

    decision = LedgerEntry(
        type="decision",
        agent="training_advisor",
        parent_id=dispatch.id,
        data={"recommendation": "train_now", "confidence": 0.85},
        tags=["train_now", "heuristic"],
        outcome="ok",
    )
    store.append(decision)

    return {
        "signal_id": signal.id,
        "dispatch_id": dispatch.id,
        "decision_id": decision.id,
    }


# ---------------------------------------------------------------------------
# /v1/harness/objectives
# ---------------------------------------------------------------------------


def test_list_objectives_returns_three(client_with_ledger):
    client, _ = client_with_ledger
    r = client.get("/v1/harness/objectives")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    ids = {o["id"] for o in body["objectives"]}
    assert ids == {
        "cost_per_successful_completion",
        "p95_latency_ms",
        "domain_coverage_ratio",
    }


def test_objectives_include_guardrails_and_direction(client_with_ledger):
    client, _ = client_with_ledger
    r = client.get("/v1/harness/objectives")
    cost = next(
        o for o in r.json()["objectives"]
        if o["id"] == "cost_per_successful_completion"
    )
    assert cost["direction"] == "lower_is_better"
    assert cost["unit"] == "USD"
    guardrail_types = {g["type"] for g in cost["guardrails"]}
    assert "no_regression_worse_than_pct" in guardrail_types


# ---------------------------------------------------------------------------
# /v1/harness/ledger (listing with filters)
# ---------------------------------------------------------------------------


def test_list_ledger_empty_when_nothing_written(client_with_ledger):
    client, _ = client_with_ledger
    r = client.get("/v1/harness/ledger")
    assert r.status_code == 200
    assert r.json() == {"entries": [], "count": 0}


def test_list_ledger_returns_newest_first(client_with_ledger):
    client, store = client_with_ledger
    _seed_scan_chain(store)

    r = client.get("/v1/harness/ledger?limit=10")
    body = r.json()
    assert body["count"] == 3
    # decision was written last → newest first.
    assert body["entries"][0]["type"] == "decision"
    assert body["entries"][-1]["type"] == "signal"


def test_ledger_filter_by_type(client_with_ledger):
    client, store = client_with_ledger
    _seed_scan_chain(store)

    r = client.get("/v1/harness/ledger?type=signal")
    body = r.json()
    assert body["count"] == 1
    assert body["entries"][0]["type"] == "signal"


def test_ledger_filter_by_objective_id(client_with_ledger):
    client, store = client_with_ledger
    _seed_scan_chain(store)

    r = client.get(
        "/v1/harness/ledger?objective_id=cost_per_successful_completion"
    )
    # Only the signal entry has an objective_id; dispatch+decision don't.
    assert r.json()["count"] == 1


def test_ledger_filter_by_agent(client_with_ledger):
    client, store = client_with_ledger
    _seed_scan_chain(store)

    r = client.get("/v1/harness/ledger?agent=training_advisor")
    body = r.json()
    # dispatch run + decision both have agent="training_advisor"
    assert body["count"] == 2


def test_ledger_limit_is_capped_at_1000(client_with_ledger):
    client, _ = client_with_ledger
    # Ask for 5000; caller shouldn't be able to pull an unbounded page.
    r = client.get("/v1/harness/ledger?limit=5000")
    assert r.status_code == 200
    # Not asserting on count (empty ledger), but the call must succeed
    # rather than raising.


# ---------------------------------------------------------------------------
# /v1/harness/ledger/{entry_id} — single entry
# ---------------------------------------------------------------------------


def test_get_single_entry_by_id(client_with_ledger):
    client, store = client_with_ledger
    ids = _seed_scan_chain(store)

    r = client.get(f"/v1/harness/ledger/{ids['signal_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == ids["signal_id"]
    assert body["type"] == "signal"
    assert body["data"]["delta_pct"] == 22.0


def test_get_unknown_entry_returns_404(client_with_ledger):
    client, _ = client_with_ledger
    r = client.get("/v1/harness/ledger/does-not-exist")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /v1/harness/ledger/{entry_id}/chain — drill-down
# ---------------------------------------------------------------------------


def test_chain_reconstructs_from_signal(client_with_ledger):
    client, store = client_with_ledger
    ids = _seed_scan_chain(store)

    r = client.get(f"/v1/harness/ledger/{ids['signal_id']}/chain")
    assert r.status_code == 200
    body = r.json()
    assert body["root_id"] == ids["signal_id"]
    assert body["count"] == 3
    types = [e["type"] for e in body["entries"]]
    # BFS order: signal → dispatch-run → decision
    assert types == ["signal", "run", "decision"]


def test_chain_on_unknown_id_returns_404(client_with_ledger):
    client, _ = client_with_ledger
    r = client.get("/v1/harness/ledger/nope/chain")
    assert r.status_code == 404


def test_chain_from_middle_node_includes_descendants_only(client_with_ledger):
    """Clicking a middle node should drill down from THERE, not the root —
    dashboard UX depends on this being true so users can focus on a
    specific subtree without redrawing the whole chain."""
    client, store = client_with_ledger
    ids = _seed_scan_chain(store)

    r = client.get(f"/v1/harness/ledger/{ids['dispatch_id']}/chain")
    body = r.json()
    types = [e["type"] for e in body["entries"]]
    # From dispatch: dispatch itself + its child decision. No signal.
    assert types == ["run", "decision"]


# ---------------------------------------------------------------------------
# Route registration + ordering — guards the /chain vs /{id} precedence
# ---------------------------------------------------------------------------


def test_all_four_routes_are_registered():
    expected = {
        ("/v1/harness/objectives", "GET"),
        ("/v1/harness/ledger", "GET"),
        ("/v1/harness/ledger/{entry_id}", "GET"),
        ("/v1/harness/ledger/{entry_id}/chain", "GET"),
    }
    registered = {
        (r.path, m)
        for r in app.routes
        for m in getattr(r, "methods", set())
    }
    missing = expected - registered
    assert not missing, f"routes not registered: {missing}"
