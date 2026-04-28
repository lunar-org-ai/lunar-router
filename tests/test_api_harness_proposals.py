"""HTTP integration tests for /v1/harness/proposals/* routes.

Stubs the budget critic via monkeypatching `harness.writes.critic_check`
so the tests don't reach the LLM. Asserts route registration, status
codes, payload shapes, and the proposal lifecycle through the HTTP
surface that the UI consumes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opentracy.api.server import app
from opentracy.harness import critic_gate
from opentracy.harness import writes as harness_writes
from opentracy.harness.ledger import LedgerStore
from opentracy.harness.ledger import _global as ledger_global


@pytest.fixture
def client_with_ledger(tmp_path, monkeypatch):
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    original = ledger_global._instance
    ledger_global._instance = store

    # Stub the critic so writes don't reach the LLM. Test toggles the
    # decision via the closure variable.
    state = {"decision": "approve"}

    async def _stub_critic_check(*, action_kind, payload, objective_id, ledger=None, runner=None, cache=None):
        from opentracy.harness.critic_gate import CriticVerdict
        from opentracy.harness.ledger import LedgerEntry

        store_ = ledger if ledger is not None else ledger_global._instance
        decision = state["decision"]
        entry = LedgerEntry(
            type="decision",
            objective_id=objective_id,
            agent="budget_justifier",
            data={
                "decision": decision,
                "rationale": f"stub-{decision}",
                "estimated_cost_usd": 0.1,
                "estimated_benefit": "test",
                "action_kind": action_kind,
            },
            tags=["critic_check", action_kind, decision],
            outcome="ok",
        )
        store_.append(entry)
        return CriticVerdict(
            decision=decision,
            rationale=f"stub-{decision}",
            estimated_cost_usd=0.1,
            estimated_benefit="test",
            decision_entry_id=entry.id,
        )

    monkeypatch.setattr(harness_writes, "critic_check", _stub_critic_check)
    critic_gate.reset_cache_for_tests()

    try:
        yield TestClient(app), store, state
    finally:
        ledger_global._instance = original
        store.close()


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def test_proposals_routes_are_registered():
    expected = {
        ("/v1/harness/proposals", "GET"),
        ("/v1/harness/proposals", "POST"),
        ("/v1/harness/proposals/{proposal_id}", "GET"),
        ("/v1/harness/proposals/{proposal_id}/approve", "POST"),
        ("/v1/harness/proposals/{proposal_id}/reject", "POST"),
        ("/v1/harness/proposals/{proposal_id}/outcome", "POST"),
    }
    registered = {
        (r.path, m)
        for r in app.routes
        for m in getattr(r, "methods", set())
    }
    missing = expected - registered
    assert not missing, f"missing routes: {missing}"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_create_proposal_returns_id_and_verdict(client_with_ledger):
    client, _, _ = client_with_ledger
    r = client.post(
        "/v1/harness/proposals",
        json={
            "kind": "queue_training",
            "payload": {"student": "x"},
            "summary": "test",
            "objective_id": "cost_per_successful_completion",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "proposal_id" in body
    assert body["verdict"]["decision"] == "approve"


def test_list_proposals_returns_pending_first(client_with_ledger):
    client, _, _ = client_with_ledger
    for n in range(3):
        client.post(
            "/v1/harness/proposals",
            json={"kind": "run_eval", "payload": {"i": n}, "summary": str(n)},
        )

    r = client.get("/v1/harness/proposals?status=pending")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    for p in body["proposals"]:
        assert p["status"] == "pending"


def test_get_proposal_404_for_unknown(client_with_ledger):
    client, _, _ = client_with_ledger
    r = client.get("/v1/harness/proposals/does-not-exist")
    assert r.status_code == 404


def test_approve_then_record_outcome_changes_status(client_with_ledger):
    client, _, _ = client_with_ledger
    created = client.post(
        "/v1/harness/proposals",
        json={"kind": "run_eval", "payload": {"i": 1}, "summary": "x"},
    ).json()
    pid = created["proposal_id"]

    r = client.post(f"/v1/harness/proposals/{pid}/approve")
    assert r.status_code == 200
    assert r.json()["decision"] == "approved"

    after_approve = client.get(f"/v1/harness/proposals/{pid}").json()
    assert after_approve["status"] == "approved"

    r2 = client.post(
        f"/v1/harness/proposals/{pid}/outcome",
        json={"result": {"score": 0.9}, "outcome": "ok"},
    )
    assert r2.status_code == 200
    assert client.get(f"/v1/harness/proposals/{pid}").json()["status"] == "executed"


def test_reject_proposal_via_http(client_with_ledger):
    client, _, _ = client_with_ledger
    created = client.post(
        "/v1/harness/proposals",
        json={"kind": "run_eval", "payload": {"i": 1}, "summary": "x"},
    ).json()
    pid = created["proposal_id"]

    r = client.post(
        f"/v1/harness/proposals/{pid}/reject",
        json={"reason": "not now"},
    )
    assert r.status_code == 200
    assert client.get(f"/v1/harness/proposals/{pid}").json()["status"] == "rejected"


def test_approve_unknown_proposal_returns_404(client_with_ledger):
    client, _, _ = client_with_ledger
    r = client.post("/v1/harness/proposals/does-not-exist/approve")
    assert r.status_code == 404


def test_outcome_with_invalid_outcome_returns_400(client_with_ledger):
    client, _, _ = client_with_ledger
    created = client.post(
        "/v1/harness/proposals",
        json={"kind": "run_eval", "payload": {"i": 1}, "summary": "x"},
    ).json()
    pid = created["proposal_id"]
    client.post(f"/v1/harness/proposals/{pid}/approve")

    r = client.post(
        f"/v1/harness/proposals/{pid}/outcome",
        json={"result": {}, "outcome": "wat"},
    )
    assert r.status_code == 400


def test_unknown_status_filter_returns_400(client_with_ledger):
    client, _, _ = client_with_ledger
    r = client.get("/v1/harness/proposals?status=wat")
    assert r.status_code == 400


def test_critic_reject_status_is_rejected_by_critic(client_with_ledger):
    client, _, state = client_with_ledger
    state["decision"] = "reject"

    created = client.post(
        "/v1/harness/proposals",
        json={"kind": "run_eval", "payload": {"i": 1}, "summary": "x"},
    ).json()
    pid = created["proposal_id"]
    body = client.get(f"/v1/harness/proposals/{pid}").json()
    assert body["status"] == "rejected_by_critic"


def test_create_proposal_requires_kind(client_with_ledger):
    client, _, _ = client_with_ledger
    r = client.post(
        "/v1/harness/proposals",
        json={"payload": {}, "summary": "x"},
    )
    assert r.status_code == 400
