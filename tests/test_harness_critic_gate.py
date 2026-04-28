"""Unit tests for the critic gate.

We never reach a real LLM — every test passes a stub runner that
returns a canned verdict. The point is to lock down the audit-trail
guarantees: every check writes exactly one decision row to the
ledger, cache hits also write a row (tagged `cached_verdict`), and a
runner exception falls back to reject without crashing the caller.
"""

from __future__ import annotations

import pytest

from opentracy.harness import critic_gate
from opentracy.harness.critic_gate import critic_check
from opentracy.harness.ledger import LedgerStore


@pytest.fixture
def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


class _StubRunner:
    """Minimal AgentRunner-shaped stub. `verdict` may be a dict, an
    exception instance to raise, or a callable that returns a dict."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    async def run(self, agent_name: str, user_input: str):
        self.calls.append((agent_name, user_input))
        v = self.verdict() if callable(self.verdict) else self.verdict
        if isinstance(v, Exception):
            raise v

        class _R:
            def __init__(self, data):
                self.data = data

        return _R(v)


def _approve_verdict(cost: float = 0.42) -> dict:
    return {
        "decision": "approve",
        "rationale": "approved for test",
        "estimated_cost_usd": cost,
        "estimated_benefit": "test benefit",
    }


def _reject_verdict() -> dict:
    return {
        "decision": "reject",
        "rationale": "too expensive for the test",
        "estimated_cost_usd": 99.0,
        "estimated_benefit": "",
    }


# ---------------------------------------------------------------------------
# Approve / reject branches
# ---------------------------------------------------------------------------


async def test_approve_writes_one_decision_row(ledger):
    runner = _StubRunner(_approve_verdict())
    cache: dict = {}
    verdict = await critic_check(
        action_kind="run_agent",
        payload={"name": "training_advisor"},
        objective_id="cost_per_successful_completion",
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    assert verdict.approved is True
    assert verdict.estimated_cost_usd == 0.42

    rows = ledger.recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.type == "decision"
    assert row.agent == "budget_justifier"
    assert "critic_check" in row.tags
    assert "approve" in row.tags
    assert "cached_verdict" not in row.tags
    assert row.id == verdict.decision_entry_id


async def test_reject_branch_records_decision(ledger):
    runner = _StubRunner(_reject_verdict())
    cache: dict = {}
    verdict = await critic_check(
        action_kind="propose_action",
        payload={"kind": "queue_training"},
        objective_id="cost_per_successful_completion",
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    assert verdict.approved is False
    rows = ledger.recent(limit=10)
    assert len(rows) == 1
    assert "reject" in rows[0].tags


# ---------------------------------------------------------------------------
# Cache semantics
# ---------------------------------------------------------------------------


async def test_cache_hit_skips_runner_but_still_writes_row(ledger):
    runner = _StubRunner(_approve_verdict())
    cache: dict = {}
    payload = {"name": "training_advisor", "input": "x"}

    first = await critic_check(
        action_kind="run_agent",
        payload=payload,
        objective_id="cost_per_successful_completion",
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    assert len(runner.calls) == 1

    second = await critic_check(
        action_kind="run_agent",
        payload=payload,
        objective_id="cost_per_successful_completion",
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    # Runner was NOT called a second time — cache hit.
    assert len(runner.calls) == 1
    assert second.approved is True
    assert second.decision_entry_id != first.decision_entry_id

    rows = ledger.recent(limit=10)
    assert len(rows) == 2
    cached_row = next(r for r in rows if r.id == second.decision_entry_id)
    assert "cached_verdict" in cached_row.tags
    fresh_row = next(r for r in rows if r.id == first.decision_entry_id)
    assert "cached_verdict" not in fresh_row.tags


async def test_different_payload_does_not_collide_in_cache(ledger):
    runner = _StubRunner(_approve_verdict())
    cache: dict = {}
    await critic_check(
        action_kind="run_agent",
        payload={"name": "a"},
        objective_id="o",
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    await critic_check(
        action_kind="run_agent",
        payload={"name": "b"},
        objective_id="o",
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    # Two distinct payloads → two real runner calls.
    assert len(runner.calls) == 2


# ---------------------------------------------------------------------------
# Defensive: parse failures, exceptions
# ---------------------------------------------------------------------------


async def test_runner_exception_falls_back_to_reject(ledger):
    runner = _StubRunner(RuntimeError("network blip"))
    cache: dict = {}
    verdict = await critic_check(
        action_kind="run_agent",
        payload={"name": "x"},
        objective_id=None,
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    assert verdict.approved is False
    assert "critic unavailable" in verdict.rationale
    rows = ledger.recent(limit=10)
    assert len(rows) == 1
    assert "reject" in rows[0].tags


async def test_unknown_decision_string_is_treated_as_reject(ledger):
    runner = _StubRunner({
        "decision": "maybe",
        "rationale": "hmm",
        "estimated_cost_usd": "not-a-number",
        "estimated_benefit": "?",
    })
    cache: dict = {}
    verdict = await critic_check(
        action_kind="run_agent",
        payload={"x": 1},
        objective_id=None,
        ledger=ledger,
        runner=runner,
        cache=cache,
    )
    assert verdict.approved is False
    assert verdict.estimated_cost_usd == 0.0


def test_reset_cache_for_tests_is_callable():
    # Just smoke-tests the helper exists and runs without error.
    critic_gate.reset_cache_for_tests()
