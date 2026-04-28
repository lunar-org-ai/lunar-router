"""Unit tests for harness write paths.

Exercises run_agent_gated, the proposal lifecycle (propose → approve/
reject → record_outcome), and list_proposals status derivation.

Stub runner returns canned verdicts plus canned agent outputs — the
gate logic is what's under test, not the LLM.
"""

from __future__ import annotations

import pytest

from opentracy.harness import writes as harness_writes
from opentracy.harness.ledger import LedgerStore


@pytest.fixture
def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


class _StubRunner:
    """Returns budget_justifier verdict dicts for that agent and a
    different canned response for any other agent."""

    def __init__(self, verdict: dict, agent_output: dict | None = None):
        self.verdict = verdict
        self.agent_output = agent_output or {"recommendation": "train_now"}
        self.calls: list[tuple[str, str]] = []

    async def run(self, agent_name: str, user_input: str, *_args, **_kwargs):
        self.calls.append((agent_name, user_input))

        class _R:
            def __init__(self, data, agent):
                self.data = data
                self.agent = agent
                self.duration_ms = 1.0
                self.cost_usd = None

        if agent_name == "budget_justifier":
            return _R(self.verdict, agent_name)
        return _R(self.agent_output, agent_name)

    async def run_with_tools(self, agent_name: str, user_input: str):
        return await self.run(agent_name, user_input)


def _approve():
    return {
        "decision": "approve",
        "rationale": "ok",
        "estimated_cost_usd": 0.5,
        "estimated_benefit": "x",
    }


def _reject():
    return {
        "decision": "reject",
        "rationale": "too costly",
        "estimated_cost_usd": 99,
        "estimated_benefit": "",
    }


# ---------------------------------------------------------------------------
# run_agent_gated
# ---------------------------------------------------------------------------


async def test_inspector_skips_critic(ledger):
    runner = _StubRunner(_reject(), agent_output={"label": "JS Concepts"})
    out = await harness_writes.run_agent_gated(
        "cluster_labeler",
        "[prompt samples]",
        ledger=ledger,
        runner=runner,
    )
    assert out["decision"] == "ungated"
    assert out["result"] == {"label": "JS Concepts"}

    # Only the agent's run row should exist — no decision row.
    rows = ledger.recent(limit=10)
    assert {r.type for r in rows} == {"run"}
    assert "cluster_labeler" in [r.agent for r in rows]
    # Critic was not called.
    assert all(name != "budget_justifier" for name, _ in runner.calls)


async def test_proposer_with_approve_runs_and_writes_two_rows(ledger):
    runner = _StubRunner(_approve(), agent_output={"recommendation": "train_now"})
    out = await harness_writes.run_agent_gated(
        "training_advisor",
        "[regression]",
        objective_id="cost_per_successful_completion",
        ledger=ledger,
        runner=runner,
    )
    assert out["decision"] == "approved"
    assert out["result"] == {"recommendation": "train_now"}

    rows = ledger.recent(limit=10)
    types = sorted(r.type for r in rows)
    assert types == ["decision", "run"]
    # The run row's parent should be the decision row.
    decision_row = next(r for r in rows if r.type == "decision")
    run_row = next(r for r in rows if r.type == "run")
    assert run_row.parent_id == decision_row.id


async def test_proposer_with_reject_does_not_run_agent(ledger):
    runner = _StubRunner(_reject(), agent_output={"should": "not_appear"})
    out = await harness_writes.run_agent_gated(
        "training_advisor",
        "[bad idea]",
        objective_id="cost_per_successful_completion",
        ledger=ledger,
        runner=runner,
    )
    assert out["decision"] == "rejected"

    # Only the critic's decision row exists; the proposer was never invoked.
    rows = ledger.recent(limit=10)
    assert {r.type for r in rows} == {"decision"}
    assert all(name == "budget_justifier" for name, _ in runner.calls)


# ---------------------------------------------------------------------------
# Proposal lifecycle
# ---------------------------------------------------------------------------


async def test_propose_action_records_proposal_with_verdict(ledger):
    runner = _StubRunner(_approve())
    out = await harness_writes.propose_action(
        kind="queue_training",
        payload={"student": "llama-1b"},
        objective_id="cost_per_successful_completion",
        summary="train tiny student",
        runner=runner,
        ledger=ledger,
    )
    proposal_id = out["proposal_id"]

    proposal = ledger.get(proposal_id)
    assert proposal is not None
    assert proposal.type == "proposal"
    assert proposal.data["kind"] == "queue_training"
    assert proposal.data["summary"] == "train tiny student"
    assert proposal.data["verdict"]["decision"] == "approve"
    assert "critic_approve" in proposal.tags

    # And one decision (the critic check) parents the proposal.
    rows = ledger.recent(limit=10)
    decision = next(r for r in rows if r.type == "decision")
    assert proposal.parent_id == decision.id


async def test_propose_with_critic_reject_status_is_rejected_by_critic(ledger):
    runner = _StubRunner(_reject())
    out = await harness_writes.propose_action(
        kind="queue_training",
        payload={"student": "llama-70b"},
        objective_id="cost_per_successful_completion",
        summary="overkill",
        runner=runner,
        ledger=ledger,
    )
    pid = out["proposal_id"]
    listed = harness_writes.list_proposals(ledger=ledger)
    assert len(listed) == 1
    assert listed[0]["id"] == pid
    assert listed[0]["status"] == "rejected_by_critic"


async def test_approve_proposal_writes_resolution(ledger):
    runner = _StubRunner(_approve())
    proposed = await harness_writes.propose_action(
        kind="run_eval",
        payload={"dataset": "x"},
        objective_id=None,
        summary="ok",
        runner=runner,
        ledger=ledger,
    )
    pid = proposed["proposal_id"]

    decided = await harness_writes.approve_proposal(
        pid, runner=runner, ledger=ledger,
    )
    assert decided["decision"] == "approved"

    # Status should now be "approved" (executed requires record_outcome).
    proposal = harness_writes.get_proposal(pid, ledger=ledger)
    assert proposal is not None
    assert proposal["status"] == "approved"


async def test_approve_with_critic_recheck_rejection(ledger):
    """Critic flips between propose and approve → resolution row is
    'rejected', status becomes 'rejected'."""
    # First propose with an approving critic.
    approving_runner = _StubRunner(_approve())
    proposed = await harness_writes.propose_action(
        kind="run_eval",
        payload={"dataset": "x"},
        objective_id="cost_per_successful_completion",
        summary="ok at first",
        runner=approving_runner,
        ledger=ledger,
    )
    pid = proposed["proposal_id"]

    # New runner that now rejects. To bypass the verdict cache, mutate
    # the payload-implicit by waiting for the cache test isolation —
    # easier here: clear cache.
    from opentracy.harness import critic_gate
    critic_gate.reset_cache_for_tests()

    rejecting_runner = _StubRunner(_reject())
    decided = await harness_writes.approve_proposal(
        pid, runner=rejecting_runner, ledger=ledger,
    )
    assert decided["decision"] == "rejected"

    proposal = harness_writes.get_proposal(pid, ledger=ledger)
    assert proposal is not None
    assert proposal["status"] == "rejected"


async def test_reject_proposal_updates_status(ledger):
    runner = _StubRunner(_approve())
    proposed = await harness_writes.propose_action(
        kind="run_eval",
        payload={"dataset": "x"},
        objective_id=None,
        summary="ok",
        runner=runner,
        ledger=ledger,
    )
    pid = proposed["proposal_id"]

    out = harness_writes.reject_proposal(pid, reason="not now", ledger=ledger)
    assert out["decision"] == "rejected"

    listed = harness_writes.list_proposals(status="rejected", ledger=ledger)
    assert any(p["id"] == pid for p in listed)


async def test_record_outcome_marks_executed_then_failed(ledger):
    runner = _StubRunner(_approve())
    p1 = await harness_writes.propose_action(
        kind="run_eval", payload={"a": 1}, objective_id=None,
        summary="a", runner=runner, ledger=ledger,
    )
    await harness_writes.approve_proposal(
        p1["proposal_id"], runner=runner, ledger=ledger,
    )

    harness_writes.record_outcome(
        p1["proposal_id"],
        result={"score": 0.91, "cost_usd": 0.05},
        outcome="ok",
        ledger=ledger,
    )
    proposal = harness_writes.get_proposal(p1["proposal_id"], ledger=ledger)
    assert proposal["status"] == "executed"

    # Approve a second one, fail it.
    from opentracy.harness import critic_gate
    critic_gate.reset_cache_for_tests()
    p2 = await harness_writes.propose_action(
        kind="run_eval", payload={"a": 2}, objective_id=None,
        summary="b", runner=runner, ledger=ledger,
    )
    await harness_writes.approve_proposal(
        p2["proposal_id"], runner=runner, ledger=ledger,
    )
    harness_writes.record_outcome(
        p2["proposal_id"], result={"err": "x"}, outcome="failed", ledger=ledger,
    )
    proposal2 = harness_writes.get_proposal(p2["proposal_id"], ledger=ledger)
    assert proposal2["status"] == "failed"


async def test_double_resolution_raises(ledger):
    runner = _StubRunner(_approve())
    proposed = await harness_writes.propose_action(
        kind="run_eval", payload={"a": 1}, objective_id=None,
        summary="a", runner=runner, ledger=ledger,
    )
    pid = proposed["proposal_id"]
    harness_writes.reject_proposal(pid, ledger=ledger)
    with pytest.raises(ValueError):
        harness_writes.reject_proposal(pid, ledger=ledger)


def test_get_proposal_unknown_returns_none(ledger):
    assert harness_writes.get_proposal("does-not-exist", ledger=ledger) is None


def test_list_proposals_empty(ledger):
    assert harness_writes.list_proposals(ledger=ledger) == []
