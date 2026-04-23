"""Tests for TriggerEngineLoop — the async wrapper that drives the
engine in production.

These tests exercise the full end-to-end wiring Step 3 of the plan
requires: cadence sensor fires → policy matches → recipe dispatches
→ executor runs each step → every ledger row chains back to the
signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from opentracy.harness.ledger import LedgerStore
from opentracy.harness.ledger import _global as ledger_global
from opentracy.harness.triggers import TriggerEngineLoop


@dataclass
class _StubAgentResult:
    data: dict
    cost_usd: float = 0.01
    duration_ms: int = 5


class _StubRunner:
    """Returns a scripted result for each agent name. Any unscripted
    call raises so tests don't silently succeed on the wrong codepath."""

    def __init__(self, scripts: dict[str, Any]):
        self.scripts = scripts
        self.calls: list[str] = []

    async def run(self, agent_name: str, user_input: str):
        self.calls.append(agent_name)
        if agent_name not in self.scripts:
            raise AssertionError(f"unexpected agent call: {agent_name}")
        return self.scripts[agent_name]


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    # Install as singleton too — runner defaults to get_ledger_store()
    # when no ledger arg is passed, and some code paths may use it.
    original = ledger_global._instance
    ledger_global._instance = store
    yield store
    ledger_global._instance = original
    store.close()


@pytest.fixture
def stub_runner():
    """Scripts every agent the shipped recipe will call."""
    return _StubRunner({
        "trace_scanner": _StubAgentResult(
            data={"issues": [{"type": "cost_anomaly", "severity": "medium"}]},
            cost_usd=0.02,
        ),
        "eval_generator": _StubAgentResult(
            data={
                "eval_case": {
                    "input": "What is the cost of this call?",
                    "expected_behavior": "respond under $0.001",
                    "check_type": "cost_bound",
                    "severity": "medium",
                },
                "rationale": "pins the per-call cost to catch regressions",
            },
            cost_usd=0.01,
        ),
        "budget_justifier": _StubAgentResult(
            data={
                "decision": "approve",
                "rationale": "22% cost drift warrants the $0.03 investigation",
                "estimated_cost_usd": 0.03,
            },
            cost_usd=0.01,
        ),
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_loop_loads_yaml_config_on_construction(ledger, stub_runner):
    loop = TriggerEngineLoop(ledger=ledger, agent_runner=stub_runner)
    assert len(loop.objectives) == 3
    assert len(loop.policies) >= 3
    assert "trace_scan_and_evaluate" in loop._recipes_by_id


# ---------------------------------------------------------------------------
# End-to-end: cadence → policy → recipe → ledger chain
# ---------------------------------------------------------------------------


async def test_run_once_fires_cadence_and_runs_full_recipe(ledger, stub_runner):
    loop = TriggerEngineLoop(
        ledger=ledger,
        agent_runner=stub_runner,
        # 1h interval; first tick always fires (CadenceSensor contract).
        cadence_interval_hours={"hourly": 1.0, "daily": 24.0, "weekly": 168.0},
    )

    await loop.run_once()

    # At least one cadence signal should have fired — first-tick semantics.
    signals = [e for e in ledger.recent(limit=50) if e.type == "signal"]
    cadence_signals = [s for s in signals if "cadence" in s.tags]
    assert len(cadence_signals) >= 1

    # The cadence signal that matches the `cadence_to_trace_scan` policy
    # (any objective with a cadence tag) should have produced a dispatch
    # run chained to it, plus the recipe's agent/action steps.
    signal = cadence_signals[0]
    chain = ledger.chain(signal.id)
    types = [e.type for e in chain]
    assert types.count("signal") == 1
    # dispatch run + 3 agent runs + 1 action = 5 children → 6 total
    assert types.count("run") >= 3  # dispatch + 3 agent steps (lenient across objectives)
    assert types.count("action") >= 1

    # Every non-signal entry must trace back to the signal via parent_id
    # chain. `ledger.chain` guarantees this when it returns entries.
    assert all(
        e.parent_id is not None or e.id == signal.id
        for e in chain
    )


async def test_recipe_agents_get_called_in_order(ledger, stub_runner):
    loop = TriggerEngineLoop(ledger=ledger, agent_runner=stub_runner)
    await loop.run_once()

    # Expected call order: trace_scanner (inspector) → eval_generator
    # (proposer) → budget_justifier (critic). The action step doesn't
    # call the runner — it's a code action.
    #
    # Multiple objectives have cadence sensors, so each cadence signal
    # fires the same policy. We may see N*3 calls where N = number of
    # objectives whose cadence tag matches the policy. Assert ordering
    # within each triplet rather than absolute count.
    assert "trace_scanner" in stub_runner.calls
    assert "eval_generator" in stub_runner.calls
    assert "budget_justifier" in stub_runner.calls

    # The first triplet must be in order.
    first_triplet = stub_runner.calls[:3]
    assert first_triplet == ["trace_scanner", "eval_generator", "budget_justifier"]


async def test_action_queues_eval_when_critic_approves(ledger, stub_runner):
    loop = TriggerEngineLoop(ledger=ledger, agent_runner=stub_runner)
    await loop.run_once()

    actions = [e for e in ledger.recent(limit=50) if e.type == "action"]
    assert len(actions) >= 1
    assert all(a.agent == "run_eval" for a in actions)
    # queued status (not executed — MVP)
    assert any(a.data.get("status") == "queued" for a in actions)


async def test_critic_rejection_skips_action(ledger):
    """Regression guard: when the critic rejects, the action step must
    NOT produce a queued eval. The executor records it as skipped with
    tag `condition_unmet`."""
    rejecting_runner = _StubRunner({
        "trace_scanner": _StubAgentResult(data={"issues": [{"severity": "low"}]}),
        "eval_generator": _StubAgentResult(data={
            "eval_case": {
                "input": "x",
                "expected_behavior": "y",
                "check_type": "valid_json",
            },
        }),
        "budget_justifier": _StubAgentResult(data={
            "decision": "reject",
            "rationale": "2% drift is within the noise floor",
        }),
    })

    loop = TriggerEngineLoop(ledger=ledger, agent_runner=rejecting_runner)
    await loop.run_once()

    # No run_eval queued — all "action" entries should be skipped.
    actions = [e for e in ledger.recent(limit=100) if e.type == "action"]
    for a in actions:
        # Either the executor wrote a skipped action entry, or run_eval
        # was never called. Both are acceptable — ensure none are ok.
        assert a.outcome in ("skipped", "failed"), (
            f"action landed with outcome {a.outcome!r} — critic rejection "
            "should have prevented it"
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_start_and_stop_are_idempotent(ledger, stub_runner):
    loop = TriggerEngineLoop(
        ledger=ledger,
        agent_runner=stub_runner,
        interval_seconds=3600,  # won't actually tick twice in this test
    )
    await loop.start()
    # Double-start is a no-op
    await loop.start()
    await loop.stop()
    # Double-stop is a no-op
    await loop.stop()
