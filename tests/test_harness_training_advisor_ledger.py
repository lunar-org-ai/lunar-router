"""Ledger wiring tests for TrainingAdvisor.check_heuristics.

Applies the same observability contract as TraceScanner: every advisor
run must land a `run` + triggered `observation`s + final `decision`
entry in the ledger, and the chain must reconstruct from the run id.

These tests stay sync because `check_heuristics` is sync — no network,
no LLM, no async machinery involved.
"""

from __future__ import annotations

import pytest

from opentracy.harness.ledger import LedgerStore
from opentracy.harness.memory_store import MemoryStore
from opentracy.harness.training_advisor import (
    SIGNAL_TO_OBJECTIVE,
    TrainingAdvisor,
)


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


@pytest.fixture
def memory(tmp_path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path / "memory")


@pytest.fixture
def advisor(ledger, memory) -> TrainingAdvisor:
    return TrainingAdvisor(memory_store=memory, ledger=ledger)


# ---------------------------------------------------------------------------
# Heuristic input shapes — small helpers to keep each test focused.
# ---------------------------------------------------------------------------


def _inputs_triggering_train_now() -> dict:
    """Error rate jump + high-severity issues → 2 signals triggered →
    heuristic layer returns `train_now` with confidence ≥ 0.7."""
    return dict(
        error_rates={"gpt-4o-mini": 0.20},
        baseline_error_rates={"gpt-4o-mini": 0.05},
        high_severity_issues=5,
        drift_ratio=1.0,
        trace_count=500,
        hours_since_last_training=72.0,
    )


def _inputs_all_clear() -> dict:
    """Everything within bounds → no triggered signals → `wait`."""
    return dict(
        error_rates={"gpt-4o-mini": 0.04},
        baseline_error_rates={"gpt-4o-mini": 0.05},
        high_severity_issues=0,
        drift_ratio=1.0,
        trace_count=500,
        hours_since_last_training=72.0,
    )


def _inputs_insufficient_data() -> dict:
    """Too few traces → `wait` (data-gate path) even though issues trigger."""
    return dict(
        error_rates={"gpt-4o-mini": 0.20},
        baseline_error_rates={"gpt-4o-mini": 0.05},
        high_severity_issues=5,
        drift_ratio=1.0,
        trace_count=50,  # below MIN_TRACES_FOR_TRAINING
        hours_since_last_training=72.0,
    )


# ---------------------------------------------------------------------------
# Happy path — train_now produces run + obs + decision.
# ---------------------------------------------------------------------------


def test_train_now_emits_full_chain(advisor, ledger):
    rec = advisor.check_heuristics(**_inputs_triggering_train_now())
    assert rec.recommendation == "train_now"

    # Exactly one run in the ledger. Grab it and walk the chain.
    recent = ledger.recent(limit=50)
    runs = [e for e in recent if e.type == "run" and e.agent == "training_advisor"]
    assert len(runs) == 1
    run_id = runs[0].id

    chain = ledger.chain(run_id)
    types = [e.type for e in chain]
    assert types.count("run") == 1
    assert types.count("decision") == 1
    # 2 triggered signals → 2 observations
    observations = [e for e in chain if e.type == "observation"]
    assert len(observations) == 2


def test_decision_entry_carries_outcome_ok(advisor, ledger):
    advisor.check_heuristics(**_inputs_triggering_train_now())

    recent = ledger.recent(limit=50)
    decisions = [e for e in recent if e.type == "decision"]
    assert len(decisions) == 1
    d = decisions[0]
    assert d.outcome == "ok"
    assert d.data["recommendation"] == "train_now"
    assert d.data["source"] == "heuristic"
    assert d.duration_ms is not None


def test_triggered_observations_carry_objective_id(advisor, ledger):
    advisor.check_heuristics(**_inputs_triggering_train_now())

    recent = ledger.recent(limit=50)
    observations = [e for e in recent if e.type == "observation"]
    names = {o.data["signal_name"] for o in observations}
    # error_rate_increase + high_severity_issues are the two that triggered.
    assert names == {"error_rate_increase", "high_severity_issues"}
    # Both map to domain_coverage_ratio per SIGNAL_TO_OBJECTIVE.
    assert all(o.objective_id == "domain_coverage_ratio" for o in observations)
    # Each observation must chain back to the run.
    run_ids = {o.parent_id for o in observations}
    assert len(run_ids) == 1


# ---------------------------------------------------------------------------
# Deferred paths — wait/investigate land as skipped decisions.
# ---------------------------------------------------------------------------


def test_all_clear_emits_decision_with_outcome_skipped(advisor, ledger):
    rec = advisor.check_heuristics(**_inputs_all_clear())
    assert rec.recommendation == "wait"

    recent = ledger.recent(limit=50)
    decisions = [e for e in recent if e.type == "decision"]
    assert len(decisions) == 1
    assert decisions[0].outcome == "skipped"
    # No triggered signals → zero observations
    observations = [e for e in recent if e.type == "observation"]
    assert observations == []


def test_insufficient_data_still_emits_triggered_signals(advisor, ledger):
    """Even when the data-gate short-circuits to `wait`, the two error
    signals were still observed — they must land in the ledger so the
    objective time-series captures the degradation."""
    rec = advisor.check_heuristics(**_inputs_insufficient_data())
    assert rec.recommendation == "wait"
    assert "Insufficient traces" in rec.reason

    recent = ledger.recent(limit=50)
    observations = [e for e in recent if e.type == "observation"]
    names = {o.data["signal_name"] for o in observations}
    assert names == {"error_rate_increase", "high_severity_issues"}


# ---------------------------------------------------------------------------
# Signal-to-objective mapping is exhaustive for the known signals.
# ---------------------------------------------------------------------------


def test_signal_to_objective_covers_all_emitted_signals(advisor, ledger):
    """If a new signal is ever added to check_heuristics without a
    SIGNAL_TO_OBJECTIVE entry, .get() returns None silently. This test
    asserts we never ship such drift by enumerating the signals the
    current implementation produces."""
    rec = advisor.check_heuristics(**_inputs_triggering_train_now())
    for s in rec.signals:
        assert s.name in SIGNAL_TO_OBJECTIVE, (
            f"Signal {s.name!r} has no SIGNAL_TO_OBJECTIVE mapping — "
            f"update training_advisor.py."
        )


# ---------------------------------------------------------------------------
# Backwards compatibility — no ledger = no writes.
# ---------------------------------------------------------------------------


def test_advisor_without_ledger_emits_nothing(memory, ledger):
    a = TrainingAdvisor(memory_store=memory)  # no ledger
    a.check_heuristics(**_inputs_triggering_train_now())

    # The fixture ledger is a separate instance — should see nothing.
    assert ledger.recent(limit=10) == []
