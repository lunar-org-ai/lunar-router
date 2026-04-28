"""Step 5.3 end-to-end: cost regression signal → evaluate_student_distillation
recipe → fetch_cost_summary + training_advisor + budget_justifier +
queue_training, all chained in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import opentracy.storage.clickhouse_client as _ch_module
from opentracy.harness.actions import get_action
from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.tasks import RecipeExecutor, load_recipe


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubAgentResult:
    data: dict
    cost_usd: float = 0.01
    duration_ms: int = 3


class _StubRunner:
    def __init__(self, scripts: dict[str, Any]):
        self.scripts = scripts
        self.calls: list[tuple[str, str]] = []

    async def run(self, agent_name: str, user_input: str):
        self.calls.append((agent_name, user_input))
        if agent_name not in self.scripts:
            raise AssertionError(f"unexpected agent: {agent_name}")
        return self.scripts[agent_name]


class _FakeCHResult:
    def __init__(self, rows):
        self.result_rows = rows


class _FakeCostClient:
    """Returns per-model cost rows for fetch_cost_summary's query."""

    def __init__(self, rows=None):
        # Default: two models, one expensive, one cheap. 1000 total calls.
        # `is None` check so callers can pass `[]` to simulate empty CH.
        self.rows = rows if rows is not None else [
            ("gpt-4o", 0.08, 0.024, 4800.0, 200),
            ("gpt-4o-mini", 0.05, 0.003, 2100.0, 800),
        ]
        self.queries = []

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
        return _FakeCHResult(self.rows)


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# fetch_cost_summary unit tests
# ---------------------------------------------------------------------------


async def test_fetch_cost_summary_happy_path(ledger, monkeypatch):
    client = _FakeCostClient()
    monkeypatch.setattr(_ch_module, "get_client", lambda: client)

    action = get_action("fetch_cost_summary")
    result = await action({"window_hours": 24}, ledger, parent_id="root-1")

    assert result.outcome == "ok"
    assert result.data["window_hours"] == 24
    assert result.data["trace_count"] == 1000
    assert result.data["worst_model_by_cost"] == "gpt-4o"
    assert len(result.data["by_model"]) == 2
    assert result.data["by_model"][0]["model"] == "gpt-4o"  # sorted by cost desc


async def test_fetch_cost_summary_no_data_fails_gracefully(ledger, monkeypatch):
    empty = _FakeCostClient(rows=[])
    monkeypatch.setattr(_ch_module, "get_client", lambda: empty)

    action = get_action("fetch_cost_summary")
    result = await action({}, ledger, parent_id="root-2")

    assert result.outcome == "failed"
    assert "no traces" in result.data["error"].lower()


async def test_fetch_cost_summary_without_ch(ledger, monkeypatch):
    monkeypatch.setattr(_ch_module, "get_client", lambda: None)
    action = get_action("fetch_cost_summary")
    result = await action({}, ledger, parent_id="root-3")
    assert result.outcome == "failed"


# ---------------------------------------------------------------------------
# queue_training unit tests
# ---------------------------------------------------------------------------


async def test_queue_training_approve_writes_queued_row(ledger):
    action = get_action("queue_training")
    result = await action(
        {
            "decision": "approve",
            "recommendation": "train_now",
            "suggested_config": {"focus_models": ["gpt-4o-mini"]},
            "estimated_cost_usd": 0.42,
        },
        ledger,
        parent_id="critic-xyz",
    )
    assert result.outcome == "ok"
    assert result.data["status"] == "queued"
    assert result.data["suggested_config"]["focus_models"] == ["gpt-4o-mini"]
    entry = ledger.get(result.ledger_entry_id)
    assert entry.outcome == "ok"
    assert entry.cost_usd == 0.42
    # MVP explicit: not actually launching a trainer yet.
    assert entry.data["execution"] == "deferred"


async def test_queue_training_reject_writes_skipped_row(ledger):
    action = get_action("queue_training")
    result = await action(
        {"decision": "reject", "suggested_config": {}},
        ledger,
        parent_id="critic-abc",
    )
    assert result.outcome == "skipped"
    entry = ledger.get(result.ledger_entry_id)
    assert entry.outcome == "skipped"
    assert entry.data["status"] == "not_queued"


# ---------------------------------------------------------------------------
# Full recipe e2e — cost regression signal → 4-step chain
# ---------------------------------------------------------------------------


def _seed_cost_regression_signal(ledger: LedgerStore) -> LedgerEntry:
    """Mirror what ObjectiveSensor would emit for cost drift."""
    entry = LedgerEntry(
        type="signal",
        objective_id="cost_per_successful_completion",
        agent="objective_sensor",
        data={
            "delta_pct": 22.0, "baseline": 0.010, "current": 0.012,
            "threshold_pct": 5.0, "direction": "lower_is_better",
        },
        tags=["objective_regression", "cost_per_successful_completion"],
    )
    ledger.append(entry)
    return entry


async def test_full_evaluate_distillation_chain_reconstructs(ledger, monkeypatch):
    signal = _seed_cost_regression_signal(ledger)

    monkeypatch.setattr(_ch_module, "get_client", lambda: _FakeCostClient())

    runner = _StubRunner({
        "training_advisor": _StubAgentResult(data={
            "recommendation": "train_now",
            "confidence": 0.85,
            "reason": "gpt-4o cost is 8x gpt-4o-mini with no quality diff",
            "signals": ["cost_drift: +22%"],
            "suggested_config": {"focus_models": ["gpt-4o-mini"], "production_alpha": 0.4},
        }, cost_usd=0.05),
        "budget_justifier": _StubAgentResult(data={
            "decision": "approve",
            "rationale": "savings of ~$3/day vs $0.4 training run",
            "estimated_cost_usd": 0.4,
            "estimated_benefit": "~70% cost reduction on gpt-4o traffic",
        }, cost_usd=0.01),
    })

    executor = RecipeExecutor(ledger, runner=runner)
    recipe = load_recipe("evaluate_student_distillation")
    assert recipe is not None

    result = await executor.execute(recipe, root_parent_id=signal.id)

    assert not result.halted
    assert set(result.step_outputs.keys()) == {"fetch", "propose", "critique", "queue"}
    assert result.step_outputs["queue"]["status"] == "queued"

    chain = ledger.chain(signal.id)
    types = [e.type for e in chain]
    # signal + action(fetch) + run(propose) + run(critique) + action(queue) = 5
    assert types == ["signal", "action", "run", "run", "action"]

    # Linear chain: every entry is a child of the previous one.
    for i in range(1, len(chain)):
        assert chain[i].parent_id == chain[i - 1].id


async def test_critic_rejects_skips_queue_training(ledger, monkeypatch):
    signal = _seed_cost_regression_signal(ledger)
    monkeypatch.setattr(_ch_module, "get_client", lambda: _FakeCostClient())

    runner = _StubRunner({
        "training_advisor": _StubAgentResult(data={
            "recommendation": "train_now",
            "confidence": 0.7,
            "reason": "borderline signal",
            "suggested_config": {},
        }),
        "budget_justifier": _StubAgentResult(data={
            "decision": "reject",
            "rationale": "cost drift under 10% not worth a $0.4 training run",
        }),
    })

    executor = RecipeExecutor(ledger, runner=runner)
    recipe = load_recipe("evaluate_student_distillation")
    result = await executor.execute(recipe, root_parent_id=signal.id)

    queue_entry = ledger.get(result.step_entry_ids["queue"])
    assert queue_entry.outcome == "skipped"
    assert "condition_unmet" in queue_entry.tags
