"""Step 5.2 end-to-end tests: new-dataset sensor → policy → recipe
→ fetch action → metrics_suggester agent, all chained in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from opentracy.harness.actions import get_action
import opentracy.storage.clickhouse_client as _ch_module
from opentracy.harness.ledger import LedgerStore
from opentracy.harness.objectives.loader import load as load_objective
from opentracy.harness.sensors import NewDatasetSensor, SENSOR_TAG_NEW_DATASET
from opentracy.harness.tasks import RecipeExecutor, load_recipe


# ---------------------------------------------------------------------------
# Shared stubs
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


class _FakeCHRow:
    """Stand-in for ClickHouse result_rows entries."""


class _FakeCHResult:
    def __init__(self, rows):
        self.result_rows = rows


class _FakeCHClient:
    """Minimal CH client for the fetch action's three queries."""

    def __init__(
        self,
        latest_run_id: str = "run-42",
        top_cluster_id: int = 7,
        domain: str = "customer_support",
        samples: list[str] = None,
    ):
        self.latest_run_id = latest_run_id
        self.top_cluster_id = top_cluster_id
        self.domain = domain
        self.samples = samples or ["how do I reset my password?", "where is my order?"]
        self.queries: list[tuple[str, dict]] = []

    def query(self, sql: str, parameters: dict | None = None):
        self.queries.append((sql, parameters or {}))
        if "FROM clustering_runs ORDER BY created_at DESC" in sql:
            return _FakeCHResult([(self.latest_run_id,)])
        if "FROM cluster_datasets" in sql and "ORDER BY trace_count DESC" in sql:
            return _FakeCHResult([(self.top_cluster_id, self.domain)])
        if "FROM trace_cluster_map" in sql:
            return _FakeCHResult([(s,) for s in self.samples])
        raise AssertionError(f"unexpected query: {sql!r}")


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# NewDatasetSensor
# ---------------------------------------------------------------------------


def test_new_dataset_sensor_emits_signal_for_new_run(ledger):
    obj = load_objective("domain_coverage_ratio")
    new_run = {
        "run_id": "run-001",
        "created_at": datetime.now(timezone.utc),
        "num_clusters": 8,
        "silhouette_score": 0.72,
        "total_traces": 420,
    }
    sensor = NewDatasetSensor(obj, ledger, find_fn=lambda since: [new_run])
    entry = sensor.tick()
    assert entry is not None
    assert entry.type == "signal"
    assert SENSOR_TAG_NEW_DATASET in entry.tags
    assert entry.data["run_id"] == "run-001"
    assert entry.data["num_clusters"] == 8


def test_new_dataset_sensor_silent_when_no_new_rows(ledger):
    obj = load_objective("domain_coverage_ratio")
    sensor = NewDatasetSensor(obj, ledger, find_fn=lambda since: [])
    assert sensor.tick() is None


def test_new_dataset_sensor_collapses_multiple_to_newest(ledger):
    """When multiple new runs exist since the last fire, the sensor
    emits one signal for the newest run and advances the anchor past it.
    The collapsed_count field records how many were skipped."""
    obj = load_objective("domain_coverage_ratio")
    ts1 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {"run_id": "a", "created_at": ts1, "num_clusters": 4,
         "silhouette_score": 0.5, "total_traces": 100},
        {"run_id": "b", "created_at": ts2, "num_clusters": 7,
         "silhouette_score": 0.6, "total_traces": 200},
    ]
    sensor = NewDatasetSensor(obj, ledger, find_fn=lambda since: rows)
    entry = sensor.tick()
    assert entry.data["run_id"] == "b"
    assert entry.data["collapsed_count"] == 2

    # Subsequent tick with no new rows must not fire again.
    sensor._find_fn = lambda since: []
    assert sensor.tick() is None


# ---------------------------------------------------------------------------
# fetch_dataset_samples action
# ---------------------------------------------------------------------------


async def test_fetch_dataset_samples_happy_path(ledger, monkeypatch):
    client = _FakeCHClient()
    monkeypatch.setattr(_ch_module, "get_client", lambda: client)

    action = get_action("fetch_dataset_samples")
    result = await action({}, ledger, parent_id="root-1")

    assert result.outcome == "ok"
    assert result.data["run_id"] == "run-42"
    assert result.data["cluster_id"] == 7
    assert result.data["domain"] == "customer_support"
    assert result.data["sample_count"] == 2

    entry = ledger.get(result.ledger_entry_id)
    assert entry.outcome == "ok"
    assert entry.parent_id == "root-1"
    # Three CH queries: latest_run_id, top_cluster, samples.
    assert len(client.queries) == 3


async def test_fetch_dataset_samples_fails_without_ch(ledger, monkeypatch):
    monkeypatch.setattr(_ch_module, "get_client", lambda: None)
    action = get_action("fetch_dataset_samples")
    result = await action({}, ledger, parent_id="root-2")
    assert result.outcome == "failed"
    assert "clickhouse unavailable" in result.data["error"]


async def test_fetch_dataset_samples_fails_when_no_runs(ledger, monkeypatch):
    class _EmptyClient:
        def query(self, sql, parameters=None):
            return _FakeCHResult([])

    monkeypatch.setattr(_ch_module, "get_client", lambda: _EmptyClient())
    action = get_action("fetch_dataset_samples")
    result = await action({}, ledger, parent_id="root-3")
    assert result.outcome == "failed"
    assert "no clustering_runs" in result.data["error"]


# ---------------------------------------------------------------------------
# Recipe end-to-end — signal → recipe → fetch + agent, full chain
# ---------------------------------------------------------------------------


async def test_full_suggest_metrics_chain_reconstructs(ledger, monkeypatch):
    # 1. Seed a new_dataset signal.
    obj = load_objective("domain_coverage_ratio")
    sensor = NewDatasetSensor(
        obj, ledger,
        find_fn=lambda since: [{
            "run_id": "run-plot",
            "created_at": datetime.now(timezone.utc),
            "num_clusters": 3,
            "silhouette_score": 0.55,
            "total_traces": 300,
        }],
    )
    signal = sensor.tick()
    assert signal is not None

    # 2. Stub CH so the fetch action succeeds.
    monkeypatch.setattr(
        _ch_module, "get_client", lambda: _FakeCHClient(latest_run_id="run-plot")
    )

    # 3. Stub the agent runner so metrics_suggester returns suggestions.
    runner = _StubRunner({
        "metrics_suggester": _StubAgentResult(data={
            "suggested_metrics": [
                {"name": "prompt_clarity", "description": "...", "type": "quality"},
                {"name": "answer_completeness", "description": "...", "type": "quality"},
            ],
        }),
    })
    executor = RecipeExecutor(ledger, runner=runner)

    # 4. Load + execute the shipped recipe.
    recipe = load_recipe("suggest_metrics")
    assert recipe is not None
    result = await executor.execute(recipe, root_parent_id=signal.id)

    assert not result.halted
    assert set(result.step_outputs.keys()) == {"fetch", "suggest"}
    assert result.step_outputs["fetch"]["run_id"] == "run-plot"
    assert len(result.step_outputs["suggest"]["suggested_metrics"]) == 2

    # 5. Chain must be contiguous: signal → fetch (action) → suggest (run).
    chain = ledger.chain(signal.id)
    types = [e.type for e in chain]
    assert types == ["signal", "action", "run"]
    # Each entry's parent is the previous entry in the chain.
    assert chain[1].parent_id == signal.id
    assert chain[2].parent_id == chain[1].id
