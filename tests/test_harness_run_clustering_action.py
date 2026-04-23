"""Tests for the `run_clustering` action + end-to-end Step 5.1 flow:
traces-threshold sensor → policy → recipe → action invokes pipeline.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from opentracy.harness.actions import get_action
from opentracy.harness.ledger import LedgerStore
from opentracy.harness.sensors import NewTracesThresholdSensor
from opentracy.harness.tasks import RecipeExecutor, load_recipe


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Action unit tests — stub ClusteringPipeline via import override
# ---------------------------------------------------------------------------


@dataclass
class _FakeClusteringResult:
    run_id: str = "run-abc"
    num_clusters: int = 7
    silhouette: float = 0.62
    trace_count: int = 350


class _FakePipeline:
    def __init__(self, strategy="auto"):
        self.strategy = strategy

    async def run(self, days=7, min_traces=50):
        return _FakeClusteringResult()


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Replace the real ClusteringPipeline import with a stub so tests
    don't need ClickHouse or embeddings on the box."""
    module = types.ModuleType("opentracy.clustering.pipeline")
    module.ClusteringPipeline = _FakePipeline
    # Preserve real module in sys.modules if present so subsequent
    # real imports aren't corrupted beyond this test.
    monkeypatch.setitem(sys.modules, "opentracy.clustering.pipeline", module)
    yield


async def test_run_clustering_action_invokes_pipeline_and_writes_ledger(
    ledger, stubbed_pipeline,
):
    action = get_action("run_clustering")
    assert action is not None

    result = await action(
        {"days": 3, "min_traces": 25, "strategy": "auto"},
        ledger,
        parent_id="root-xyz",
    )

    assert result.outcome == "ok"
    assert result.data["clusters_created"] == 7
    assert result.data["silhouette"] == pytest.approx(0.62)
    assert result.data["run_id"] == "run-abc"

    entry = ledger.get(result.ledger_entry_id)
    assert entry is not None
    assert entry.type == "action"
    assert entry.agent == "run_clustering"
    assert entry.parent_id == "root-xyz"
    assert entry.outcome == "ok"
    assert "clustering" in entry.tags


async def test_run_clustering_action_fails_gracefully_on_pipeline_error(
    ledger, monkeypatch,
):
    class _ExplodingPipeline:
        def __init__(self, **_): ...
        async def run(self, **_):
            raise RuntimeError("clickhouse unavailable")

    module = types.ModuleType("opentracy.clustering.pipeline")
    module.ClusteringPipeline = _ExplodingPipeline
    monkeypatch.setitem(sys.modules, "opentracy.clustering.pipeline", module)

    action = get_action("run_clustering")
    result = await action({}, ledger, parent_id="root-abc")

    assert result.outcome == "failed"
    assert "clickhouse unavailable" in result.data["error"]
    entry = ledger.get(result.ledger_entry_id)
    assert entry.outcome == "failed"
    assert "pipeline_failed" in entry.tags


# ---------------------------------------------------------------------------
# End-to-end: sensor → recipe → action
# ---------------------------------------------------------------------------


async def test_end_to_end_traces_threshold_drives_cluster_and_label_recipe(
    ledger, stubbed_pipeline,
):
    """Constructs the sensor directly, asserts its signal + the shipped
    recipe execute as one causal chain in the ledger."""
    from opentracy.harness.objectives.loader import load as load_objective

    objective = load_objective("domain_coverage_ratio")
    assert objective is not None

    sensor = NewTracesThresholdSensor(
        objective, ledger, threshold=100,
        count_fn=lambda since: 5000,
    )
    signal = sensor.tick()
    assert signal is not None

    recipe = load_recipe("cluster_and_label")
    assert recipe is not None

    executor = RecipeExecutor(ledger, runner=None)  # action-only recipe
    result = await executor.execute(recipe, root_parent_id=signal.id)

    assert not result.halted
    assert "cluster" in result.step_outputs
    assert result.step_outputs["cluster"]["clusters_created"] == 7

    chain = ledger.chain(signal.id)
    types = [e.type for e in chain]
    # signal + action (1-step recipe)
    assert types == ["signal", "action"]
    assert chain[1].parent_id == signal.id
    assert chain[1].data["run_id"] == "run-abc"
