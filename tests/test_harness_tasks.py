"""Tests for recipe schema, loader, and executor.

The executor is the heart of the YAML-declarative layer: these tests
assert recipes load without a Python change, execute steps in order,
chain every ledger row correctly, enforce budget, honor conditions,
and isolate failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.tasks import (
    Recipe,
    RecipeExecutor,
    load_recipe,
    load_recipes,
)
from opentracy.harness.tasks.schema import (
    RecipeBudget,
    RecipeCondition,
    RecipeStep,
)


# ---------------------------------------------------------------------------
# Stub AgentRunner
# ---------------------------------------------------------------------------


@dataclass
class _StubAgentResult:
    """Shape-compatible with AgentRunner's real result."""

    data: dict
    cost_usd: float = 0.0
    duration_ms: int = 1


class StubRunner:
    """Returns a scripted `_StubAgentResult` per agent name.

    Tests load a dict of {agent_name: result} and each `run(name, _)`
    call pops the expected result. Unknown agents or missing scripts
    raise loudly so tests don't silently pass on the wrong path.
    """

    def __init__(self, scripts: dict[str, Any]):
        self.scripts = scripts
        self.calls: list[tuple[str, str]] = []

    async def run(self, agent_name: str, user_input: str):
        self.calls.append((agent_name, user_input))
        script = self.scripts.get(agent_name)
        if script is None:
            raise AssertionError(f"unexpected agent call: {agent_name}")
        if isinstance(script, Exception):
            raise script
        return script


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


def _signal(ledger: LedgerStore) -> LedgerEntry:
    """Seed a signal entry the executor can chain off."""
    entry = LedgerEntry(
        type="signal",
        objective_id="cost_per_successful_completion",
        agent="objective_sensor",
        tags=["cadence"],
    )
    ledger.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Schema + loader
# ---------------------------------------------------------------------------


def test_shipping_recipe_loads():
    recipe = load_recipe("trace_scan_and_evaluate")
    assert recipe is not None
    step_ids = [s.id for s in recipe.steps]
    assert step_ids == ["inspect", "propose", "critique", "queue_eval"]
    # Final step is conditional on the critic's decision.
    assert recipe.steps[-1].condition is not None
    assert recipe.steps[-1].condition.from_step == "critique"


def test_load_all_recipes_returns_non_empty():
    recipes = load_recipes()
    assert len(recipes) >= 1


def test_recipe_validator_rejects_duplicate_step_ids():
    with pytest.raises(Exception):
        Recipe(
            id="bad",
            steps=[
                RecipeStep(id="a", type="agent", agent="x"),
                RecipeStep(id="a", type="agent", agent="y"),
            ],
        )


def test_recipe_validator_rejects_forward_reference():
    with pytest.raises(Exception):
        Recipe(
            id="bad",
            steps=[
                RecipeStep(id="a", type="agent", agent="x", input_from="b"),
                RecipeStep(id="b", type="agent", agent="y"),
            ],
        )


def test_recipe_step_requires_agent_when_type_agent():
    with pytest.raises(Exception):
        RecipeStep(id="x", type="agent")  # no agent field


def test_recipe_step_rejects_both_input_and_input_from():
    with pytest.raises(Exception):
        RecipeStep(
            id="x", type="agent", agent="a", input="hi", input_from="other",
        )


# ---------------------------------------------------------------------------
# Execution — happy path
# ---------------------------------------------------------------------------


@pytest.fixture
def three_step_recipe() -> Recipe:
    return Recipe(
        id="mini",
        steps=[
            RecipeStep(id="inspect", type="agent", agent="trace_scanner", input="scan"),
            RecipeStep(id="propose", type="agent", agent="eval_generator", input_from="inspect"),
            RecipeStep(id="critique", type="agent", agent="budget_justifier", input_from="propose"),
            RecipeStep(
                id="queue_eval",
                type="action",
                action="run_eval",
                input_from="propose",
                condition=RecipeCondition(from_step="critique", field="decision", equals="approve"),
            ),
        ],
    )


async def test_full_recipe_executes_all_steps_and_chains(ledger, three_step_recipe):
    signal = _signal(ledger)
    runner = StubRunner({
        "trace_scanner": _StubAgentResult(data={"issues": [{"type": "cost_anomaly"}]}, cost_usd=0.05),
        "eval_generator": _StubAgentResult(data={
            "eval_case": {"input": "x", "expected_behavior": "y", "check_type": "valid_json"},
            "rationale": "catches cost spike pattern",
        }, cost_usd=0.03),
        "budget_justifier": _StubAgentResult(data={"decision": "approve", "rationale": "ok"}, cost_usd=0.01),
    })
    executor = RecipeExecutor(ledger, runner=runner)

    result = await executor.execute(three_step_recipe, root_parent_id=signal.id)

    assert not result.halted
    assert set(result.step_outputs.keys()) == {"inspect", "propose", "critique", "queue_eval"}
    assert result.total_cost_usd == pytest.approx(0.09)

    chain = ledger.chain(signal.id)
    # signal + 3 agent runs + 1 action = 5 entries
    assert len(chain) == 5
    types = [e.type for e in chain]
    assert types == ["signal", "run", "run", "run", "action"]


async def test_chain_is_linear_parent_id_follows_step_order(ledger, three_step_recipe):
    signal = _signal(ledger)
    runner = StubRunner({
        "trace_scanner": _StubAgentResult(data={"issues": []}),
        "eval_generator": _StubAgentResult(data={
            "eval_case": {"input": "x", "expected_behavior": "y", "check_type": "valid_json"},
        }),
        "budget_justifier": _StubAgentResult(data={"decision": "approve"}),
    })
    executor = RecipeExecutor(ledger, runner=runner)
    result = await executor.execute(three_step_recipe, root_parent_id=signal.id)

    # Each step should be a child of the previous step.
    inspect = ledger.get(result.step_entry_ids["inspect"])
    propose = ledger.get(result.step_entry_ids["propose"])
    critique = ledger.get(result.step_entry_ids["critique"])
    queue_eval = ledger.get(result.step_entry_ids["queue_eval"])

    assert inspect.parent_id == signal.id
    assert propose.parent_id == inspect.id
    assert critique.parent_id == propose.id
    assert queue_eval.parent_id == critique.id


# ---------------------------------------------------------------------------
# Condition-based skipping
# ---------------------------------------------------------------------------


async def test_queue_eval_skipped_when_critic_rejects(ledger, three_step_recipe):
    signal = _signal(ledger)
    runner = StubRunner({
        "trace_scanner": _StubAgentResult(data={"issues": [{"severity": "low"}]}),
        "eval_generator": _StubAgentResult(data={
            "eval_case": {"input": "x", "expected_behavior": "y", "check_type": "valid_json"},
        }),
        "budget_justifier": _StubAgentResult(data={"decision": "reject", "rationale": "cost too high"}),
    })
    executor = RecipeExecutor(ledger, runner=runner)
    result = await executor.execute(three_step_recipe, root_parent_id=signal.id)

    queue_entry = ledger.get(result.step_entry_ids["queue_eval"])
    assert queue_entry.outcome == "skipped"
    assert "condition_unmet" in queue_entry.tags
    # Chain still intact: critic → skipped queue_eval.
    assert queue_entry.parent_id == result.step_entry_ids["critique"]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


async def test_budget_cap_halts_recipe(ledger):
    recipe = Recipe(
        id="expensive",
        budget=RecipeBudget(max_cost_usd=0.10),
        steps=[
            RecipeStep(id="a", type="agent", agent="trace_scanner", input="a"),
            RecipeStep(id="b", type="agent", agent="eval_generator", input_from="a"),
            RecipeStep(id="c", type="agent", agent="budget_justifier", input_from="b"),
        ],
    )
    # Each step costs 0.06, so a+b > 0.10 → c must be skipped.
    runner = StubRunner({
        "trace_scanner": _StubAgentResult(data={"x": 1}, cost_usd=0.06),
        "eval_generator": _StubAgentResult(data={"x": 2}, cost_usd=0.06),
        "budget_justifier": _StubAgentResult(data={"decision": "approve"}, cost_usd=0.06),
    })
    executor = RecipeExecutor(ledger, runner=runner)
    signal = _signal(ledger)
    result = await executor.execute(recipe, root_parent_id=signal.id)

    assert result.halted
    assert result.halt_reason == "budget_exceeded"
    c_entry = ledger.get(result.step_entry_ids["c"])
    assert c_entry.outcome == "skipped"
    assert "budget_exceeded" in c_entry.tags


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


async def test_agent_exception_halts_and_cascades(ledger, three_step_recipe):
    signal = _signal(ledger)
    runner = StubRunner({
        "trace_scanner": _StubAgentResult(data={"issues": []}),
        "eval_generator": RuntimeError("model timeout"),
        # critic/action should not run — cascade halt.
    })
    executor = RecipeExecutor(ledger, runner=runner)
    result = await executor.execute(three_step_recipe, root_parent_id=signal.id)

    assert result.halted
    assert "eval_generator" in result.halt_reason.lower() or "propose" in result.halt_reason
    propose_entry = ledger.get(result.step_entry_ids["propose"])
    assert propose_entry.outcome == "failed"
    # Downstream steps are recorded but skipped.
    critique_entry = ledger.get(result.step_entry_ids["critique"])
    assert critique_entry.outcome == "skipped"
    assert "cascade_halt" in critique_entry.tags


# ---------------------------------------------------------------------------
# Action-only smoke test — no AgentRunner needed if recipe has no agents
# ---------------------------------------------------------------------------


async def test_recipe_with_only_actions_works_without_runner(ledger):
    recipe = Recipe(
        id="action_only",
        steps=[
            RecipeStep(
                id="queue_it",
                type="action",
                action="run_eval",
                input="dummy",  # run_eval handles non-dict input via text coercion
            ),
        ],
    )
    signal = _signal(ledger)
    executor = RecipeExecutor(ledger, runner=None)
    result = await executor.execute(recipe, root_parent_id=signal.id)

    # run_eval requires eval_case fields → this coerces to {"text": "dummy"}
    # which doesn't have the required fields → outcome=failed for that step.
    assert result.step_entry_ids.get("queue_it") is not None
    entry = ledger.get(result.step_entry_ids["queue_it"])
    assert entry.type == "action"
    assert entry.outcome == "failed"  # validation failure in run_eval
