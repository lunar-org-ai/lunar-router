"""Tests for the actions module.

Actions are the rightmost layer of a recipe — they're the only side
effects the harness performs. These tests pin the `run_eval` action's
contract (validates eval_case shape, writes ledger row, outcomes ok /
failed depending on input).
"""

from __future__ import annotations

import pytest

from opentracy.harness.actions import get_action, list_actions
from opentracy.harness.ledger import LedgerStore


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


def test_run_eval_is_registered():
    assert "run_eval" in list_actions()
    assert callable(get_action("run_eval"))


async def test_run_eval_happy_path_writes_ok_action(ledger):
    action = get_action("run_eval")
    assert action is not None

    result = await action(
        {
            "eval_case": {
                "input": "What is 2+2?",
                "expected_behavior": "answer 4",
                "check_type": "quality_threshold",
                "severity": "medium",
                "tags": ["math"],
            },
            "rationale": "detected hallucination on arithmetic",
        },
        ledger,
        parent_id="root-abc",
    )

    assert result.outcome == "ok"
    assert result.ledger_entry_id is not None
    entry = ledger.get(result.ledger_entry_id)
    assert entry is not None
    assert entry.type == "action"
    assert entry.agent == "run_eval"
    assert entry.parent_id == "root-abc"
    assert entry.outcome == "ok"
    assert entry.data["check_type"] == "quality_threshold"
    # MVP: the action queues rather than executes against a live model.
    assert entry.data["status"] == "queued"


async def test_run_eval_missing_fields_fails_gracefully(ledger):
    action = get_action("run_eval")
    result = await action(
        {"eval_case": {"input": "hi"}},  # missing expected_behavior, check_type
        ledger,
        parent_id="root-xyz",
    )

    assert result.outcome == "failed"
    assert set(result.data["missing_fields"]) == {"expected_behavior", "check_type"}
    entry = ledger.get(result.ledger_entry_id)
    assert entry.outcome == "failed"
    assert "validation_failed" in entry.tags


async def test_run_eval_chains_parent_id(ledger):
    """The action's ledger row must be a child of whatever parent_id
    the executor passes — recipe chains depend on this."""
    action = get_action("run_eval")
    result = await action(
        {
            "eval_case": {
                "input": "x",
                "expected_behavior": "y",
                "check_type": "valid_json",
            }
        },
        ledger,
        parent_id="recipe-step-42",
    )
    entry = ledger.get(result.ledger_entry_id)
    assert entry.parent_id == "recipe-step-42"
