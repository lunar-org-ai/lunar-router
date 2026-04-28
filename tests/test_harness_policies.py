"""Policy YAML loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from opentracy.harness.triggers.policies import (
    Policy,
    PolicyBudget,
    PolicyMatch,
    load_policies,
    load_policy,
)


def test_load_all_returns_shipped_policies():
    policies = load_policies()
    ids = {p.id for p in policies}
    # Policies that ship with the repo today. New ones are added as
    # YAML drops; this test ensures all of them parse.
    assert ids == {
        "cost_drift_to_evaluate_student_distillation",
        "latency_drift_to_trace_scanner",
        "cadence_to_trace_scan",
        "new_traces_to_cluster_and_label",
        "new_dataset_to_suggest_metrics",
    }


def test_policy_shape_is_parsed():
    p = load_policy("cost_drift_to_evaluate_student_distillation")
    assert p is not None
    assert p.match.objective_id == "cost_per_successful_completion"
    assert "objective_regression" in p.match.signal_tags
    assert p.dispatch.recipe == "evaluate_student_distillation"
    assert p.dispatch.agent is None
    assert p.budget.max_per_day == 3


def test_load_unknown_policy_returns_none():
    assert load_policy("does_not_exist") is None


def test_empty_directory_returns_empty(tmp_path):
    assert load_policies(tmp_path) == []


def test_policy_loader_round_trip_with_custom_yaml(tmp_path):
    (tmp_path / "custom.yaml").write_text(
        yaml.safe_dump({
            "id": "custom",
            "description": "inline",
            "match": {"signal_tags": ["objective_regression"]},
            "dispatch": {"agent": "trace_scanner"},
            "budget": {"max_per_day": 2},
        })
    )
    policies = load_policies(tmp_path)
    assert len(policies) == 1
    p = policies[0]
    assert p.id == "custom"
    assert p.match.objective_id is None  # wildcard
    assert p.budget.max_per_day == 2


def test_defaults_populate_when_fields_omitted():
    p = Policy(
        id="minimal",
        description="",
        match=PolicyMatch(),
        dispatch={"agent": "x"},  # type: ignore[arg-type]
    )
    assert p.budget == PolicyBudget()
    assert p.budget.max_per_day == 10


def test_broken_yaml_raises(tmp_path):
    """A policy file with missing required fields must raise at load
    time — silently dropping a malformed policy would hide config bugs
    until the broken behavior mattered, which is exactly the scenario
    the ledger is supposed to prevent."""
    (tmp_path / "broken.yaml").write_text("id: oops\n")  # no match, no dispatch
    with pytest.raises(Exception):  # pydantic.ValidationError
        load_policies(tmp_path)
