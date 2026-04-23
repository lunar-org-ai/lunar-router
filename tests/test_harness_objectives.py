"""Tests for objective YAML loading + compute-function graceful degradation."""

from __future__ import annotations

from unittest.mock import patch

from opentracy.harness.objectives import compute, loader


EXPECTED_IDS = {
    "cost_per_successful_completion",
    "p95_latency_ms",
    "domain_coverage_ratio",
}


def test_load_all_returns_three_objectives():
    objectives = loader.load_all()
    assert {o.id for o in objectives} == EXPECTED_IDS


def test_load_returns_objective_by_id():
    obj = loader.load("p95_latency_ms")
    assert obj is not None
    assert obj.id == "p95_latency_ms"
    assert obj.direction == "lower_is_better"
    assert obj.unit == "ms"


def test_load_unknown_objective_returns_none():
    assert loader.load("does_not_exist") is None


def test_each_objective_resolves_to_callable():
    for obj in loader.load_all():
        fn = loader.resolve_compute_fn(obj)
        assert callable(fn)


def test_compute_fns_return_empty_when_clickhouse_disabled():
    """Graceful degradation: CH disabled ⇒ empty list, not an exception."""
    with patch("opentracy.harness.objectives.compute.get_client", return_value=None):
        assert compute.cost_per_successful_completion() == []
        assert compute.p95_latency_ms() == []
        assert compute.domain_coverage_ratio() == []


def test_objective_has_guardrails_and_dimensions_parsed():
    obj = loader.load("cost_per_successful_completion")
    assert obj is not None
    assert obj.dimensions == ["selected_model"]
    assert len(obj.guardrails) == 2
    guardrail_types = {g.type for g in obj.guardrails}
    assert "no_regression_worse_than_pct" in guardrail_types
    assert "min_sample_size" in guardrail_types
