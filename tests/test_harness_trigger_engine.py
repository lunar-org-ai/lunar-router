"""End-to-end tests for TriggerEngine.tick().

The engine's public contract: turn a compute_fn measurement into a
signal → policy match → dispatch ledger row, with budget enforcement
and optional handler invocation. All tests run in-memory — compute_fns
are stubbed via the Objective's `compute_fn` module path pointing at a
scratch module dynamically.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pytest

from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.objectives.schemas import (
    GuardrailSpec,
    Objective,
    ObjectiveMeasurement,
)
from opentracy.harness.triggers import Policy, TriggerEngine
from opentracy.harness.triggers.policies import PolicyDispatch, PolicyMatch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_measurement(value: float, sample_size: int = 100) -> ObjectiveMeasurement:
    return ObjectiveMeasurement(
        objective_id="cost_per_successful_completion",
        value=value,
        unit="USD",
        sample_size=sample_size,
        window_start=_now(),
        window_end=_now(),
        computed_at=_now(),
    )


def _install_fake_compute(module_name: str, func_name: str, queue: list):
    """Install a module at `module_name` exposing `func_name`, which
    pops one value off `queue` each call. Tests push measurements in
    the order they want them consumed."""
    mod = types.ModuleType(module_name)

    def _fn():
        return queue.pop(0) if queue else []

    setattr(mod, func_name, _fn)
    sys.modules[module_name] = mod


def _make_objective(compute_fn: str, regression_pct: float = 5.0) -> Objective:
    return Objective(
        id="cost_per_successful_completion",
        description="test",
        compute_fn=compute_fn,
        unit="USD",
        direction="lower_is_better",
        guardrails=[
            GuardrailSpec(type="no_regression_worse_than_pct", threshold=regression_pct),
        ],
    )


def _make_policy(
    policy_id: str = "cost_to_advisor",
    max_per_day: int = 3,
    agent: str = "training_advisor",
) -> Policy:
    return Policy(
        id=policy_id,
        description="test policy",
        match=PolicyMatch(
            signal_tags=["objective_regression"],
            objective_id="cost_per_successful_completion",
        ),
        dispatch=PolicyDispatch(agent=agent, parameters={"hint": "x"}),
        budget={"max_per_day": max_per_day},  # type: ignore[arg-type]
    )


@pytest.fixture
def ledger(tmp_path) -> LedgerStore:
    store = LedgerStore(db_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Signal → dispatch round trip
# ---------------------------------------------------------------------------


def test_first_tick_is_silent_then_regression_emits_and_dispatches(ledger):
    queue: list[list[ObjectiveMeasurement]] = [
        [_make_measurement(0.010)],  # baseline
        [_make_measurement(0.020)],  # +100% — regression
    ]
    _install_fake_compute("tests._trigger_fake_cost_a", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_a:compute")
    engine = TriggerEngine([obj], [_make_policy()], ledger)

    first = engine.tick()
    assert first.signals == []
    assert first.dispatches == []

    second = engine.tick()
    assert len(second.signals) == 1
    assert len(second.dispatches) == 1

    signal = second.signals[0]
    dispatch = second.dispatches[0]
    assert signal.type == "signal"
    assert dispatch.type == "run"
    assert dispatch.parent_id == signal.id
    assert dispatch.parameters_in["policy_id"] == "cost_to_advisor"
    assert dispatch.parameters_in["hint"] == "x"
    assert "policy_dispatch" in dispatch.tags


def test_dispatch_handler_receives_signal_and_dispatch_id(ledger):
    queue = [
        [_make_measurement(0.010)],
        [_make_measurement(0.020)],
    ]
    _install_fake_compute("tests._trigger_fake_cost_b", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_b:compute")

    captured: list[tuple] = []

    def handler(signal, dispatch_id, policy):
        captured.append((signal.id, dispatch_id, policy.id))

    engine = TriggerEngine(
        [obj], [_make_policy()], ledger,
        dispatch_handlers={"training_advisor": handler},
    )
    engine.tick()  # baseline
    engine.tick()

    assert len(captured) == 1
    signal_id, dispatch_id, policy_id = captured[0]
    assert policy_id == "cost_to_advisor"
    # Handler gets the actual ids of the ledger entries it chains onto.
    assert ledger.get(dispatch_id) is not None
    assert ledger.get(dispatch_id).parent_id == signal_id


def test_missing_handler_still_writes_dispatch_row(ledger):
    """A policy without a registered handler must still produce a
    ledger row so staged policies are auditable."""
    queue = [
        [_make_measurement(0.010)],
        [_make_measurement(0.020)],
    ]
    _install_fake_compute("tests._trigger_fake_cost_c", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_c:compute")
    engine = TriggerEngine([obj], [_make_policy()], ledger)
    engine.tick()
    r = engine.tick()
    assert len(r.dispatches) == 1


def test_handler_exception_does_not_break_loop(ledger):
    """Two policies match the same signal; one handler crashes. The
    other handler must still fire and both dispatch rows must land."""
    queue = [
        [_make_measurement(0.010)],
        [_make_measurement(0.020)],
    ]
    _install_fake_compute("tests._trigger_fake_cost_d", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_d:compute")

    good_calls: list[str] = []

    def crashing(signal, dispatch_id, policy):
        raise RuntimeError("boom")

    def good(signal, dispatch_id, policy):
        good_calls.append(dispatch_id)

    engine = TriggerEngine(
        [obj],
        [
            _make_policy("p_crash", agent="crashing"),
            _make_policy("p_good", agent="good"),
        ],
        ledger,
        dispatch_handlers={"crashing": crashing, "good": good},
    )
    engine.tick()
    r = engine.tick()
    assert len(r.dispatches) == 2
    assert len(good_calls) == 1


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_budget_caps_repeated_dispatches(ledger):
    # 4 ticks, each oscillating value to re-trigger regression.
    values = [0.010, 0.020, 0.010, 0.020, 0.010, 0.020, 0.010, 0.020]
    queue = [[_make_measurement(v)] for v in values]
    _install_fake_compute("tests._trigger_fake_cost_e", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_e:compute")

    engine = TriggerEngine([obj], [_make_policy(max_per_day=2)], ledger)

    # Drive ticks until we've exhausted the measurement queue.
    results = [engine.tick() for _ in range(8)]
    total_dispatches = sum(len(r.dispatches) for r in results)
    total_skipped = sum(len(r.skipped_budget) for r in results)

    assert total_dispatches == 2  # cap honored
    assert total_skipped >= 1     # at least one skip recorded


# ---------------------------------------------------------------------------
# Compute_fn failures must not crash the loop
# ---------------------------------------------------------------------------


def test_compute_fn_exception_is_swallowed(ledger):
    mod = types.ModuleType("tests._trigger_fake_cost_f")

    def _raise():
        raise RuntimeError("ch down")

    mod.compute = _raise
    sys.modules["tests._trigger_fake_cost_f"] = mod

    obj = _make_objective("tests._trigger_fake_cost_f:compute")
    engine = TriggerEngine([obj], [_make_policy()], ledger)
    result = engine.tick()
    assert result.signals == []
    assert result.dispatches == []


# ---------------------------------------------------------------------------
# Policy filter semantics
# ---------------------------------------------------------------------------


def test_objective_id_mismatch_skips_policy(ledger):
    queue = [
        [_make_measurement(0.010)],
        [_make_measurement(0.020)],
    ]
    _install_fake_compute("tests._trigger_fake_cost_g", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_g:compute")

    # Policy targets a different objective → should not dispatch.
    mismatched = Policy(
        id="other",
        description="",
        match=PolicyMatch(
            signal_tags=["objective_regression"],
            objective_id="p95_latency_ms",
        ),
        dispatch=PolicyDispatch(agent="trace_scanner"),
    )
    engine = TriggerEngine([obj], [mismatched], ledger)
    engine.tick()
    r = engine.tick()
    assert len(r.signals) == 1
    assert r.dispatches == []


def test_measurement_observation_written_per_tick(ledger):
    """The engine writes one `measurement` observation per objective
    per tick, even when no regression fires. This is what gives the
    dashboard plot a continuous time-series to draw."""
    queue = [
        [_make_measurement(0.010, sample_size=100)],
        [_make_measurement(0.011, sample_size=100)],
    ]
    _install_fake_compute("tests._trigger_fake_cost_m", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_m:compute", regression_pct=50.0)
    engine = TriggerEngine([obj], [_make_policy()], ledger)

    engine.tick()  # baseline, no regression
    engine.tick()  # within threshold, no regression

    measurements = [
        e for e in ledger.recent(limit=50)
        if e.type == "observation" and "measurement" in e.tags
    ]
    assert len(measurements) == 2
    values = [m.data["value"] for m in measurements]
    # newest-first, so 0.011 before 0.010
    assert values == [pytest.approx(0.011), pytest.approx(0.010)]
    assert all(m.data["sample_size"] == 100 for m in measurements)
    assert all(m.objective_id == obj.id for m in measurements)


def test_no_measurement_written_when_compute_returns_empty(ledger):
    mod = types.ModuleType("tests._trigger_fake_empty")
    mod.compute = lambda: []
    sys.modules["tests._trigger_fake_empty"] = mod

    obj = _make_objective("tests._trigger_fake_empty:compute")
    engine = TriggerEngine([obj], [_make_policy()], ledger)
    engine.tick()
    measurements = [
        e for e in ledger.recent(limit=50)
        if e.type == "observation" and "measurement" in e.tags
    ]
    assert measurements == []


def test_full_chain_is_reconstructible_from_signal(ledger):
    queue = [
        [_make_measurement(0.010)],
        [_make_measurement(0.020)],
    ]
    _install_fake_compute("tests._trigger_fake_cost_h", "compute", queue)
    obj = _make_objective("tests._trigger_fake_cost_h:compute")
    engine = TriggerEngine([obj], [_make_policy()], ledger)
    engine.tick()
    r = engine.tick()
    signal = r.signals[0]

    chain = ledger.chain(signal.id)
    types_in_chain = [e.type for e in chain]
    assert types_in_chain == ["signal", "run"]
    assert chain[1].parent_id == signal.id
