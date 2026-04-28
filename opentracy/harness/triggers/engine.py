"""TriggerEngine — turns ledger signals into agent dispatches.

One `tick()` is one pass through the loop:
    sensors → signals → policies → budget check → dispatch → ledger row

The engine is the only component that knows about all four primitives
(objective, sensor, policy, agent). Everything else stays local to its
role: sensors don't know about policies, policies don't know about
agents beyond a name, dispatch handlers don't know about the engine.

Dispatch handlers are opt-in callables keyed by agent name. When an
agent has no registered handler, the dispatch ledger row is still
written (the causal chain stays intact) but no code runs. This lets
you stage policies before wiring them to real agents.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.objectives.loader import resolve_compute_fn
from opentracy.harness.objectives.schemas import Objective
from opentracy.harness.sensors import (
    CadenceSensor,
    NewDatasetSensor,
    NewTracesThresholdSensor,
    ObjectiveSensor,
)

from .policies import Policy

logger = logging.getLogger(__name__)


# Signature for a dispatch handler. Receives the matching signal entry
# and the dispatch ledger row's id so the handler can chain its own
# runs as children of the dispatch.
DispatchHandler = Callable[[LedgerEntry, str, Policy], None]


@dataclass
class TickResult:
    """What one engine tick produced. Useful for dashboards and tests.

    `pending` holds coroutines returned by async dispatch handlers
    (chiefly the recipe handler). The runner loop awaits these before
    sleeping to the next tick; callers invoking the engine directly can
    do `await asyncio.gather(*result.pending)` to drive them to
    completion deterministically.
    """

    signals: list[LedgerEntry] = field(default_factory=list)
    dispatches: list[LedgerEntry] = field(default_factory=list)
    skipped_budget: list[str] = field(default_factory=list)  # policy ids
    pending: list[Awaitable] = field(default_factory=list)


class TriggerEngine:
    """Runs one cycle at a time. An asyncio loop that calls `.tick()`
    every N seconds is a trivial wrapper — deliberately out of scope
    here so the engine stays synchronous, deterministic, and easy to
    drive from tests."""

    def __init__(
        self,
        objectives: list[Objective],
        policies: list[Policy],
        ledger: LedgerStore,
        dispatch_handlers: Optional[dict[str, DispatchHandler]] = None,
        cadence_sensors: Optional[list[CadenceSensor]] = None,
        traces_threshold_sensors: Optional[list[NewTracesThresholdSensor]] = None,
        new_dataset_sensors: Optional[list[NewDatasetSensor]] = None,
    ):
        self.objectives = objectives
        self.policies = policies
        self.ledger = ledger
        self.dispatch_handlers: dict[str, DispatchHandler] = dispatch_handlers or {}
        self.sensors: dict[str, ObjectiveSensor] = {
            obj.id: ObjectiveSensor(obj, ledger) for obj in objectives
        }
        # Cadence and traces-threshold sensors are opt-in lists — their
        # construction depends on deployment config (intervals, trace
        # volume thresholds) that doesn't belong in the objective YAML.
        self.cadence_sensors: list[CadenceSensor] = cadence_sensors or []
        self.traces_threshold_sensors: list[NewTracesThresholdSensor] = (
            traces_threshold_sensors or []
        )
        self.new_dataset_sensors: list[NewDatasetSensor] = new_dataset_sensors or []
        # Collected during each tick for async handler coroutines.
        self._pending: list[Awaitable] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def tick(self, now: Optional[datetime] = None) -> TickResult:
        """Run one cycle across all sensors. `now` is threaded into
        cadence sensors so tests can advance wall-clock deterministically.
        """
        self._pending = []
        result = TickResult()

        # Cadence sensors first — heartbeat signals go out before the
        # more expensive regression path runs. Both go through the same
        # policy matcher + dispatcher.
        for sensor in self.cadence_sensors:
            signal = sensor.tick(now=now)
            if signal is not None:
                self._handle_signal(signal, result)

        # Traces-threshold sensors — volume-driven, independent of
        # regression or cadence. Fired signals drive batch recipes
        # like re-clustering.
        for sensor in self.traces_threshold_sensors:
            signal = sensor.tick(now=now)
            if signal is not None:
                self._handle_signal(signal, result)

        # New-dataset sensors — fire when a clustering run lands.
        # Downstream policies drive metric suggestion + other
        # post-clustering work.
        for sensor in self.new_dataset_sensors:
            signal = sensor.tick(now=now)
            if signal is not None:
                self._handle_signal(signal, result)

        # Regression sensors (one per objective, constructed in __init__)
        for obj in self.objectives:
            signal = self._tick_objective(obj)
            if signal is None:
                continue
            self._handle_signal(signal, result)

        result.pending = list(self._pending)
        self._pending = []
        return result

    def _handle_signal(self, signal: LedgerEntry, result: TickResult) -> None:
        result.signals.append(signal)
        for policy in self._policies_matching(signal):
            if not self._budget_ok(policy):
                result.skipped_budget.append(policy.id)
                continue
            dispatch = self._dispatch(signal, policy)
            result.dispatches.append(dispatch)

    # ------------------------------------------------------------------
    # Sensor-layer invocation
    # ------------------------------------------------------------------

    def _tick_objective(self, objective: Objective) -> Optional[LedgerEntry]:
        """Run the objective's compute_fn, record a measurement
        observation, and feed the result into the regression sensor.

        Recording a measurement row per tick (even when no signal fires)
        is what gives the Objectives dashboard a continuous time-series
        to plot. Without it, the plot only has data at regression
        points — useless for showing trends.

        Graceful-degrade on any compute_fn failure: log, skip, keep the
        loop alive for remaining objectives.
        """
        try:
            fn = resolve_compute_fn(objective)
            measurements = fn()
        except Exception as e:
            logger.warning(
                f"compute_fn for {objective.id} raised {type(e).__name__}: {e}"
            )
            return None

        self._record_measurement(objective, measurements)
        return self.sensors[objective.id].tick(measurements)

    def _record_measurement(
        self, objective: Objective, measurements: list,
    ) -> None:
        """Append an `observation` ledger row summarizing this tick's
        aggregate. Per-dimension detail lives in the raw compute_fn
        output; the row captures enough for the UI plot to draw a
        single point on the objective's time series.
        """
        if not measurements:
            return
        total_weight = 0
        weighted_sum = 0.0
        dims_present: set[str] = set()
        for m in measurements:
            if m.value is None or m.sample_size <= 0:
                continue
            weighted_sum += m.value * m.sample_size
            total_weight += m.sample_size
            dims_present.update(m.dimension_values.keys())
        if total_weight == 0:
            return

        entry = LedgerEntry(
            type="observation",
            objective_id=objective.id,
            agent="objective_sensor",
            data={
                "value": weighted_sum / total_weight,
                "sample_size": total_weight,
                "measurement_count": len(measurements),
                "unit": objective.unit,
                "dimensions": sorted(dims_present),
            },
            tags=["measurement", objective.id],
        )
        try:
            self.ledger.append(entry)
        except Exception as e:
            logger.debug(f"Failed to append measurement for {objective.id}: {e}")

    # ------------------------------------------------------------------
    # Policy matching
    # ------------------------------------------------------------------

    def _policies_matching(self, signal: LedgerEntry) -> list[Policy]:
        return [p for p in self.policies if self._signal_matches(p, signal)]

    @staticmethod
    def _signal_matches(policy: Policy, signal: LedgerEntry) -> bool:
        if (
            policy.match.objective_id is not None
            and signal.objective_id != policy.match.objective_id
        ):
            return False
        required_tags = set(policy.match.signal_tags)
        if required_tags and not required_tags.issubset(set(signal.tags)):
            return False
        return True

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _budget_ok(self, policy: Policy) -> bool:
        """Count this policy's dispatches in the trailing 24h window.

        Uses a linear scan over `recent()` rather than a dedicated SQL
        path — acceptable at the write volumes we care about, and
        avoids adding a tag index. When dispatches outstrip this, the
        right fix is to push filtering into the store, not to tune the
        window here.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        count = 0
        for e in self.ledger.recent(limit=2000):
            if e.ts < cutoff:
                break
            if "policy_dispatch" in e.tags and policy.id in e.tags:
                count += 1
                if count >= policy.budget.max_per_day:
                    return False
        return count < policy.budget.max_per_day

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, signal: LedgerEntry, policy: Policy) -> LedgerEntry:
        """Always write a `run` entry chained to the signal. Then, if a
        handler is registered, invoke it — handler failures are logged
        but do not block the engine or other policies' dispatches.

        Agent dispatch → looks up `policy.dispatch.agent` in handlers.
        Recipe dispatch → looks up the reserved `__recipe__` handler,
        which the runner registers once and which knows how to invoke
        RecipeExecutor with the recipe id from the policy.
        """
        target = policy.dispatch.agent or policy.dispatch.recipe
        dispatch_tags = ["policy_dispatch", policy.id]
        if target:
            dispatch_tags.append(target)
        if policy.dispatch.recipe:
            dispatch_tags.append("recipe")

        dispatch_entry = LedgerEntry(
            type="run",
            agent=policy.dispatch.agent,  # may be None for recipe dispatch
            parent_id=signal.id,
            parameters_in={
                "policy_id": policy.id,
                "recipe_id": policy.dispatch.recipe,
                **policy.dispatch.parameters,
            },
            tags=dispatch_tags,
        )
        try:
            self.ledger.append(dispatch_entry)
        except Exception as e:
            logger.warning(f"Failed to append dispatch entry: {e}")
            return dispatch_entry

        handler_key = "__recipe__" if policy.dispatch.recipe else policy.dispatch.agent
        handler = self.dispatch_handlers.get(handler_key)
        if handler is not None:
            try:
                maybe_coro = handler(signal, dispatch_entry.id, policy)
                # Async handlers return a coroutine — defer execution
                # so tick() stays synchronous and the runner loop can
                # gather them in one place.
                if inspect.iscoroutine(maybe_coro):
                    self._pending.append(maybe_coro)
            except Exception as e:
                logger.warning(
                    f"Dispatch handler for {handler_key!r} "
                    f"(policy {policy.id}) raised: {e}"
                )
        return dispatch_entry
