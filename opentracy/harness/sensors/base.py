"""Objective-driven sensor: compares successive compute_fn results and
emits a ledger `signal` when the objective regresses past its guardrail.

The sensor is deliberately dumb: it does not call compute_fns itself
and does not know about the trigger engine. Callers pass measurements
in, the sensor produces at most one signal per tick, and it holds the
baseline in-memory. This lets the engine drive timing, makes the sensor
trivially testable, and keeps ClickHouse I/O out of the hot path.
"""

from __future__ import annotations

import logging
from typing import Optional

from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.objectives.schemas import (
    Direction,
    Objective,
    ObjectiveMeasurement,
)

logger = logging.getLogger(__name__)


SENSOR_TAG_REGRESSION = "objective_regression"

# Guardrail key name we look for on the objective. Centralized here so
# a rename in YAML stays a one-line change.
_REGRESSION_GUARDRAIL_TYPE = "no_regression_worse_than_pct"


class ObjectiveSensor:
    """Emits a `signal` ledger entry when the objective worsens past its
    configured percentage threshold.

    Aggregation: the current value is a sample-size-weighted mean across
    all dimension values returned by the compute_fn. Per-dimension drift
    detection is a future refinement; for now a single aggregate is
    sufficient to close the observability loop and avoids fanning out
    one signal per (objective × dimension) combination.
    """

    def __init__(self, objective: Objective, ledger: LedgerStore):
        self.objective = objective
        self.ledger = ledger
        # In-memory baseline. Lost on process restart; re-established on
        # the first tick after restart. Acceptable for MVP since the loop
        # runs every ~60s and missing one regression signal around a
        # restart is tolerable.
        self._last_value: Optional[float] = None

    def tick(self, measurements: list[ObjectiveMeasurement]) -> Optional[LedgerEntry]:
        """Feed one tick of measurements. Returns the emitted signal
        (already appended to the ledger) or None if nothing fired.

        Returns None when:
          - `measurements` is empty (compute_fn degraded)
          - this is the first tick (baseline being established)
          - the delta is within the guardrail threshold
          - the movement is an improvement, not a regression
        """
        current = _aggregate(measurements)
        if current is None:
            # Either no measurements or all values were None. Nothing to
            # compare against; do not emit, do not update baseline.
            return None

        # Always update baseline at the end. Early-return paths below
        # must mirror this so the next tick has a fresh reference point.
        if self._last_value is None:
            self._last_value = current
            return None

        delta_pct = _signed_delta_pct(
            baseline=self._last_value,
            current=current,
            direction=self.objective.direction,
        )
        threshold = self._regression_threshold_pct()
        self._last_value = current

        if threshold is None:
            # Objective declared no regression guardrail → nothing to
            # trip on. Still useful to track via time_series, but no
            # ledger signal.
            return None

        if delta_pct <= threshold:
            # Either improvement (delta_pct <= 0) or movement within
            # tolerance. No signal.
            return None

        entry = LedgerEntry(
            type="signal",
            objective_id=self.objective.id,
            agent="objective_sensor",
            data={
                "baseline": self._last_value_before(current, delta_pct),
                "current": current,
                "delta_pct": round(delta_pct, 3),
                "threshold_pct": threshold,
                "direction": self.objective.direction,
                "measurement_count": len(measurements),
            },
            tags=[SENSOR_TAG_REGRESSION, self.objective.id],
        )
        try:
            self.ledger.append(entry)
        except Exception as e:
            logger.warning(
                f"Failed to append signal for {self.objective.id}: {e}"
            )
            return None
        return entry

    def _regression_threshold_pct(self) -> Optional[float]:
        for g in self.objective.guardrails:
            if g.type == _REGRESSION_GUARDRAIL_TYPE and g.threshold is not None:
                return float(g.threshold)
        return None

    def _last_value_before(self, current: float, delta_pct: float) -> float:
        """Reverse-engineer the baseline from current + delta_pct so the
        emitted entry records the actual prior value, not the value we
        already mutated self._last_value to."""
        # delta_pct is % worse; for lower_is_better: current = baseline*(1+delta/100)
        # for higher_is_better: current = baseline*(1-delta/100).
        factor = 1 + delta_pct / 100
        if self.objective.direction == "higher_is_better":
            factor = 1 - delta_pct / 100
        if factor == 0:
            return current
        return round(current / factor, 6)


def _aggregate(measurements: list[ObjectiveMeasurement]) -> Optional[float]:
    """Sample-size-weighted mean across measurements. Returns None if
    there are no measurements with both a value and a positive sample
    size."""
    total_weight = 0
    weighted_sum = 0.0
    for m in measurements:
        if m.value is None:
            continue
        if m.sample_size <= 0:
            continue
        weighted_sum += m.value * m.sample_size
        total_weight += m.sample_size
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def _signed_delta_pct(
    baseline: float,
    current: float,
    direction: Direction,
) -> float:
    """Return the percentage by which `current` is WORSE than `baseline`.

    Positive = regression (worse), zero or negative = stable or improved.
    Normalizes across both objective directions so the caller can
    compare against a single positive threshold.
    """
    if baseline == 0:
        # Degenerate case: can't compute % change from zero. Treat any
        # nonzero current as infinite regression in the bad direction.
        if direction == "lower_is_better":
            return 0.0 if current <= 0 else float("inf")
        return 0.0 if current >= 0 else float("inf")

    raw = (current - baseline) / baseline * 100
    if direction == "lower_is_better":
        # Increase = worse.
        return raw
    # higher_is_better: decrease = worse → invert sign.
    return -raw
