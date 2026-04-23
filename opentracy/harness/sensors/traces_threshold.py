"""NewTracesThresholdSensor — fires when enough fresh traces have
accumulated since the last fire.

Complements ObjectiveSensor (regression-driven) and CadenceSensor
(time-driven): this one is volume-driven. Emits a `signal` that
downstream policies can match to trigger expensive batch work like
clustering or re-embedding, which is wasteful to run on a fixed
cadence but wasteful NOT to run when traffic surges.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from opentracy.harness.ledger import LedgerEntry, LedgerStore

logger = logging.getLogger(__name__)


SENSOR_TAG_TRACES_THRESHOLD = "new_traces_threshold"


# Default count function reads ClickHouse; tests inject a stub.
def _default_count_fn(since: datetime) -> int:
    try:
        from opentracy.storage.clickhouse_client import query_trace_count
    except Exception:
        return 0
    try:
        return query_trace_count(start=since)
    except Exception as e:
        logger.warning(f"query_trace_count failed in traces-threshold sensor: {e}")
        return 0


class NewTracesThresholdSensor:
    """Emits a signal when trace volume since the last fire crosses a
    threshold. Cooldown prevents rapid re-fires.

    Construction parameters:
      - objective: the objective this sensor is tied to (for time-series
        indexing + policy matching on objective_id).
      - threshold: minimum count of new traces required to fire.
      - cooldown_hours: minimum wall-clock gap between fires, enforced
        by the sensor even if the threshold is crossed multiple times
        in quick succession.
      - count_fn: `(since: datetime) → int`. Defaults to the shipped
        ClickHouse-backed query; tests pass a stub.
    """

    def __init__(
        self,
        objective,
        ledger: LedgerStore,
        threshold: int = 1000,
        cooldown_hours: float = 6.0,
        count_fn: Optional[Callable[[datetime], int]] = None,
    ):
        self.objective = objective
        self.ledger = ledger
        self.threshold = threshold
        self.cooldown = timedelta(hours=cooldown_hours)
        self._count_fn = count_fn or _default_count_fn
        self._last_fire: Optional[datetime] = None
        # Anchor the first count window from sensor construction. Without
        # this the sensor would always see the full CH history on the
        # first tick and fire spuriously.
        self._since: datetime = datetime.now(timezone.utc)

    def tick(self, now: Optional[datetime] = None) -> Optional[LedgerEntry]:
        moment = now or datetime.now(timezone.utc)

        if self._last_fire is not None and moment - self._last_fire < self.cooldown:
            return None

        try:
            count = int(self._count_fn(self._since))
        except Exception as e:
            logger.warning(
                f"traces-threshold count_fn raised for {self.objective.id}: {e}"
            )
            return None

        if count < self.threshold:
            return None

        entry = LedgerEntry(
            type="signal",
            objective_id=self.objective.id,
            agent="new_traces_threshold_sensor",
            data={
                "trace_count": count,
                "threshold": self.threshold,
                "window_start": self._since.isoformat(),
                "window_end": moment.isoformat(),
            },
            tags=[SENSOR_TAG_TRACES_THRESHOLD, self.objective.id],
        )
        try:
            self.ledger.append(entry)
        except Exception as e:
            logger.warning(
                f"Failed to append new_traces_threshold signal for "
                f"{self.objective.id}: {e}"
            )
            return None

        # Advance the window anchor so the next tick counts only traces
        # that arrived AFTER this fire.
        self._last_fire = moment
        self._since = moment
        return entry
