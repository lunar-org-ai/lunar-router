"""Cadence sensor — emits a signal on a fixed time interval.

Complements `ObjectiveSensor` (which emits on regression). Where the
regression sensor is event-driven (fires when something moved), the
cadence sensor is schedule-driven (fires because it's been N hours).
The plan's Step 3 calls out both: cadence triggers the periodic scan,
regression triggers the reactive scan.

The sensor tracks its last-fire time in-memory. On process restart the
next tick acts as the first fire (since we don't have prior state),
which is acceptable for MVP — missing one cadence tick around a
restart is not a correctness issue.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from opentracy.harness.ledger import LedgerEntry, LedgerStore
from opentracy.harness.objectives.schemas import Objective

logger = logging.getLogger(__name__)


SENSOR_TAG_CADENCE = "cadence"


class CadenceSensor:
    """Fires every `interval_hours` wall-clock hours, emitting one
    `signal(type=signal, tags=[cadence, <objective_id>])` ledger entry.

    Construction does not start a timer; the trigger engine's main loop
    calls `.tick(now=...)` on its own schedule. Taking `now` as an arg
    makes behavior deterministic in tests.
    """

    def __init__(
        self,
        objective: Objective,
        ledger: LedgerStore,
        interval_hours: float = 24.0,
    ):
        self.objective = objective
        self.ledger = ledger
        self.interval = timedelta(hours=interval_hours)
        self._last_fire: Optional[datetime] = None

    def tick(self, now: Optional[datetime] = None) -> Optional[LedgerEntry]:
        moment = now or datetime.now(timezone.utc)

        if self._last_fire is None:
            # First tick fires immediately so the system has a heartbeat
            # signal right after boot. Without this the first cadence
            # wouldn't appear until `interval_hours` had elapsed, which
            # delays end-to-end visibility.
            return self._emit(moment)

        if moment - self._last_fire < self.interval:
            return None

        return self._emit(moment)

    def _emit(self, moment: datetime) -> Optional[LedgerEntry]:
        entry = LedgerEntry(
            type="signal",
            objective_id=self.objective.id,
            agent="cadence_sensor",
            data={
                "interval_hours": self.interval.total_seconds() / 3600,
                "cadence": self.objective.update_cadence,
            },
            tags=[SENSOR_TAG_CADENCE, self.objective.id],
        )
        try:
            self.ledger.append(entry)
        except Exception as e:
            logger.warning(
                f"Failed to append cadence signal for {self.objective.id}: {e}"
            )
            return None
        self._last_fire = moment
        return entry
