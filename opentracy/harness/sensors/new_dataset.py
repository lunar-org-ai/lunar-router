"""NewDatasetSensor — fires when a new clustering run lands in
ClickHouse.

Complements the traces-threshold sensor: where that one triggers the
clustering pipeline, this one triggers what happens AFTER clustering
(metrics suggestions for the freshly-produced datasets). Emits a
`signal` per newly-discovered `clustering_runs.run_id`.

Deduplication: the sensor keeps the latest `created_at` timestamp it
has emitted a signal for. On each tick it queries for rows newer than
that. No cross-process persistence — a restart re-anchors at sensor
construction, which is acceptable (missing one suggestion batch around
a restart is not a correctness issue).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from opentracy.harness.ledger import LedgerEntry, LedgerStore

logger = logging.getLogger(__name__)


SENSOR_TAG_NEW_DATASET = "new_dataset"


_FindFn = Callable[[datetime], list[dict]]


def _default_find_fn(since: datetime) -> list[dict]:
    """Query clustering_runs for rows newer than `since`, oldest first.

    Returns dicts with `run_id`, `created_at`, `num_clusters`,
    `silhouette_score`, `total_traces`. Empty list when CH unavailable
    or no new rows.
    """
    try:
        from opentracy.storage.clickhouse_client import get_client
    except Exception:
        return []
    client = get_client()
    if client is None:
        return []
    try:
        r = client.query(
            "SELECT run_id, created_at, num_clusters, silhouette_score, total_traces "
            "FROM clustering_runs "
            "WHERE created_at > {since:DateTime64(3)} "
            "ORDER BY created_at ASC",
            parameters={"since": since},
        )
    except Exception as e:
        logger.warning(f"new_dataset_sensor: query failed: {e}")
        return []
    rows = []
    for row in r.result_rows or []:
        rows.append({
            "run_id": row[0],
            "created_at": row[1],
            "num_clusters": int(row[2]) if row[2] is not None else None,
            "silhouette_score": float(row[3]) if row[3] is not None else None,
            "total_traces": int(row[4]) if row[4] is not None else None,
        })
    return rows


class NewDatasetSensor:
    """Emits a signal when a new clustering run has landed since the
    last fire. One signal per newest run per tick (older intervening
    runs are collapsed into the latest)."""

    def __init__(
        self,
        objective,
        ledger: LedgerStore,
        find_fn: Optional[_FindFn] = None,
    ):
        self.objective = objective
        self.ledger = ledger
        self._find_fn = find_fn or _default_find_fn
        self._since: datetime = datetime.now(timezone.utc)

    def tick(self, now: Optional[datetime] = None) -> Optional[LedgerEntry]:
        try:
            rows = self._find_fn(self._since)
        except Exception as e:
            logger.warning(
                f"new_dataset_sensor find_fn raised for {self.objective.id}: {e}"
            )
            return None
        if not rows:
            return None

        # Emit for the newest row; advance anchor so older rows in the
        # same set, if any, don't trigger on the next tick.
        newest = max(rows, key=lambda r: r["created_at"])
        entry = LedgerEntry(
            type="signal",
            objective_id=self.objective.id,
            agent="new_dataset_sensor",
            data={
                "run_id": newest["run_id"],
                "num_clusters": newest["num_clusters"],
                "silhouette_score": newest["silhouette_score"],
                "total_traces": newest["total_traces"],
                "collapsed_count": len(rows),
            },
            tags=[SENSOR_TAG_NEW_DATASET, self.objective.id],
        )
        try:
            self.ledger.append(entry)
        except Exception as e:
            logger.warning(
                f"Failed to append new_dataset signal for {self.objective.id}: {e}"
            )
            return None

        # Ensure datetime is tz-aware before advancing the anchor;
        # ClickHouse returns naive datetimes by default.
        ts = newest["created_at"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self._since = ts
        return entry
