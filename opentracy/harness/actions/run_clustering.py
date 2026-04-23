"""`run_clustering` action — invokes the clustering pipeline when a
policy fires. Thin wrapper: pipeline logic stays in
`opentracy/clustering/pipeline.py`, the action's only job is to
translate recipe inputs into pipeline arguments and write a ledger row
capturing the outcome.

Inputs it understands (from the recipe step's `input_from` source or
policy parameters):
  - days (int, default 7)          — trailing window of traces to cluster
  - min_traces (int, default 50)   — minimum traces required to run
  - strategy (str, default "auto") — clustering strategy identifier

Outputs in the ledger row's `data`:
  - run_id, clusters_created, silhouette, trace_count
  - error (on failure)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from opentracy.harness.ledger import LedgerEntry, LedgerStore

from .base import ActionResult, register_action

logger = logging.getLogger(__name__)


_DEFAULT_DAYS = 7
_DEFAULT_MIN_TRACES = 50
_DEFAULT_STRATEGY = "auto"


@register_action("run_clustering")
async def execute(
    inputs: dict[str, Any],
    ledger: LedgerStore,
    parent_id: str,
) -> ActionResult:
    started = datetime.now(timezone.utc)

    days = int(inputs.get("days", _DEFAULT_DAYS))
    min_traces = int(inputs.get("min_traces", _DEFAULT_MIN_TRACES))
    strategy = str(inputs.get("strategy", _DEFAULT_STRATEGY))

    try:
        from opentracy.clustering.pipeline import ClusteringPipeline
    except Exception as e:
        return _fail(
            ledger, parent_id, started,
            error=f"ClusteringPipeline import failed: {e}",
            tag="import_failed",
        )

    try:
        pipeline = ClusteringPipeline(strategy=strategy)
        result = await pipeline.run(days=days, min_traces=min_traces)
    except Exception as e:
        logger.warning(f"clustering run failed: {type(e).__name__}: {e}")
        return _fail(
            ledger, parent_id, started,
            error=f"{type(e).__name__}: {e}",
            tag="pipeline_failed",
        )

    # `result` is a ClusteringResult or similar; pull the fields we know
    # exist on the shipped implementation defensively so a schema change
    # in the pipeline doesn't crash the action.
    payload = {
        "run_id": _safe_attr(result, "run_id"),
        "clusters_created": _safe_attr(result, "num_clusters"),
        "silhouette": _safe_attr(result, "silhouette"),
        "trace_count": _safe_attr(result, "trace_count"),
        "days": days,
        "strategy": strategy,
    }

    entry = LedgerEntry(
        type="action",
        agent="run_clustering",
        parent_id=parent_id,
        data={k: v for k, v in payload.items() if v is not None},
        tags=["run_clustering", "clustering"],
        outcome="ok",
        duration_ms=_elapsed_ms(started),
    )
    ledger.append(entry)

    return ActionResult(
        outcome="ok",
        data=payload,
        duration_ms=entry.duration_ms or 0,
        ledger_entry_id=entry.id,
    )


def _fail(
    ledger: LedgerStore, parent_id: str, started: datetime,
    error: str, tag: str,
) -> ActionResult:
    entry = LedgerEntry(
        type="action",
        agent="run_clustering",
        parent_id=parent_id,
        data={"error": error},
        tags=["run_clustering", tag],
        outcome="failed",
        duration_ms=_elapsed_ms(started),
    )
    ledger.append(entry)
    return ActionResult(
        outcome="failed",
        data={"error": error},
        duration_ms=entry.duration_ms or 0,
        ledger_entry_id=entry.id,
    )


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        v = getattr(obj, name, None)
    except Exception:
        return None
    # Coerce numpy scalars and similar so JSON serialization stays clean.
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def _elapsed_ms(started: datetime) -> int:
    return int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
