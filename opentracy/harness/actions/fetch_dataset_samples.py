"""`fetch_dataset_samples` action — pulls sample prompts + domain from
the latest clustering run's top cluster so the next recipe step (the
metrics_suggester agent) has something to reason about.

The action needs to know WHICH clustering run to target. Three fallbacks
in order of preference:
  1. `inputs["run_id"]` — explicit, set by caller / previous step.
  2. Latest row from `clustering_runs` by `created_at` — the default
     when the recipe is triggered by the `new_dataset` signal.
  3. Failure — return outcome=failed if neither is available.

Output dict (consumed by `metrics_suggester` via `input_from`):
  - sample_prompts: list[str] (≤10)
  - domain: str
  - run_id: str
  - cluster_id: int
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from opentracy.harness.ledger import LedgerEntry, LedgerStore

from .base import ActionResult, register_action

logger = logging.getLogger(__name__)


_SAMPLE_LIMIT = 10


@register_action("fetch_dataset_samples")
async def execute(
    inputs: dict[str, Any],
    ledger: LedgerStore,
    parent_id: str,
) -> ActionResult:
    started = datetime.now(timezone.utc)

    try:
        from opentracy.storage.clickhouse_client import get_client
    except Exception as e:
        return _fail(ledger, parent_id, started, f"import failed: {e}")

    client = get_client()
    if client is None:
        return _fail(ledger, parent_id, started, "clickhouse unavailable")

    run_id = inputs.get("run_id")
    if not run_id:
        run_id = _latest_run_id(client)
    if not run_id:
        return _fail(ledger, parent_id, started, "no clustering_runs found")

    top = _top_cluster(client, run_id)
    if top is None:
        return _fail(ledger, parent_id, started, f"no clusters for run {run_id!r}")

    cluster_id, domain = top
    samples = _sample_prompts(client, run_id, cluster_id)

    payload = {
        "run_id": run_id,
        "cluster_id": cluster_id,
        "domain": domain,
        "sample_prompts": samples,
        "sample_count": len(samples),
    }

    entry = LedgerEntry(
        type="action",
        agent="fetch_dataset_samples",
        parent_id=parent_id,
        data=payload,
        tags=["fetch_dataset_samples"],
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


def _latest_run_id(client) -> Optional[str]:
    try:
        r = client.query(
            "SELECT run_id FROM clustering_runs ORDER BY created_at DESC LIMIT 1"
        )
    except Exception as e:
        logger.warning(f"latest clustering_run lookup failed: {e}")
        return None
    if not r.result_rows:
        return None
    return r.result_rows[0][0]


def _top_cluster(client, run_id: str) -> Optional[tuple[int, str]]:
    """Pick the cluster with the most traces — it's the one humans are
    most likely to want metrics for first."""
    try:
        r = client.query(
            "SELECT cluster_id, domain_label FROM cluster_datasets "
            "WHERE run_id = {rid:String} "
            "ORDER BY trace_count DESC LIMIT 1",
            parameters={"rid": run_id},
        )
    except Exception as e:
        logger.warning(f"top_cluster lookup failed: {e}")
        return None
    if not r.result_rows:
        return None
    cid = int(r.result_rows[0][0])
    label = r.result_rows[0][1] or "general"
    return cid, label


def _sample_prompts(client, run_id: str, cluster_id: int) -> list[str]:
    try:
        r = client.query(
            "SELECT input_text FROM trace_cluster_map "
            "WHERE run_id = {rid:String} AND cluster_id = {cid:UInt32} "
            f"LIMIT {_SAMPLE_LIMIT}",
            parameters={"rid": run_id, "cid": cluster_id},
        )
    except Exception as e:
        logger.warning(f"sample_prompts lookup failed: {e}")
        return []
    return [row[0] for row in (r.result_rows or []) if row[0]]


def _fail(
    ledger: LedgerStore, parent_id: str, started: datetime, error: str,
) -> ActionResult:
    entry = LedgerEntry(
        type="action",
        agent="fetch_dataset_samples",
        parent_id=parent_id,
        data={"error": error},
        tags=["fetch_dataset_samples", "failed"],
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


def _elapsed_ms(started: datetime) -> int:
    return int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
