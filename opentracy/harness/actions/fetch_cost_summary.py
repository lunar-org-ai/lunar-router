"""`fetch_cost_summary` action — pulls per-model cost + error rates for
a trailing window so the training_advisor agent has real data to reason
about. Without this, the advisor just sees the raw signal's delta_pct,
which is too thin for nuanced decisions.

Output shape (consumed by training_advisor via input_from):
  - window_hours: int
  - trace_count: int
  - by_model: [{model, error_rate, avg_cost_usd, avg_latency_ms, count}]
  - worst_model_by_cost: str
  - worst_cost_increase_pct: float (vs domain average)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from opentracy.harness.ledger import LedgerEntry, LedgerStore

from .base import ActionResult, register_action

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_HOURS = 48


@register_action("fetch_cost_summary")
async def execute(
    inputs: dict[str, Any],
    ledger: LedgerStore,
    parent_id: str,
) -> ActionResult:
    started = datetime.now(timezone.utc)
    window_hours = int(inputs.get("window_hours", _DEFAULT_WINDOW_HOURS))

    try:
        from opentracy.storage.clickhouse_client import get_client
    except Exception as e:
        return _fail(ledger, parent_id, started, f"import failed: {e}")

    client = get_client()
    if client is None:
        return _fail(ledger, parent_id, started, "clickhouse unavailable")

    start = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    try:
        r = client.query(
            "SELECT selected_model, "
            "       countIf(is_error = 1) / count() AS error_rate, "
            "       avg(total_cost_usd) AS avg_cost_usd, "
            "       avg(latency_ms) AS avg_latency_ms, "
            "       count() AS n "
            "FROM llm_traces "
            "WHERE timestamp >= {start:DateTime64(3)} "
            "GROUP BY selected_model "
            "HAVING n > 0 "
            "ORDER BY avg_cost_usd DESC",
            parameters={"start": start},
        )
    except Exception as e:
        return _fail(ledger, parent_id, started, f"ch query failed: {e}")

    rows = r.result_rows or []
    if not rows:
        return _fail(ledger, parent_id, started, "no traces in window")

    by_model = []
    for row in rows:
        by_model.append({
            "model": row[0],
            "error_rate": float(row[1]) if row[1] is not None else 0.0,
            "avg_cost_usd": float(row[2]) if row[2] is not None else 0.0,
            "avg_latency_ms": float(row[3]) if row[3] is not None else 0.0,
            "count": int(row[4]) if row[4] is not None else 0,
        })

    total_count = sum(m["count"] for m in by_model)
    weighted_cost = (
        sum(m["avg_cost_usd"] * m["count"] for m in by_model) / total_count
        if total_count > 0 else 0.0
    )
    worst = max(by_model, key=lambda m: m["avg_cost_usd"])
    pct_above_avg = (
        (worst["avg_cost_usd"] - weighted_cost) / weighted_cost * 100
        if weighted_cost > 0 else 0.0
    )

    payload = {
        "window_hours": window_hours,
        "trace_count": total_count,
        "by_model": by_model,
        "worst_model_by_cost": worst["model"],
        "worst_cost_increase_pct": round(pct_above_avg, 2),
        "domain_avg_cost_usd": round(weighted_cost, 6),
    }

    entry = LedgerEntry(
        type="action",
        agent="fetch_cost_summary",
        parent_id=parent_id,
        data=payload,
        tags=["fetch_cost_summary"],
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
    ledger: LedgerStore, parent_id: str, started: datetime, error: str,
) -> ActionResult:
    entry = LedgerEntry(
        type="action",
        agent="fetch_cost_summary",
        parent_id=parent_id,
        data={"error": error},
        tags=["fetch_cost_summary", "failed"],
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
