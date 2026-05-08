"""Restate service that runs the JSONL→Parquet compactor on a daily cron.

The pattern is a self-rescheduling tick: the handler compacts yesterday's
JSONL, then schedules its own next invocation aligned to the next 00:05 UTC
via ctx.service_send(..., send_delay=...). Restate persists the delayed
call durably, so the schedule survives Restate-server and worker restarts.

Why Restate (vs. plain cron / systemd timer):
  • The compaction step is wrapped in ctx.run() — Restate journals the
    result, retries on failure, and (combined with the date-based
    idempotency key) ensures exactly-once semantics if a tick is replayed.
  • The next-day rescheduling is also durable — no separate cron daemon.
  • Audit trail: every compaction run is queryable via the Restate admin
    UI/API. Useful when you want to know "did 2026-05-07 actually compact?"

Cost: a Restate server has to be running. See infra/restate/README.md for
the honest tradeoffs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import restate
from restate import Context, RunOptions, Service

from runtime.store.compactor import compact_day

logger = restate.getLogger(__name__) if hasattr(restate, "getLogger") else logging.getLogger(__name__)

compactor = Service("compactor")


def _yesterday_iso(now: datetime) -> str:
    return (now.date() - timedelta(days=1)).isoformat()


def _next_run_at(now: datetime, hour: int = 0, minute: int = 5) -> datetime:
    """Next wall-clock 00:05 UTC strictly after `now`."""
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _compact_yesterday(day: str) -> str:
    """The actual side effect — runs inside ctx.run() so Restate journals
    the result and replays it on retry."""
    out: Path | None = compact_day(day, force=False)
    if out is None:
        return f"skipped:{day}"
    n_parts = sum(1 for _ in out.rglob("*.parquet"))
    return f"compacted:{day}:{n_parts}_parts"


@compactor.handler()
async def tick(ctx: Context) -> dict:
    """One daily tick. Idempotent at the Restate layer (date-keyed) and at
    the compactor layer (skips if dt= already exists). Reschedules itself
    for the next UTC 00:05."""
    # ctx.time() returns a journaled timestamp — deterministic across replays.
    now_ts = await ctx.time()
    now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    day = _yesterday_iso(now)

    result = await ctx.run_typed(
        "compact",
        _compact_yesterday,
        RunOptions(max_attempts=5, max_retry_duration=timedelta(minutes=10)),
        day,
    )

    next_at = _next_run_at(now)
    delay = next_at - now
    # Idempotency key per UTC date prevents duplicate ticks if the cron is
    # bootstrapped twice or a stray invocation slips in.
    ctx.service_send(
        tick,
        arg=None,
        send_delay=delay,
        idempotency_key=f"compactor-tick-{next_at.date().isoformat()}",
    )

    return {
        "compacted_day": day,
        "result": result,
        "next_run_utc": next_at.isoformat(),
        "delay_seconds": int(delay.total_seconds()),
    }


@compactor.handler()
async def run_now(ctx: Context, day: str) -> dict:
    """Ad-hoc compaction for a specific YYYY-MM-DD — handy for backfills
    without disturbing the daily tick chain."""
    result = await ctx.run_typed(
        "compact",
        _compact_yesterday,
        RunOptions(max_attempts=3),
        day,
    )
    return {"day": day, "result": result}


app = restate.app(services=[compactor])
