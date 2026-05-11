"""Auto-rollback watcher (P16.4).

The Policies UI has long carried an ``auto_rollback`` block — CSAT drop
threshold, resolution-rate drop threshold, window, notify channels —
but nothing watched production metrics and triggered rollback. With
P16.2 (real cost + CSAT signals) and P16.3 (CSAT/feedback channel
populated by users), the upstream signals finally exist.

This module:

  1. Finds the most recent ``promote`` ledger entry (the "last
     promotion under suspicion").
  2. Compares CSAT + resolution_rate AFTER the promotion to
     BEFORE the promotion, both in the same trailing window.
  3. If CSAT dropped by ≥ ``policy.auto_rollback.csat_drop`` OR
     resolution dropped by ≥ ``policy.auto_rollback.resolution_drop``,
     the check returns a ``RollbackDecision`` carrying the target
     version (the agent version that was live before the promote).
  4. Already-rolled-back promotions are idempotently ignored —
     a rollback entry that post-dates the last promote means the
     decision was already executed.

The watcher is PURE LOGIC: it never calls ``rollback_to()`` on its
own. The caller (``harness.wakeup.runner.run_wakeup``) executes the
rollback + writes the Lesson + notifies channels. This keeps the
watcher trivially testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger("harness.watchers.auto_rollback")


@dataclass(frozen=True)
class WindowMetrics:
    """Aggregate metrics over a [since_iso, until_iso) window."""
    csat: Optional[float]
    resolution_rate: Optional[float]
    n_traces: int
    n_feedback: int

    def has_signal(self) -> bool:
        """True iff we have at least one metric (CSAT OR resolution)."""
        return self.csat is not None or self.resolution_rate is not None


@dataclass(frozen=True)
class RollbackDecision:
    """A non-None return from ``check_auto_rollback``.

    Carries everything the executor needs to roll back + tell the
    operator why. ``target_version`` is the version that was live
    BEFORE the suspect promotion (so rolling there undoes the bad
    change).
    """
    target_version: str
    suspect_version: str
    reason: str  # human-readable, e.g. "CSAT dropped 4.5 → 3.1 in 24h"
    before: WindowMetrics
    after: WindowMetrics
    promote_entry_id: str
    promote_timestamp: str


def check_auto_rollback(
    *,
    policy,                              # harness.approver.policy.Policy
    now_iso: Optional[str] = None,
    entries_iter: Optional[Iterable] = None,
    feedback_iter: Optional[Iterable] = None,
    traces_iter: Optional[Iterable] = None,
) -> Optional[RollbackDecision]:
    """Decide whether the last promotion warrants an auto-rollback.

    All iterators default to the production stores; tests inject
    in-memory iterables to drive each branch.

    Returns ``None`` when:
      - there's no recent promote to evaluate
      - the promote was already rolled back (entry post-dates it)
      - the window has no signal (no feedback + no traces in window)
      - no metric exceeds the policy threshold
    """
    if entries_iter is None:
        from ledger.writer import read_entries
        entries_iter = read_entries()

    entries = list(entries_iter)
    promote = _last_promote(entries)
    if promote is None:
        logger.debug("no recent promote — nothing to roll back")
        return None

    if _has_later_rollback(entries, after_id=promote["entry_id"], after_ts=promote["timestamp"]):
        logger.debug("promote %s already rolled back — skipping", promote["entry_id"])
        return None

    target_version = (promote.get("agent_version_before") or "").strip()
    if not target_version:
        logger.debug("promote %s has no agent_version_before — can't determine target",
                     promote["entry_id"])
        return None

    suspect_version = (promote.get("agent_version_after") or "").strip() or target_version

    now = _parse_iso(now_iso) or datetime.now(timezone.utc)
    promote_dt = _parse_iso(promote["timestamp"])
    if promote_dt is None:
        return None

    window_h = max(1, int(policy.auto_rollback.window_hours))
    half = timedelta(hours=window_h)
    after_start = promote_dt
    after_end = min(now, promote_dt + half)
    before_start = promote_dt - half
    before_end = promote_dt

    if after_end <= after_start:
        # not enough time elapsed since the promote to evaluate
        return None

    feedback_rows = list(_resolve_feedback_iter(feedback_iter))
    traces_rows = list(_resolve_traces_iter(traces_iter, after_start - half))

    before = _aggregate(
        feedback_rows, traces_rows,
        since=_iso(before_start), until=_iso(before_end),
    )
    after = _aggregate(
        feedback_rows, traces_rows,
        since=_iso(after_start), until=_iso(after_end),
    )

    # Require some signal in BOTH windows so a brand-new feedback
    # channel doesn't immediately fire (before=None vs after=4.5
    # would look like an infinite drop).
    if not (before.has_signal() and after.has_signal()):
        return None

    reasons: list[str] = []
    if (
        before.csat is not None
        and after.csat is not None
        and (before.csat - after.csat) >= policy.auto_rollback.csat_drop
    ):
        reasons.append(
            f"CSAT dropped {before.csat:.2f} → {after.csat:.2f} "
            f"(threshold {policy.auto_rollback.csat_drop:.2f}, window {window_h}h)"
        )
    if (
        before.resolution_rate is not None
        and after.resolution_rate is not None
        and (before.resolution_rate - after.resolution_rate)
        >= policy.auto_rollback.resolution_drop
    ):
        reasons.append(
            f"resolution_rate dropped "
            f"{before.resolution_rate * 100:.1f}% → {after.resolution_rate * 100:.1f}% "
            f"(threshold {policy.auto_rollback.resolution_drop * 100:.1f}pp, "
            f"window {window_h}h)"
        )

    if not reasons:
        return None

    reason = "; ".join(reasons)
    return RollbackDecision(
        target_version=target_version,
        suspect_version=suspect_version,
        reason=reason,
        before=before,
        after=after,
        promote_entry_id=promote["entry_id"],
        promote_timestamp=promote["timestamp"],
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _last_promote(entries: list) -> Optional[dict]:
    """Find the most recent entry of kind=promote.

    Accepts both LedgerEntry dataclass instances and plain dicts so
    tests can construct in-memory entries without importing the type.
    """
    best: Optional[dict] = None
    best_ts = ""
    for raw in entries:
        e = _as_dict(raw)
        if e.get("kind") != "promote":
            continue
        ts = e.get("timestamp") or ""
        if ts > best_ts:
            best = e
            best_ts = ts
    return best


def _has_later_rollback(entries: list, *, after_id: str, after_ts: str) -> bool:
    """True iff the ledger has a rollback entry that post-dates `after_ts`.

    We check timestamp (string-compared, ISO sorts lex == temporal) so
    a manual rollback also counts — we never auto-roll-back twice.
    """
    for raw in entries:
        e = _as_dict(raw)
        if e.get("kind") != "rollback":
            continue
        if (e.get("timestamp") or "") > after_ts:
            return True
    return False


def _aggregate(
    feedback_rows: list,
    traces_rows: list,
    *,
    since: str,
    until: str,
) -> WindowMetrics:
    """CSAT + resolution_rate + counts in [since, until)."""
    # CSAT — mean of feedback scores whose `at` falls in the window.
    csat_total = 0.0
    csat_n = 0
    for r in feedback_rows:
        at = (r.get("at") or "")
        if not (since <= at < until):
            continue
        try:
            csat_total += float(r["score"])
            csat_n += 1
        except (KeyError, TypeError, ValueError):
            continue
    csat = (csat_total / csat_n) if csat_n > 0 else None

    # Resolution — fraction of traces where success=True and no stage error.
    n_traces = 0
    n_resolved = 0
    for t in traces_rows:
        ts = (t.get("timestamp") or "")
        if not (since <= ts < until):
            continue
        n_traces += 1
        if _trace_resolved(t):
            n_resolved += 1
    resolution = (n_resolved / n_traces) if n_traces > 0 else None

    return WindowMetrics(
        csat=csat,
        resolution_rate=resolution,
        n_traces=n_traces,
        n_feedback=csat_n,
    )


def _trace_resolved(t: dict) -> bool:
    """Mirror metrics_traces_window's resolution definition: response set,
    success=True, and no stage error."""
    if not t.get("success"):
        return False
    if t.get("response") is None:
        return False
    for s in (t.get("stages") or []):
        if isinstance(s, dict) and s.get("error"):
            return False
    return True


def _resolve_feedback_iter(it):
    if it is not None:
        return it
    from runtime.store.feedback import iter_feedback
    return iter_feedback()


def _resolve_traces_iter(it, since: datetime):
    if it is not None:
        return it
    # Stream traces partitions across the union view; the test stub
    # short-circuits this so production code doesn't need a fancy
    # window-aware iterator.
    from runtime.store.traces import _connect, _traces_view_sql, _fetch_dicts

    with _connect() as con:
        src = _traces_view_sql(con)
        if src is None:
            return []
        # Pull every trace from `since` onward; aggregation re-filters
        # to the exact [start, end) window so a single fetch covers
        # both the before+after halves.
        cur = con.execute(
            f"WITH t AS ({src}) SELECT * FROM t "
            "WHERE substr(timestamp, 1, 10) >= ?",
            [_iso(since)[:10]],
        )
        return _fetch_dicts(cur)


def _as_dict(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    # LedgerEntry dataclass instance — dataclasses.asdict is overkill here.
    return {
        "entry_id": getattr(raw, "entry_id", ""),
        "kind": getattr(raw, "kind", ""),
        "timestamp": getattr(raw, "timestamp", ""),
        "agent_version_before": getattr(raw, "agent_version_before", None),
        "agent_version_after": getattr(raw, "agent_version_after", None),
        "summary": getattr(raw, "summary", ""),
        "payload": getattr(raw, "payload", {}) or {},
    }


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")
