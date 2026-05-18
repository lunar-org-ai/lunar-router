"""Feedback side-table — one row per CSAT submission.

Layout::

    traces/feedback/<YYYY-MM-DD>.jsonl

Each line: ``{"trace_id": str, "score": 1-5, "comment": str|None, "at": iso}``.

We use a side-table (vs extending the trace JSONL row) because:
  - Trace rows are append-only; feedback arrives asynchronously and
    rewriting the whole partition for one row is wasteful.
  - Multiple feedback submissions per trace are possible (correction
    flow); the side-table keeps each event.
  - The aggregation query joins on `trace_id` like any other side-table.

P15.3.7's `feedback_signals` mining adapter (currently a stub) is the
natural consumer of this data once it lands.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DEFAULT_FEEDBACK_ROOT = Path("traces") / "feedback"
# Back-compat alias — tests monkeypatch this to point at tmp_path. The
# resolver below checks whether the alias was overridden and honors it.
_FEEDBACK_ROOT = _DEFAULT_FEEDBACK_ROOT


def _feedback_root_for(agent_id: Optional[str] = None) -> Path:
    from runtime.agent_context import get_active
    return Path("traces") / (agent_id or get_active()) / "feedback"


def _resolve_root(root: Optional[Path], agent_id: Optional[str] = None) -> Path:
    """Resolution order:
      1. explicit ``root=`` param (preferred for new callsites)
      2. monkeypatched ``_FEEDBACK_ROOT`` module attr (back-compat tests)
      3. partition under the active agent
    """
    if root is not None:
        return root
    if _FEEDBACK_ROOT != _DEFAULT_FEEDBACK_ROOT:
        return _FEEDBACK_ROOT
    return _feedback_root_for(agent_id)


def write_feedback(
    trace_id: str,
    score: int,
    comment: Optional[str] = None,
    *,
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> dict:
    """Append a feedback row for `trace_id`. Returns the row written.

    Raises ValueError for out-of-range scores; callers (the API handler)
    convert that to 400 for the client.
    """
    if not isinstance(score, int) or not (1 <= score <= 5):
        raise ValueError(f"score must be int in [1, 5], got {score!r}")

    root = _resolve_root(root)
    root.mkdir(parents=True, exist_ok=True)

    at = now_iso or _now_iso()
    row = {
        "trace_id": trace_id,
        "score": int(score),
        "comment": comment if comment else None,
        "at": at,
    }
    date = at[:10]
    target = root / f"{date}.jsonl"
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def iter_feedback(
    *,
    since_iso: Optional[str] = None,
    root: Optional[Path] = None,
):
    """Stream feedback rows from JSONL partitions, date-filtered."""
    root = _resolve_root(root)
    if not root.exists():
        return
    since_date = (since_iso or "")[:10] if since_iso else ""
    for path in sorted(root.glob("*.jsonl")):
        date = path.stem
        if since_date and date < since_date:
            continue
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue


def list_feedback_for_trace(
    trace_id: str,
    *,
    root: Optional[Path] = None,
) -> list[dict]:
    """All feedback rows for one trace (most recent last)."""
    return [r for r in iter_feedback(root=root) if r.get("trace_id") == trace_id]


def csat_for_window(
    window_days: int = 7,
    *,
    root: Optional[Path] = None,
) -> Optional[float]:
    """Mean feedback score in the trailing `window_days`. None when empty."""
    if window_days <= 0:
        return None
    cutoff_date = (
        datetime.now(timezone.utc).date()
        - _timedelta_days(window_days - 1)
    ).isoformat()
    total = 0.0
    n = 0
    for row in iter_feedback(since_iso=cutoff_date, root=root):
        try:
            total += float(row["score"])
            n += 1
        except (KeyError, TypeError, ValueError):
            continue
    if n == 0:
        return None
    return round(total / n, 3)


def _now_iso() -> str:
    # Microsecond precision so back-to-back submissions (corrections,
    # tests) tie-break cleanly when the aggregator picks "latest".
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _timedelta_days(n: int):
    from datetime import timedelta
    return timedelta(days=n)
