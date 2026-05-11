"""Trace flag side-table — operator-marked failures.

Layout::

    traces/flagged/<YYYY-MM-DD>.jsonl

Each line: ``{"trace_id": str, "reason": str|None, "source": str, "at": iso}``.

``source`` distinguishes manual operator flags (``"manual"``) from
automated rule-based flags (``"csat_low"``, ``"latency_outlier"``,
etc.). Multiple rows per trace are possible — auto-flag + later manual
flag both persist. The aggregate "is flagged?" check is "any row
exists and no later unflag row".

Unflag is recorded as a row with ``source="unflag"``. The latest row
(by `at`) for a trace wins.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_FLAGS_ROOT = Path("traces") / "flagged"


def write_flag(
    trace_id: str,
    *,
    reason: Optional[str] = None,
    source: str = "manual",
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> dict:
    """Append a flag row for `trace_id`. Returns the row written."""
    if source not in {"manual", "csat_low", "latency_outlier", "error", "unflag"}:
        raise ValueError(
            f"source must be one of "
            f"{{manual, csat_low, latency_outlier, error, unflag}}, got {source!r}"
        )

    root = root or _FLAGS_ROOT
    root.mkdir(parents=True, exist_ok=True)

    at = now_iso or _now_iso()
    row = {
        "trace_id": trace_id,
        "reason": reason if reason else None,
        "source": source,
        "at": at,
    }
    date = at[:10]
    target = root / f"{date}.jsonl"
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def iter_flag_rows(
    *,
    since_iso: Optional[str] = None,
    root: Optional[Path] = None,
):
    """Stream flag rows from JSONL partitions, date-filtered."""
    root = root or _FLAGS_ROOT
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


def is_flagged(
    trace_id: str,
    *,
    root: Optional[Path] = None,
) -> bool:
    """True iff the latest row for this trace is a flag (not an unflag)."""
    latest = _latest_row(trace_id, root=root)
    if latest is None:
        return False
    return latest.get("source") != "unflag"


def list_flag_rows_for_trace(
    trace_id: str,
    *,
    root: Optional[Path] = None,
) -> list[dict]:
    return [r for r in iter_flag_rows(root=root) if r.get("trace_id") == trace_id]


def flagged_trace_ids(
    *,
    root: Optional[Path] = None,
) -> set[str]:
    """Return the set of trace IDs whose latest row is a flag (not unflag)."""
    # Track latest row per trace_id by (at, source).
    latest_by_trace: dict[str, dict] = {}
    for row in iter_flag_rows(root=root):
        tid = row.get("trace_id")
        if not tid:
            continue
        prev = latest_by_trace.get(tid)
        if prev is None or (row.get("at") or "") > (prev.get("at") or ""):
            latest_by_trace[tid] = row
    return {
        tid for tid, row in latest_by_trace.items()
        if row.get("source") != "unflag"
    }


def _latest_row(trace_id: str, *, root: Optional[Path] = None) -> Optional[dict]:
    rows = list_flag_rows_for_trace(trace_id, root=root)
    if not rows:
        return None
    return max(rows, key=lambda r: r.get("at") or "")


def _now_iso() -> str:
    # Microsecond precision so back-to-back writes in tests + tight
    # operator workflows (flag → unflag) tie-break correctly.
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
