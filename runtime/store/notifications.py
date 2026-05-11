"""Notification side-table for auto-rollback events (P16.4).

Layout::

    ledger/notifications/<YYYY-MM-DD>.jsonl

Each line records one notification dispatch:
``{"channel": str, "subject": str, "body": str, "kind": str,
   "lesson_id": str|None, "at": iso}``

Real delivery (email, Slack, webhook) is out of scope — we persist
the payload so operators can audit what would have been sent. When
real channels land later, a separate worker reads these rows and
delivers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_NOTIFICATIONS_ROOT = Path("ledger") / "notifications"


def write_notification(
    *,
    channel: str,
    subject: str,
    body: str,
    kind: str = "auto_rollback",
    lesson_id: Optional[str] = None,
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> dict:
    """Append a notification row. Returns the row written."""
    root = root or _NOTIFICATIONS_ROOT
    root.mkdir(parents=True, exist_ok=True)

    at = now_iso or _now_iso()
    row = {
        "channel": channel,
        "subject": subject,
        "body": body,
        "kind": kind,
        "lesson_id": lesson_id,
        "at": at,
    }
    date = at[:10]
    target = root / f"{date}.jsonl"
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def notify_channels(
    channels: list[str],
    *,
    subject: str,
    body: str,
    kind: str = "auto_rollback",
    lesson_id: Optional[str] = None,
    root: Optional[Path] = None,
) -> list[dict]:
    """Write one notification row per channel. Returns the rows written."""
    return [
        write_notification(
            channel=c, subject=subject, body=body,
            kind=kind, lesson_id=lesson_id, root=root,
        )
        for c in channels
    ]


def iter_notifications(
    *,
    since_iso: Optional[str] = None,
    root: Optional[Path] = None,
):
    """Stream notification rows from JSONL partitions, date-filtered."""
    root = root or _NOTIFICATIONS_ROOT
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


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
