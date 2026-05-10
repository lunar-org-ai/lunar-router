"""Append-only ledger writer.

JSONL per day in ledger/entries/<YYYY-MM-DD>.jsonl. Lessons are written as
individual files in ledger/lessons/<lesson_id>.json so the UI can read them
directly without scanning the JSONL.

Entries are immutable. To change anything, write a *new* entry referencing
the old via parent_entry_id.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ledger.types import ENTRY_KINDS, LedgerEntry, Lesson

# Anchor to project root so MCP-server-launched subprocesses (which can have
# arbitrary cwd) still find the ledger reliably.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENTRIES_DIR = _PROJECT_ROOT / "ledger" / "entries"
LESSONS_DIR = _PROJECT_ROOT / "ledger" / "lessons"
DECISIONS_DIR = _PROJECT_ROOT / "ledger" / "decisions"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _new_entry_id() -> str:
    """Sortable id: <ts>-<rand>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"led_{ts}_{secrets.token_hex(3)}"


def write_entry(
    kind: str,
    *,
    summary: Optional[str] = None,
    parent_entry_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    agent_version_before: Optional[str] = None,
    agent_version_after: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    entries_dir: Path | str = ENTRIES_DIR,
) -> LedgerEntry:
    """Append a LedgerEntry. Returns the entry (with assigned id + timestamp)."""
    if kind not in ENTRY_KINDS:
        raise ValueError(f"unknown ledger entry kind {kind!r}; must be one of {ENTRY_KINDS}")

    entry = LedgerEntry(
        entry_id=_new_entry_id(),
        kind=kind,
        timestamp=_now_iso(),
        parent_entry_id=parent_entry_id,
        candidate_id=candidate_id,
        agent_version_before=agent_version_before,
        agent_version_after=agent_version_after,
        summary=summary,
        payload=payload or {},
    )

    out_dir = Path(entries_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"{today}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entry


def write_lesson(lesson: Lesson, lessons_dir: Path | str = LESSONS_DIR) -> Path:
    """Persist a Lesson as a standalone JSON file the UI can read."""
    out_dir = Path(lessons_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{lesson.id}.json"
    with path.open("w") as f:
        json.dump(asdict(lesson), f, indent=2, ensure_ascii=False)
    return path


def write_decision(
    kind: str,
    payload: dict[str, Any],
    *,
    decisions_dir: Path | str = DECISIONS_DIR,
) -> Path:
    """Persist a brain decision artifact (P15.3.9).

    Decisions are distinct from entries: they record what the brain
    CHOSE, not what the system DID. A "skipped" wake-up is a valid
    decision and worth persisting so Evolution can render
    "Claude Code declined retrain at T".

    Filename: ``<kind>_<utc_iso_compact>.json`` (e.g.
    ``router_wakeup_20260510T143022Z.json``). Compact ISO is a sort key.
    """
    out_dir = Path(decisions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{kind}_{ts}.json"
    full = {
        "kind": kind,
        "timestamp": ts,
        "payload": payload,
    }
    with path.open("w") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    return path


def read_entries(entries_dir: Path | str = ENTRIES_DIR) -> list[LedgerEntry]:
    """Read all entries (chronological). For the harness, ledger/ui to query."""
    out: list[LedgerEntry] = []
    root = Path(entries_dir)
    if not root.exists():
        return out
    for p in sorted(root.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(LedgerEntry(**d))
    return out


def read_lessons(lessons_dir: Path | str = LESSONS_DIR) -> list[Lesson]:
    out: list[Lesson] = []
    root = Path(lessons_dir)
    if not root.exists():
        return out
    for p in sorted(root.glob("*.json")):
        with p.open() as f:
            d = json.load(f)
        out.append(Lesson(**d))
    return out


def read_lesson(lesson_id: str, lessons_dir: Path | str = LESSONS_DIR) -> Optional[Lesson]:
    """Read a single Lesson by id. Returns None if not found."""
    p = Path(lessons_dir) / f"{lesson_id}.json"
    if not p.exists():
        return None
    with p.open() as f:
        return Lesson(**json.load(f))


def update_lesson(
    lesson_id: str, *, lessons_dir: Path | str = LESSONS_DIR, **fields: Any
) -> Lesson:
    """Mutate a Lesson on disk. Provisional lessons are written by the loop
    and later mutated when a human approves or rejects them — that's the only
    legitimate path for a Lesson to change. Returns the updated Lesson.
    """
    lesson = read_lesson(lesson_id, lessons_dir)
    if lesson is None:
        raise FileNotFoundError(f"lesson {lesson_id!r} not found")
    for k, v in fields.items():
        if not hasattr(lesson, k):
            raise AttributeError(f"Lesson has no field {k!r}")
        setattr(lesson, k, v)
    write_lesson(lesson, lessons_dir)
    return lesson
