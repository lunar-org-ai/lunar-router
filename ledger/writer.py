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

_LEDGER_ROOT = _PROJECT_ROOT / "ledger"


def _agent_partition() -> str:
    """Lookup the active agent id at call time. Indirect import so tests
    that monkeypatch ``runtime.agent_context._active`` see the override."""
    from runtime.agent_context import get_active
    return get_active()


def _ledger_root() -> Path:
    """Effective ledger root for the current request.

    OSS mode (default) → legacy ``ledger/`` at project root.
    Infra mode (``OPENTRACY_MULTI_TENANT=1``) → ``tenants/<active>/ledger/``
    so each tenant's audit trail stays isolated. Falls back to
    ``_default`` when no tenant context is set (background tasks,
    boot-time writes)."""
    from runtime.tenants.feature import is_multi_tenant_enabled
    if not is_multi_tenant_enabled():
        return _LEDGER_ROOT
    from runtime.tenant_context import get_active as _get_tenant
    from runtime.tenants.registry import get_tenant_dir
    return get_tenant_dir(_get_tenant()) / "ledger"


def _entries_dir_for(agent_id: Optional[str] = None) -> Path:
    return _ledger_root() / (agent_id or _agent_partition()) / "entries"


def _lessons_dir_for(agent_id: Optional[str] = None) -> Path:
    return _ledger_root() / (agent_id or _agent_partition()) / "lessons"


def _decisions_dir_for(agent_id: Optional[str] = None) -> Path:
    return _ledger_root() / (agent_id or _agent_partition()) / "decisions"


# Back-compat aliases — preserved so existing tests that monkeypatch
# these names keep working. Callers should treat them as fallbacks; the
# resolver above is the live source of truth.
ENTRIES_DIR = _LEDGER_ROOT / "entries"
LESSONS_DIR = _LEDGER_ROOT / "lessons"
DECISIONS_DIR = _LEDGER_ROOT / "decisions"


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
    entries_dir: Optional[Path | str] = None,
    agent_id: Optional[str] = None,
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

    out_dir = Path(entries_dir) if entries_dir is not None else _entries_dir_for(agent_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"{today}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entry


def write_lesson(
    lesson: Lesson,
    lessons_dir: Optional[Path | str] = None,
    *,
    agent_id: Optional[str] = None,
) -> Path:
    """Persist a Lesson as a standalone JSON file the UI can read."""
    out_dir = Path(lessons_dir) if lessons_dir is not None else _lessons_dir_for(agent_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{lesson.id}.json"
    with path.open("w") as f:
        json.dump(asdict(lesson), f, indent=2, ensure_ascii=False)
    return path


def write_decision(
    kind: str,
    payload: dict[str, Any],
    *,
    decisions_dir: Optional[Path | str] = None,
    agent_id: Optional[str] = None,
) -> Path:
    """Persist a brain decision artifact (P15.3.9).

    Decisions are distinct from entries: they record what the brain
    CHOSE, not what the system DID. A "skipped" wake-up is a valid
    decision and worth persisting so Evolution can render
    "Claude Code declined retrain at T".

    Filename: ``<kind>_<utc_iso_compact>.json`` (e.g.
    ``router_wakeup_20260510T143022Z.json``). Compact ISO is a sort key.
    """
    out_dir = Path(decisions_dir) if decisions_dir is not None else _decisions_dir_for(agent_id)
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


def _all_agent_entries_dirs() -> list[Path]:
    """All ``<tenant>/<agent>/entries`` dirs under the current ledger root.

    Same aggregator pattern as :func:`_all_agent_lessons_dirs` — the
    harness introspection tools (``list_recent_promotions``,
    ``list_recent_rollbacks``) read across every agent in the tenant,
    so when no specific agent is requested we walk every dir."""
    root = _ledger_root()
    if not root.exists():
        return []
    return [d / "entries" for d in root.iterdir() if d.is_dir()]


def read_entries(
    entries_dir: Optional[Path | str] = None,
    *,
    agent_id: Optional[str] = None,
) -> list[LedgerEntry]:
    """Read all entries (chronological). For the harness, ledger/ui to query."""
    out: list[LedgerEntry] = []
    if entries_dir is not None:
        roots: list[Path] = [Path(entries_dir)]
    elif agent_id is not None:
        roots = [_entries_dir_for(agent_id)]
    else:
        roots = _all_agent_entries_dirs()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*.jsonl")):
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    out.append(LedgerEntry(**d))
    # Aggregator pulls multiple dirs — final sort by timestamp keeps
    # the chronological contract callers expect.
    out.sort(key=lambda e: e.timestamp or "")
    return out


def _all_agent_lessons_dirs() -> list[Path]:
    """Every ``<tenant>/<agent>/lessons`` dir under the current ledger root.

    Used to aggregate lessons across agents when the caller didn't ask
    for a specific agent — the UI lesson list, evolution timeline,
    router history, etc. don't carry agent context, so without this
    they'd see only whatever ``agent_context._active`` happened to be
    on the Cloud Run instance handling the request (racy across
    instances).
    """
    root = _ledger_root()
    if not root.exists():
        return []
    return [d / "lessons" for d in root.iterdir() if d.is_dir()]


def read_lessons(
    lessons_dir: Optional[Path | str] = None,
    *,
    agent_id: Optional[str] = None,
) -> list[Lesson]:
    out: list[Lesson] = []
    if lessons_dir is not None:
        roots: list[Path] = [Path(lessons_dir)]
    elif agent_id is not None:
        roots = [_lessons_dir_for(agent_id)]
    else:
        roots = _all_agent_lessons_dirs()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*.json")):
            with p.open() as f:
                d = json.load(f)
            out.append(Lesson(**d))
    return out


def read_lesson(
    lesson_id: str,
    lessons_dir: Optional[Path | str] = None,
    *,
    agent_id: Optional[str] = None,
) -> Optional[Lesson]:
    """Read a single Lesson by id. Returns None if not found.

    When neither ``lessons_dir`` nor ``agent_id`` is given, walks every
    agent dir under the current tenant — mirrors :func:`read_lessons`.
    """
    if lessons_dir is not None:
        roots: list[Path] = [Path(lessons_dir)]
    elif agent_id is not None:
        roots = [_lessons_dir_for(agent_id)]
    else:
        roots = _all_agent_lessons_dirs()
    for root in roots:
        p = root / f"{lesson_id}.json"
        if not p.exists():
            continue
        with p.open() as f:
            return Lesson(**json.load(f))
    return None


def update_lesson(
    lesson_id: str,
    *,
    lessons_dir: Optional[Path | str] = None,
    agent_id: Optional[str] = None,
    **fields: Any,
) -> Lesson:
    """Mutate a Lesson on disk. Provisional lessons are written by the loop
    and later mutated when a human approves or rejects them — that's the only
    legitimate path for a Lesson to change. Returns the updated Lesson.
    """
    lesson = read_lesson(lesson_id, lessons_dir, agent_id=agent_id)
    if lesson is None:
        raise FileNotFoundError(f"lesson {lesson_id!r} not found")
    for k, v in fields.items():
        if not hasattr(lesson, k):
            raise AttributeError(f"Lesson has no field {k!r}")
        setattr(lesson, k, v)
    write_lesson(lesson, lessons_dir, agent_id=agent_id)
    return lesson
