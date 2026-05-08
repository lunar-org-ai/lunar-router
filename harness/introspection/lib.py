"""Pure functions backing the introspection tools.

These do not depend on MCP or any LLM — they read from the existing data
substrate (ledger, lessons, distilled epochs). The MCP server in this package
just wraps them. Other consumers (HTTP endpoint, CLI, tests) can import the
same functions directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from ledger.writer import read_entries, read_lessons

# Anchor paths to the project root, not CWD — the MCP server is sometimes
# spawned with a different cwd (e.g. when Claude Code launches it as
# subprocess), and relative-path resolution would silently return empty.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EPOCHS_DIR = _PROJECT_ROOT / "traces" / "distilled" / "epochs"
SESSIONS_DIR = _PROJECT_ROOT / "traces" / "distilled" / "sessions"
POLICIES_PATH = _PROJECT_ROOT / "policies" / "auto_approve.yaml"
ENTRIES_DIR = _PROJECT_ROOT / "ledger" / "entries"
LESSONS_DIR = _PROJECT_ROOT / "ledger" / "lessons"


# ---------- helpers ----------


def _entry_to_dict(e: Any) -> dict[str, Any]:
    return {
        "entry_id": e.entry_id,
        "kind": e.kind,
        "timestamp": e.timestamp,
        "parent_entry_id": e.parent_entry_id,
        "candidate_id": e.candidate_id,
        "agent_version_before": e.agent_version_before,
        "agent_version_after": e.agent_version_after,
        "summary": e.summary,
        "payload": e.payload,
    }


# ---------- tools ----------


def list_recent_promotions(since_iso: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """List recent promotions (candidate → live agent)."""
    entries = [e for e in read_entries() if e.kind == "promote"]
    if since_iso:
        entries = [e for e in entries if e.timestamp >= since_iso]
    entries = entries[-limit:]
    return [
        {
            "entry_id": e.entry_id,
            "timestamp": e.timestamp,
            "candidate_id": e.candidate_id,
            "version_before": e.agent_version_before,
            "version_after": e.agent_version_after,
            "summary": e.summary,
            "mutations": e.payload.get("mutations", []),
            "delta_overall": (
                e.payload.get("delta", {}).get("overall_score")
                if isinstance(e.payload.get("delta"), dict)
                else None
            ),
            "prediction": e.payload.get("prediction"),
            "verification": e.payload.get("verification"),
        }
        for e in entries
    ]


def list_recent_rollbacks(since_iso: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """List recent rollbacks (live agent reverted to a prior version)."""
    entries = [e for e in read_entries() if e.kind == "rollback"]
    if since_iso:
        entries = [e for e in entries if e.timestamp >= since_iso]
    entries = entries[-limit:]
    return [
        {
            "entry_id": e.entry_id,
            "timestamp": e.timestamp,
            "version_before": e.agent_version_before,
            "version_after": e.agent_version_after,
            "summary": e.summary,
            "reason": e.payload.get("reason"),
        }
        for e in entries
    ]


def get_lesson(lesson_id: str) -> dict[str, Any]:
    """Fetch one approved lesson by id (the UI 'card')."""
    for lesson in read_lessons():
        if lesson.id == lesson_id:
            return asdict(lesson)
    return {"error": f"lesson not found: {lesson_id}"}


def get_day_epoch(date: str) -> dict[str, Any]:
    """Read the distilled day epoch (e.g. '2026-05-07'). Distills on-demand if missing."""
    path = EPOCHS_DIR / f"day_{date}.json"
    if not path.exists():
        try:
            from harness.observability.distillation import distill_day

            distill_day(date)
        except Exception as e:
            return {"error": f"could not distill day {date}: {e}"}
    if not path.exists():
        return {"error": f"no epoch for day {date}"}
    with path.open() as f:
        return json.load(f)


def list_predictions(verdict: str = "", limit: int = 50) -> list[dict[str, Any]]:
    """Find promotions that carried predictions, filterable by verification verdict.

    `verdict` ∈ {"verified", "partial", "wrong", "no_change"} or empty for all.
    Each row pairs the prediction with what actually happened.
    """
    out: list[dict[str, Any]] = []
    for e in read_entries():
        pred = e.payload.get("prediction") if isinstance(e.payload, dict) else None
        ver = e.payload.get("verification") if isinstance(e.payload, dict) else None
        if not pred:
            continue
        if verdict and (not ver or ver.get("verdict") != verdict):
            continue
        out.append(
            {
                "entry_id": e.entry_id,
                "timestamp": e.timestamp,
                "candidate_id": e.candidate_id,
                "version_after": e.agent_version_after,
                "prediction": pred,
                "verification": ver,
            }
        )
    return out[-limit:]


# ---------- discovery helper ----------


def list_available_epochs() -> dict[str, list[str]]:
    """List which days/versions have been distilled."""
    if not EPOCHS_DIR.exists():
        return {"days": [], "versions": []}
    days = sorted(p.stem.replace("day_", "") for p in EPOCHS_DIR.glob("day_*.json"))
    versions = sorted(p.stem.replace("version_", "") for p in EPOCHS_DIR.glob("version_*.json"))
    return {"days": days, "versions": versions}
