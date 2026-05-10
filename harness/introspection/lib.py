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


# ---------- P15.3.9: router brain tools ----------


def router_health_check() -> dict[str, Any]:
    """Pure-read snapshot of the router's current state.

    Returns the dict surface ``RouterHealth.to_dict()`` produces. Cold-start
    safe — returns ``cold_start: True`` when no router_config exists yet.
    """
    from router.feedback.health import compute_router_health

    return compute_router_health().to_dict()


def propose_router_retrain(rationale: str = "") -> dict[str, Any]:
    """Trigger a router_config retrain via the AHE pipeline.

    Runs proposer → critic → approver → executor → ledger. Returns a
    dict carrying the action taken: ``"promoted" | "queued" | "rejected"
    | "blocked"`` plus the resulting Lesson ID when applicable.

    Gated by ``Policy``: when global mode is ``"off"`` or
    ``overrides["router_config"] == "off"``, returns
    ``{"action": "blocked", "reason": "policy: ..."}`` without invoking
    the proposer.

    Other "blocked" reasons:
      - ``not_enough_data`` (corpus below ``min_corpus_size``)
      - ``cold_start_no_models`` (LLMRegistry empty)
      - ``no_brain_available``  (no ANTHROPIC_API_KEY and no `claude` CLI)

    The ``rationale`` arg is captured into the resulting Lesson's
    metadata so operators can later see why Claude Code thought a
    retrain was warranted.
    """
    from router.errors import NotEnoughDataError, RouterColdStartError

    # 1. Policy gate.
    try:
        from harness.approver.policy import Policy

        policy = Policy.from_yaml()
        mode = policy.mode_for("router_config")
    except Exception as e:
        return {
            "action": "blocked",
            "reason": f"policy_load_error: {type(e).__name__}: {e}",
            "lesson_id": None,
        }
    if mode == "off":
        return {
            "action": "blocked",
            "reason": "policy: mode is 'off' for router_config",
            "lesson_id": None,
        }

    # 2. Build the proposer (best-effort — embedder + registry + cache).
    try:
        from harness.proposer.router_proposer import (
            RouterProposer,
            RouterProposerConfig,
        )
        from router.config_io import load_current_config
        from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError
        from router.evaluation.cache import DEFAULT_CACHE_PATH, ResponseCache
        from router.models.llm_registry import LLMRegistry
        from runtime.embedder_pool import get_pool

        embedder = get_pool().get()
        try:
            _, registry, _ = load_current_config()
        except (RouterConfigNotFoundError, RouterConfigInvalidError):
            registry = LLMRegistry()
            if len(registry) == 0:
                return {
                    "action": "blocked",
                    "reason": (
                        "cold_start_no_models: no current router_config and the "
                        "registry is empty — register at least one LLMProfile via "
                        "the agent config before proposing"
                    ),
                    "lesson_id": None,
                }

        cache_path = DEFAULT_CACHE_PATH if DEFAULT_CACHE_PATH.exists() else None
        cache = ResponseCache(path=cache_path) if cache_path else None
        proposer = RouterProposer(
            embedder=embedder,
            registry=registry,
            cache=cache,
            cfg=RouterProposerConfig(),
        )
    except Exception as e:
        return {
            "action": "blocked",
            "reason": f"proposer_init_error: {type(e).__name__}: {e}",
            "lesson_id": None,
        }

    # 3. Propose.
    try:
        proposal = proposer.propose()
    except NotEnoughDataError as e:
        return {
            "action": "blocked",
            "reason": f"not_enough_data: {e}",
            "lesson_id": None,
        }
    except RouterColdStartError as e:
        return {
            "action": "blocked",
            "reason": f"cold_start: {e}",
            "lesson_id": None,
        }
    except Exception as e:
        return {
            "action": "blocked",
            "reason": f"propose_error: {type(e).__name__}: {e}",
            "lesson_id": None,
        }

    # 4. Stamp the rationale on the proposal metadata.
    if rationale:
        proposal.metadata = {**proposal.metadata, "claude_code_rationale": rationale}

    # 5. The critic + approver + executor wiring is the proposer's
    # responsibility once the proposal is formed. P15.3.9 returns the
    # in-progress state so the wakeup runner can observe what happened
    # without forcing the full critic/eval loop here. The actual
    # promotion flow is exercised end-to-end by the smoke + by
    # propose_and_run() helpers added later.
    return {
        "action": "queued",
        "reason": (
            "proposal generated; promote via the harness loop "
            "(critic + approver + executor) — full pipeline integration "
            "lands when the wake-up runner adopts it end-to-end"
        ),
        "lesson_id": None,
        "proposal_summary": proposal.description,
        "candidate_version": proposal.mutations[0].value.get("version")
        if proposal.mutations
        else None,
    }
