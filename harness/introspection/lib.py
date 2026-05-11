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

    # 5. Critic + approver + executor — full AHE pipeline.
    return _run_router_pipeline(proposal, policy, mode)


# ---------- P15.3 follow-ups: end-to-end promotion ----------


def _run_router_pipeline(proposal, policy, mode: str) -> dict[str, Any]:
    """Run critic → approver → executor for a router_config Proposal.

    Returns a typed dict carrying the final action: ``"promoted"`` (auto),
    ``"queued"`` (status=awaiting_review for human approval), ``"rejected"``
    (critic blocked or policy=off), or ``"blocked"`` (cache/dataset missing
    so the critic can't run).
    """
    from harness.approver.policy import ApprovalDecision, decide
    from harness.critics.router_critic import RouterCritic
    from harness.types import CriticContext, LoopOutcome
    from harness.executor.promote import promote_router_config

    # 0. Defensive — refuse if policy is off. The MCP entry point already
    # filters this, but the function-level guard means a direct caller
    # can't accidentally bypass.
    if mode == "off":
        return {
            "action": "blocked",
            "reason": "policy: mode is 'off' for router_config",
            "lesson_id": None,
        }

    # 1. Build cache + dataset for the critic. Either missing → block, NOT crash.
    crit_inputs = _critic_inputs_for_proposal(proposal)
    if crit_inputs.get("blocked"):
        return {
            "action": "blocked",
            "reason": crit_inputs["reason"],
            "lesson_id": None,
        }

    # 2. Run critic.
    critic = RouterCritic(params=crit_inputs)
    ctx = CriticContext(proposal=proposal, candidate_result=None)
    try:
        verdict = critic.verdict(ctx)
    except Exception as e:
        return {
            "action": "blocked",
            "reason": f"critic_error: {type(e).__name__}: {e}",
            "lesson_id": None,
        }

    # 3. Critic blocks → write a rejection Lesson regardless of mode.
    if not verdict.approved:
        lesson = _write_rejected_router_lesson(proposal, verdict)
        return {
            "action": "rejected",
            "reason": verdict.reason,
            "lesson_id": lesson.id,
        }

    # 4. Critic passed — branch on policy mode directly. We don't call the
    # generic decide() because it expects a candidate_result with an
    # "overall_score" delta (set by the agent eval suite). router_config
    # candidates carry their own scoring inside the critic verdict, so we
    # map mode → action directly. policy=off was already filtered upstream.
    outcome = LoopOutcome(
        proposal=proposal,
        candidate_id=None,
        verdicts=[verdict],
        candidate_result=None,
        final="approved",
    )

    if mode == "auto":
        try:
            new_version, lesson_id = promote_router_config(outcome)
        except Exception as e:
            return {
                "action": "blocked",
                "reason": f"executor_error: {type(e).__name__}: {e}",
                "lesson_id": None,
            }
        return {
            "action": "promoted",
            "lesson_id": lesson_id,
            "version": new_version,
            "reason": verdict.reason,
        }

    # mode == "review" (or anything else not handled above).
    lesson = _write_queued_router_lesson(proposal, verdict)
    return {
        "action": "queued",
        "lesson_id": lesson.id,
        "version": (
            int(proposal.mutations[0].value.get("version"))
            if proposal.mutations
            else None
        ),
        "reason": (
            "policy=review for router_config — pending human approval. "
            "Approve via /v1/lessons/{lesson_id}/approve or the Review screen."
        ),
    }


def _critic_inputs_for_proposal(proposal) -> dict[str, Any]:
    """Resolve the cache + dataset the critic needs.

    Returns ``{cache, dataset, embedder, centroids, eval_lambda_steps}``
    when ready, or ``{blocked: True, reason: str}`` when something's
    missing. Operators populate the cache via
    ``tools/populate_response_cache.py``.
    """
    from runtime.embedder_pool import get_pool
    from router.evaluation.cache import DEFAULT_CACHE_PATH, ResponseCache
    from router.data.dataset import PromptDataset

    if not DEFAULT_CACHE_PATH.exists():
        return {
            "blocked": True,
            "reason": (
                "cache_missing: evals/_response_cache/cache.jsonl is empty. "
                "Run `python -m tools.populate_response_cache` to seed it."
            ),
        }
    cache = ResponseCache(path=DEFAULT_CACHE_PATH)
    if len(cache) == 0:
        return {
            "blocked": True,
            "reason": "cache_missing: cache file exists but has no entries",
        }

    samples = _load_goldens_as_samples()
    if not samples:
        return {
            "blocked": True,
            "reason": "no_goldens: evals/golden/ is empty; cannot score the candidate",
        }
    dataset = PromptDataset(samples, name="router_critic_default")

    centroids = None
    if proposal.mutations:
        payload = proposal.mutations[0].value
        if isinstance(payload, dict):
            centroids = payload.get("centroids")

    return {
        "cache": cache,
        "dataset": dataset,
        "embedder": get_pool().get(),
        "centroids": centroids,
        "eval_lambda_steps": 5,
    }


def _load_goldens_as_samples() -> list[Any]:
    """Read evals/golden/*.yaml and convert to PromptSample list."""
    from pathlib import Path

    from evals.loader import load_golden
    from router.data.dataset import PromptSample

    golden_dir = Path("evals") / "golden"
    if not golden_dir.exists():
        return []

    samples: list[Any] = []
    for path in sorted(golden_dir.glob("*.yaml")):
        gid = path.stem
        try:
            g = load_golden(gid)
        except Exception:
            continue
        samples.append(
            PromptSample(
                prompt=g.input.request,
                ground_truth=g.expected.exact or "",
                category=g.expected.category,
            )
        )
    return samples


def _write_rejected_router_lesson(proposal, verdict) -> Any:
    from datetime import datetime, timezone
    import secrets

    from ledger.types import Lesson
    from ledger.writer import write_entry, write_lesson

    candidate_payload = proposal.mutations[0].value if proposal.mutations else {}
    candidate_version = (
        int(candidate_payload.get("version", 0))
        if isinstance(candidate_payload, dict)
        else 0
    )
    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    summary = f"router_config v{candidate_version} rejected by critic"

    entry = write_entry(
        kind="rejected",
        summary=summary,
        payload={
            "kind": "router_config",
            "source": proposal.source,
            "verdict_reason": verdict.reason,
            "candidate_version": candidate_version,
        },
    )

    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}"
    )
    lesson = Lesson(
        id=lesson_id,
        version="",
        kind="router_config",
        status="human_rejected",
        title=f"router_config v{candidate_version} rejected",
        summary=verdict.reason or "Critic blocked the candidate.",
        proposal_source=proposal.source,
        delta={},
        mutations=[m.describe() for m in proposal.mutations],
        parent_version="",
        candidate_id="",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice="I tried to refit my routing but the critic said it didn't help.",
    )
    write_lesson(lesson)
    return lesson


def _write_queued_router_lesson(proposal, verdict) -> Any:
    from datetime import datetime, timezone
    import secrets

    from ledger.types import Lesson
    from ledger.writer import write_entry, write_lesson

    candidate_payload = proposal.mutations[0].value if proposal.mutations else {}
    candidate_version = (
        int(candidate_payload.get("version", 0))
        if isinstance(candidate_payload, dict)
        else 0
    )
    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    summary = f"router_config v{candidate_version} awaiting human review"

    entry = write_entry(
        kind="queued_review",
        summary=summary,
        payload={
            "kind": "router_config",
            "source": proposal.source,
            "verdict_reason": verdict.reason,
            "candidate_version": candidate_version,
            "candidate_payload": candidate_payload,
        },
    )

    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}"
    )
    lesson = Lesson(
        id=lesson_id,
        version="",
        kind="router_config",
        status="awaiting_review",
        title=f"router_config v{candidate_version} awaiting review",
        summary=verdict.reason or "Critic passed — awaiting human approval.",
        proposal_source=proposal.source,
        delta={},
        mutations=[m.describe() for m in proposal.mutations],
        parent_version="",
        candidate_id="",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice="I refit my routing and it looks good — checking with you before promoting.",
    )
    write_lesson(lesson)
    return lesson
