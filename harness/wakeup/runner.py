"""Wakeup runner — composes the prompt + invokes Claude Code + persists decision.

Called from ``harness/wakeup/scheduler.py:maybe_fire`` in a daemon thread
so it never blocks ``/run``. The runner reuses the existing
``harness/introspection/agent.py:introspect`` entry — same brain, same
tool surface (now including ``router_health_check`` +
``propose_router_retrain``).

Every wake-up writes a decision artifact, regardless of outcome:

  - ``proposed`` — Claude Code called ``propose_router_retrain``;
    artifact carries the resulting Lesson ID.
  - ``skipped`` — Claude Code declined; artifact carries the rationale.
  - ``blocked`` — policy off OR no brain transport OR
    ``not_enough_data``; artifact carries the reason.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("harness.wakeup.runner")


@dataclass
class WakeupOutcome:
    action: str               # "proposed" | "skipped" | "blocked" | "rolled_back"
    rationale: str
    lesson_id: Optional[str] = None
    target: Optional[str] = None  # "router" | "dataset" | "auto_rollback" | None
    reason: Optional[str] = None
    health_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    rollback_metadata: Optional[dict[str, Any]] = None  # P16.4 — populated when action="rolled_back"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "lesson_id": self.lesson_id,
            "target": self.target,
            "reason": self.reason,
            "health_snapshot": self.health_snapshot,
            "timestamp": self.timestamp,
            "rollback_metadata": self.rollback_metadata,
        }


def run_wakeup(
    *,
    threshold: int = 50,
    introspect_fn: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> WakeupOutcome:
    """Compose prompt with current health, call Claude Code, persist outcome."""
    from harness.wakeup.prompt import WAKEUP_PROMPT
    from ledger.writer import write_decision
    from router.feedback.health import compute_router_health

    health = compute_router_health(embedder=embedder)
    router_health = health.to_dict()
    dataset_health = _safe_dataset_health()
    health_snapshot = {
        "router": router_health,
        "datasets": dataset_health,
    }

    # P16.4 — Auto-rollback watcher. Runs BEFORE the brain transport
    # check because rollback is a safety mechanism that doesn't need
    # an LLM: a deterministic ledger + metric query is enough.
    rollback_outcome = _maybe_auto_rollback(health_snapshot)
    if rollback_outcome is not None:
        _persist(rollback_outcome)
        return rollback_outcome

    # Detect "no brain" up front so the operator gets a clear artifact
    # rather than a wake-up that silently fails.
    transport_check = _check_transport()
    if transport_check is not None:
        outcome = WakeupOutcome(
            action="blocked",
            rationale=f"no brain transport: {transport_check}",
            reason="no_brain_available",
            health_snapshot=health_snapshot,
            timestamp=_now_iso(),
        )
        _persist(outcome)
        return outcome

    prompt = WAKEUP_PROMPT.format(
        n_traces=threshold,
        router_health_json=json.dumps(router_health, indent=2),
        dataset_health_json=json.dumps(dataset_health, indent=2),
    )

    # Invoke Claude Code via the existing introspection entry — it carries
    # the full TOOLS list, so router_health_check + propose_router_retrain
    # are reachable.
    intro = introspect_fn or _default_introspect
    try:
        result = intro(prompt)
    except Exception as e:
        outcome = WakeupOutcome(
            action="blocked",
            rationale=f"introspect call failed: {type(e).__name__}: {e}",
            reason="introspect_error",
            health_snapshot=health_snapshot,
            timestamp=_now_iso(),
        )
        _persist(outcome)
        return outcome

    # Decide what action was taken: distinguish router_retrain vs dataset_curation.
    proposed = _extract_proposed_lesson(result)
    response_text = (
        getattr(result, "response", "") or ""
    ).strip()

    if proposed is not None:
        target, lesson_id = proposed
        outcome = WakeupOutcome(
            action="proposed",
            rationale=response_text,
            lesson_id=lesson_id,
            target=target,
            health_snapshot=health_snapshot,
            timestamp=_now_iso(),
        )
    else:
        outcome = WakeupOutcome(
            action="skipped",
            rationale=response_text or "(no rationale captured)",
            health_snapshot=health_snapshot,
            timestamp=_now_iso(),
        )

    _persist(outcome)
    return outcome


def _maybe_auto_rollback(health_snapshot: dict[str, Any]) -> Optional[WakeupOutcome]:
    """P16.4 — run the auto-rollback watcher; execute rollback if triggered.

    Returns a populated WakeupOutcome when a rollback fires (caller
    persists + returns it). Returns None when nothing to do — the
    wake-up continues with the normal proposer flow.

    Errors are logged + swallowed: a watcher crash must NEVER block
    the brain from running. Safety must be additive, not subtractive.
    """
    try:
        from harness.approver.policy import Policy
        from harness.watchers.auto_rollback import check_auto_rollback

        policy = Policy.from_yaml()
        decision = check_auto_rollback(policy=policy)
    except Exception as e:
        logger.warning("auto-rollback check failed: %s", e)
        return None

    if decision is None:
        return None

    # We have a rollback decision. Execute + notify.
    try:
        from harness.rollback.rollback import rollback_to
        from runtime.store.notifications import notify_channels

        actual_version = rollback_to(
            decision.target_version,
            reason=f"auto-rollback: {decision.reason}",
        )

        # Write a Lesson so Evolution surfaces this alongside promotions.
        lesson = _write_auto_rollback_lesson(decision, actual_version)

        # Notify each configured channel (persisted as audit; real
        # delivery is out of scope).
        if policy.auto_rollback.notify_channels:
            notify_channels(
                list(policy.auto_rollback.notify_channels),
                subject=f"Auto-rollback: {decision.suspect_version} → {actual_version}",
                body=decision.reason,
                kind="auto_rollback",
                lesson_id=lesson.id,
            )

        return WakeupOutcome(
            action="rolled_back",
            rationale=decision.reason,
            lesson_id=lesson.id,
            target="auto_rollback",
            health_snapshot=health_snapshot,
            timestamp=_now_iso(),
            rollback_metadata={
                "from_version": decision.suspect_version,
                "to_version": actual_version,
                "promote_entry_id": decision.promote_entry_id,
                "csat_before": decision.before.csat,
                "csat_after": decision.after.csat,
                "resolution_before": decision.before.resolution_rate,
                "resolution_after": decision.after.resolution_rate,
            },
        )
    except Exception as e:
        logger.exception("auto-rollback execution failed: %s", e)
        return WakeupOutcome(
            action="blocked",
            rationale=f"auto-rollback execution failed: {type(e).__name__}: {e}",
            reason="auto_rollback_failed",
            health_snapshot=health_snapshot,
            timestamp=_now_iso(),
        )


def _write_auto_rollback_lesson(decision, actual_version: str):
    """Surface the auto-rollback in Evolution with kind='rollback'."""
    from datetime import datetime, timezone
    import secrets

    from ledger.types import Lesson
    from ledger.writer import write_lesson

    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}"
    )
    lesson = Lesson(
        id=lesson_id,
        version=actual_version,
        kind="rollback",
        status="rolled_back",
        title=f"Auto-rollback {decision.suspect_version} → {actual_version}",
        summary=decision.reason,
        proposal_source="auto",
        delta={
            "csat_before": float(decision.before.csat or 0.0),
            "csat_after": float(decision.after.csat or 0.0),
            "resolution_before": float(decision.before.resolution_rate or 0.0),
            "resolution_after": float(decision.after.resolution_rate or 0.0),
        },
        mutations=[f"rollback:{decision.suspect_version} -> {actual_version}"],
        parent_version=decision.suspect_version,
        candidate_id="",
        promoted_at=promoted_at,
        ledger_entry_id="",  # rollback_to() wrote its own entry already
        voice=(
            f"I rolled back to {actual_version} on my own — production "
            f"metrics regressed after the last promotion."
        ),
    )
    write_lesson(lesson)
    return lesson


def _safe_dataset_health() -> dict[str, Any]:
    """Pull dataset health without crashing if the library isn't reachable."""
    try:
        from harness.introspection.lib import dataset_health_check
        return dataset_health_check()
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("dataset_health_check failed: %s", e)
        return {"datasets": [], "error": str(e)}


# ---------------------------------------------------------------------------


def _check_transport() -> Optional[str]:
    """Return None when a brain transport is reachable, else a reason string."""
    try:
        from harness.brain.transport import select_transport
    except ImportError:
        return "harness.brain.transport not importable"
    chosen = select_transport()
    if chosen == "none":
        return "no ANTHROPIC_API_KEY and no `claude` CLI on PATH"
    return None


def _default_introspect(prompt: str):
    """Default introspect callable — uses harness.introspection.agent."""
    from harness.introspection.agent import introspect

    return introspect(prompt)


_TARGET_BY_TOOL = {
    "propose_router_retrain": "router",
    "propose_dataset_curation": "dataset",
}


def _extract_proposed_lesson(result: Any) -> Optional[tuple[str, str]]:
    """Look at the introspect result for evidence that the model called
    one of the proposer tools and got back a lesson_id.

    Returns ``(target, lesson_id)`` when found, else ``None``.

    Two channels:
    1. ``result.tool_calls`` — preferred. Maps the tool name to a target.
    2. ``result.response`` text — fallback regex for the lesson_id; we
       can't distinguish target here, so default to ``"router"`` for
       backwards-compatibility (P15.3.9 only had the router tool).
    """
    tool_calls = getattr(result, "tool_calls", None) or []
    for call in tool_calls:
        tool = getattr(call, "tool", None)
        target = _TARGET_BY_TOOL.get(tool)
        if target is None:
            continue
        preview = getattr(call, "output_preview", "") or ""
        m = re.search(r'"lesson_id"\s*:\s*"(L-[\w\-]+)"', preview)
        if m:
            return target, m.group(1)

    text = getattr(result, "response", "") or ""
    m = re.search(r"L-\d{8}-\d{6}-[a-f0-9]+", text)
    if m:
        # Fallback — target unknown. Default to router for back-compat.
        return "router", m.group(0)
    return None


def _persist(outcome: WakeupOutcome) -> Path:
    from ledger.writer import write_decision

    return write_decision("router_wakeup", outcome.to_dict())


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _cli_main() -> None:
    """Manual operator-triggered wake-up: ``python -m harness.wakeup.runner``."""
    out = run_wakeup()
    print(json.dumps(out.to_dict(), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
