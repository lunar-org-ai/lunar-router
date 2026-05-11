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
    action: str  # "proposed" | "skipped" | "blocked"
    rationale: str
    lesson_id: Optional[str] = None
    reason: Optional[str] = None
    health_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "lesson_id": self.lesson_id,
            "reason": self.reason,
            "health_snapshot": self.health_snapshot,
            "timestamp": self.timestamp,
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
    health_dict = health.to_dict()

    # Detect "no brain" up front so the operator gets a clear artifact
    # rather than a wake-up that silently fails.
    transport_check = _check_transport()
    if transport_check is not None:
        outcome = WakeupOutcome(
            action="blocked",
            rationale=f"no brain transport: {transport_check}",
            reason="no_brain_available",
            health_snapshot=health_dict,
            timestamp=_now_iso(),
        )
        _persist(outcome)
        return outcome

    prompt = WAKEUP_PROMPT.format(
        n_traces=threshold,
        health_json=json.dumps(health_dict, indent=2),
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
            health_snapshot=health_dict,
            timestamp=_now_iso(),
        )
        _persist(outcome)
        return outcome

    # Decide what action was taken: any tool call to propose_router_retrain
    # in the captured tool_calls counts as "proposed".
    proposed_lesson_id = _extract_proposed_lesson(result)
    response_text = (
        getattr(result, "response", "") or ""
    ).strip()

    if proposed_lesson_id:
        outcome = WakeupOutcome(
            action="proposed",
            rationale=response_text,
            lesson_id=proposed_lesson_id,
            health_snapshot=health_dict,
            timestamp=_now_iso(),
        )
    else:
        outcome = WakeupOutcome(
            action="skipped",
            rationale=response_text or "(no rationale captured)",
            health_snapshot=health_dict,
            timestamp=_now_iso(),
        )

    _persist(outcome)
    return outcome


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


def _extract_proposed_lesson(result: Any) -> Optional[str]:
    """Look at the introspect result for evidence that the model called
    ``propose_router_retrain`` and got back a lesson_id.

    Two channels:
    1. ``result.tool_calls`` — when the introspect path captures them.
       Look for ``propose_router_retrain`` outputs that contain a
       ``"lesson_id"`` field.
    2. ``result.response`` text — fallback regex for ``L-YYYYMMDD-...``
       which is the lesson_id format used elsewhere in the codebase.
    """
    tool_calls = getattr(result, "tool_calls", None) or []
    for call in tool_calls:
        if getattr(call, "tool", None) == "propose_router_retrain":
            preview = getattr(call, "output_preview", "") or ""
            m = re.search(r'"lesson_id"\s*:\s*"(L-[\w\-]+)"', preview)
            if m:
                return m.group(1)

    text = getattr(result, "response", "") or ""
    m = re.search(r"L-\d{8}-\d{6}-[a-f0-9]+", text)
    if m:
        return m.group(0)
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
