"""Day-0 onboarding state — `agent/onboarding.json`.

When the operator first opens the UI, the Onboarding screen takes over
until they pick a template, name + describe the agent, choose a model,
and select channels. ``record_complete()`` materializes that into the
actual agent surface:

  - ``agent/prompts/system.md`` — rewritten with the rendered prompt
    (``{{company}}`` filled in).
  - ``agent/agent.yaml`` description — set to the agent name + template.
  - ``agent/onboarding.json`` — full config saved for replay/audit.
  - Manual Lesson ``kind=agent_created`` — so the agent's birth shows
    up in Evolution alongside future autonomous changes.

Tools + channels are persisted but NOT wired to actual implementations
yet — that's a separate phase. The onboarding config carries enough to
build them later.

Idempotency: posting the same config twice is allowed (UI may retry on
flaky networks). The Lesson is only written on the first completion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("runtime.store.onboarding")

_DEFAULT_PATH = Path("agent") / "onboarding.json"
_DEFAULT_PROMPT_PATH = Path("agent") / "prompts" / "system.md"


@dataclass
class OnboardingConfig:
    template: Optional[str] = None
    name: str = ""
    company: str = ""
    prompt: str = ""
    model: str = "claude-sonnet-4-6"
    tools: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    completed: bool = False
    completed_at: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "name": self.name,
            "company": self.company,
            "prompt": self.prompt,
            "model": self.model,
            "tools": list(self.tools),
            "channels": list(self.channels),
            "completed": self.completed,
            "completed_at": self.completed_at,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OnboardingConfig":
        return cls(
            template=data.get("template"),
            name=str(data.get("name", "")),
            company=str(data.get("company", "")),
            prompt=str(data.get("prompt", "")),
            model=str(data.get("model", "claude-sonnet-4-6")),
            tools=list(data.get("tools", []) or []),
            channels=list(data.get("channels", []) or []),
            completed=bool(data.get("completed", False)),
            completed_at=data.get("completed_at"),
            skipped=bool(data.get("skipped", False)),
        )


def load_state(path: Optional[Path] = None) -> OnboardingConfig:
    """Read agent/onboarding.json, or return an empty config when missing."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.is_file():
        return OnboardingConfig()
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return OnboardingConfig.from_dict(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("onboarding state at %s unreadable (%s) — treating as empty", p, e)
        return OnboardingConfig()


def save_state(
    config: OnboardingConfig,
    path: Optional[Path] = None,
) -> Path:
    """Write the JSON. Used by record_complete and the skip path."""
    p = Path(path) if path else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return p


def render_prompt(template: str, company: str) -> str:
    """Substitute ``{{company}}`` with the company name (or a sane default)."""
    fill = company.strip() or "your company"
    return template.replace("{{company}}", fill)


def record_complete(
    payload: dict[str, Any],
    *,
    path: Optional[Path] = None,
    prompt_path: Optional[Path] = None,
    write_lesson_hook=None,
) -> OnboardingConfig:
    """Persist the config, write the system prompt, optionally write a Lesson.

    Idempotent: re-posting writes the JSON again but skips the Lesson if
    the previous state was already ``completed``. The Lesson hook is
    factored so tests can inject without touching the ledger writer.
    """
    p = Path(path) if path else _DEFAULT_PATH
    prev = load_state(p)

    config = OnboardingConfig.from_dict(payload)
    config.completed = True
    config.completed_at = _now_iso()
    config.skipped = False

    if config.prompt.strip():
        rendered = render_prompt(config.prompt, config.company)
        _write_prompt(rendered, prompt_path)

    save_state(config, p)

    if not prev.completed:
        try:
            (write_lesson_hook or _default_lesson_hook)(config)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("failed to write agent_created Lesson: %s", e)

    return config


def record_skip(path: Optional[Path] = None) -> OnboardingConfig:
    """Mark onboarding as dismissed by the operator without launching an
    agent. Useful for power users who configured the agent by hand."""
    config = load_state(path)
    config.completed = True
    config.skipped = True
    config.completed_at = _now_iso()
    save_state(config, path)
    return config


def _write_prompt(text: str, path: Optional[Path] = None) -> Path:
    p = Path(path) if path else _DEFAULT_PROMPT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    body = text.strip() + "\n\nThis prompt is part of the trainable surface. The harness may mutate it.\n"
    p.write_text(body, encoding="utf-8")
    return p


def _default_lesson_hook(config: OnboardingConfig) -> None:
    """Record the agent's birth as a manual Lesson in Evolution."""
    import secrets
    from ledger.types import Lesson
    from ledger.writer import write_lesson

    lid = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}"
    )
    title = f"Agent created — {config.name or 'unnamed'}"
    summary = (
        f"Operator finished day-0 onboarding. Template: "
        f"{config.template or 'blank'}. Model: {config.model}. "
        f"Tools: {len(config.tools)}. Channels: {len(config.channels)}."
    )
    write_lesson(
        Lesson(
            id=lid,
            version=None,
            kind="agent_created",
            status="approved",
            title=title,
            summary=summary,
            voice=(
                "I was just born — the operator picked my template, gave me "
                "a brain, and chose where I live. Now I wait for my first "
                "conversation."
            ),
            delta={},
            mutations=["agent/prompts/system.md", "agent/onboarding.json"],
            parent_version=None,
            candidate_id=None,
            promoted_at=config.completed_at,
            ledger_entry_id="",
            proposal_source="human",
        )
    )


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
