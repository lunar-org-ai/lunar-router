"""Per-agent self-improvement brain config (P3.2).

The "self-improvement engineer" — the harness brain that proposes
router retrains, dataset curations, prompt mutations, etc — used to
share one global transport + model with every install. Now each agent
can configure its own:

  agents/<id>/improvement.yaml
    enabled: true                # turn autonomous improvement on/off
    transport: auto              # auto | claude_code_cli | anthropic_api
    model: claude-sonnet-4-6     # which model to use when calling the brain
    cadence_minutes: 30          # wakeup interval; 0 disables periodic
    notes: ""                    # operator-facing description

This module is the canonical loader. ``harness/brain/transport.py`` and
``harness/wakeup`` consult ``resolve_for_active_agent()`` so the
selected brain matches the operator's preference at run time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("runtime.agents.improvement")


_FILENAME = "improvement.yaml"
_VALID_TRANSPORTS = ("auto", "claude_code_cli", "anthropic_api", "disabled")


@dataclass
class ImprovementConfig:
    """The shape persisted on disk + returned by the API."""

    enabled: bool = True
    transport: str = "auto"
    model: str = "claude-sonnet-4-6"
    cadence_minutes: int = 30
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "transport": self.transport,
            "model": self.model,
            "cadence_minutes": int(self.cadence_minutes),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImprovementConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            transport=_normalize_transport(data.get("transport", "auto")),
            model=str(data.get("model", "claude-sonnet-4-6") or "claude-sonnet-4-6"),
            cadence_minutes=max(0, int(data.get("cadence_minutes", 30) or 0)),
            notes=str(data.get("notes", "") or ""),
        )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _path(agent_id: str, *, root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root) / agent_id / _FILENAME
    # Honor the registry's root so tests that monkeypatch
    # ``runtime.agents.registry._DEFAULT_ROOT`` to tmp also redirect
    # improvement.yaml reads/writes without separate patching.
    try:
        from runtime.agents import registry as _reg
        return _reg._DEFAULT_ROOT / agent_id / _FILENAME
    except Exception:
        return Path("agents") / agent_id / _FILENAME


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def load(agent_id: str, *, root: Optional[Path] = None) -> ImprovementConfig:
    """Read ``agents/<id>/improvement.yaml``. Missing/unreadable → defaults.

    Uses PyYAML's safe loader so the file can carry comments + future
    keys the runtime doesn't know about yet without breaking.
    """
    path = _path(agent_id, root=root)
    if not path.is_file():
        return ImprovementConfig()
    try:
        import yaml
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning("improvement.yaml at %s is not a mapping — using defaults", path)
            return ImprovementConfig()
        return ImprovementConfig.from_dict(data)
    except Exception as e:
        logger.warning("failed to parse %s: %s — using defaults", path, e)
        return ImprovementConfig()


def save(
    agent_id: str,
    config: ImprovementConfig,
    *,
    root: Optional[Path] = None,
) -> Path:
    """Persist ``improvement.yaml``. Writes via a temp file + atomic rename
    so partial writes don't leave the operator with a broken config."""
    import yaml

    path = _path(agent_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    body = (
        "# Self-improvement brain config (P3.2). Edit here or via\n"
        "# PUT /v1/agents/<id>/improvement.\n"
    )
    body += yaml.safe_dump(
        config.to_dict(),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )

    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Resolution for the harness
# ---------------------------------------------------------------------------


def resolve_for_active_agent(
    *,
    default_transport: Optional[str] = None,
    default_model: Optional[str] = None,
    root: Optional[Path] = None,
) -> ImprovementConfig:
    """Read the active agent's improvement config; return defaults when
    no agent is active or the file is missing.

    ``default_transport`` / ``default_model`` are honored only when the
    config's value is ``"auto"`` / falsy — the operator's explicit
    setting always wins.
    """
    try:
        from runtime.agent_context import get_active
        agent_id = get_active()
    except Exception:
        return ImprovementConfig(
            transport=_normalize_transport(default_transport or "auto"),
            model=default_model or "claude-sonnet-4-6",
        )

    cfg = load(agent_id, root=root)
    if cfg.transport == "auto" and default_transport:
        cfg.transport = _normalize_transport(default_transport)
    if cfg.model in ("", None):
        cfg.model = default_model or "claude-sonnet-4-6"
    return cfg


def is_enabled(*, root: Optional[Path] = None) -> bool:
    """Convenience for the wakeup runner: should we run at all?"""
    cfg = resolve_for_active_agent(root=root)
    return cfg.enabled and cfg.transport != "disabled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_transport(raw: Any) -> str:
    v = str(raw or "auto").strip().lower()
    return v if v in _VALID_TRANSPORTS else "auto"
