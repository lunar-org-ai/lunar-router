"""Per-agent channel registry (P3.3).

Each agent's outbound surface (API token, Slack workspace, WhatsApp
number, web widget) lives under ``agents/<id>/integrations/`` as a
mix of JSON config files + secret material. This module is the
common store: a single ``list_status`` / ``get`` / ``put`` / ``remove``
surface the UI talks to, plus a per-channel ``meta`` blob each
channel handler defines.

Layout::

    agents/<id>/integrations/
      api.json            # API channel — {token, created_at, last_used_at}
      slack.json          # Slack — {team_id, team_name, bot_token, ...}
      whatsapp.json       # WhatsApp/Twilio — {account_sid, auth_token, from}

Files are gitignored (the ``/agents/`` ignore from P2.0 already
covers them). Permissions are 0600 on POSIX where supported.

The channel-specific business logic (OAuth flow, webhook validation,
outbound API calls) lives in dedicated handler modules. This module
is just the metadata store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("runtime.agents.channels")


KNOWN_CHANNELS = ("api", "slack", "whatsapp", "web")


def _integrations_dir(agent_id: str, *, root: Optional[Path] = None) -> Path:
    """Where the agent's channel configs live. Honors the registry's
    root so tests that scope agents/ to tmp also scope integrations/."""
    if root is None:
        try:
            from runtime.agents import registry as _reg
            root = _reg.agents_root()
        except Exception:
            root = Path("agents")
    return root / agent_id / "integrations"


def _path(agent_id: str, channel: str, *, root: Optional[Path] = None) -> Path:
    return _integrations_dir(agent_id, root=root) / f"{channel}.json"


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def load(
    agent_id: str,
    channel: str,
    *,
    root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Read the channel's config dict. Returns ``None`` when not connected."""
    path = _path(agent_id, channel, root=root)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("channel %s for %s unreadable (%s)", channel, agent_id, e)
        return None


def save(
    agent_id: str,
    channel: str,
    config: dict[str, Any],
    *,
    root: Optional[Path] = None,
) -> Path:
    """Persist the channel's config. Creates the file with mode 0600
    when supported. Atomic via .tmp rename."""
    path = _path(agent_id, channel, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


def remove(
    agent_id: str,
    channel: str,
    *,
    root: Optional[Path] = None,
) -> bool:
    """Disconnect the channel by deleting its config file. Returns True
    when something was removed, False when no config existed."""
    path = _path(agent_id, channel, root=root)
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as e:
        logger.warning("failed to remove %s for %s: %s", channel, agent_id, e)
        return False


# ---------------------------------------------------------------------------
# UI status surface
# ---------------------------------------------------------------------------


def status(agent_id: str, *, root: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """Per-channel snapshot the AgentSheet renders. Each entry has:
       - ``connected`` (bool) — config file exists
       - ``meta`` (dict) — channel-specific summary (no secrets)
    Callers (server) overlay a sanitized projection per channel before
    sending to the UI.
    """
    out: dict[str, dict[str, Any]] = {}
    for ch in KNOWN_CHANNELS:
        cfg = load(agent_id, ch, root=root)
        out[ch] = {
            "connected": cfg is not None,
            "meta": _summarize(ch, cfg) if cfg else {},
        }
    return out


def _summarize(channel: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Channel-specific projection for the UI. No raw secrets ever leave
    this function. New channels add their own branch here."""
    if channel == "api":
        token = cfg.get("token", "")
        return {
            "created_at": cfg.get("created_at"),
            "last_used_at": cfg.get("last_used_at"),
            "mask": _mask_token(token),
        }
    if channel == "slack":
        return {
            "team_name": cfg.get("team_name"),
            "team_id": cfg.get("team_id"),
            "installed_at": cfg.get("installed_at"),
        }
    if channel == "whatsapp":
        return {
            "from_number": cfg.get("from_number"),
            "account_sid_mask": _mask_token(cfg.get("account_sid", "")),
            "provider": cfg.get("provider", "twilio"),
        }
    if channel == "web":
        return {
            "widget_id": cfg.get("widget_id"),
            "installed_at": cfg.get("installed_at"),
            "signing_secret_mask": _mask_token(cfg.get("signing_secret", "")),
            "allowed_domains": list(cfg.get("allowed_domains", [])),
            "settings": cfg.get("settings", {}),
        }
    return {}


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 10:
        return token[:2] + "…" + token[-2:]
    return f"{token[:6]}…{token[-4:]}"


# ---------------------------------------------------------------------------
# Resolve agent_id from inbound channel hints
# ---------------------------------------------------------------------------


def find_agent_by_channel(
    channel: str,
    matcher: dict[str, str],
    *,
    root: Optional[Path] = None,
) -> Optional[str]:
    """Reverse lookup: given a channel-specific key (Slack team_id,
    Twilio From number, API token), return the owning agent_id.

    Used by inbound webhooks to route a message to the right agent.
    Naive linear scan — fine while the agent count is small; can move
    to an indexed lookup later if needed.
    """
    try:
        from runtime.agents import registry as _reg
        rroot = root or _reg.agents_root()
    except Exception:
        rroot = root or Path("agents")
    if not rroot.is_dir():
        return None
    for entry in rroot.iterdir():
        if not entry.is_dir() or entry.name.startswith(("_deleted", ".")):
            continue
        if entry.name == "_deleted":
            continue
        cfg = load(entry.name, channel, root=rroot)
        if cfg is None:
            continue
        if all(cfg.get(k) == v for k, v in matcher.items()):
            return entry.name
    return None
