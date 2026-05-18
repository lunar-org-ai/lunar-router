"""Per-agent MCP server config (P3.4).

Each agent declares zero or more MCP servers it can call as tools at
runtime. Config lives at ``agents/<id>/mcp.json``:

  {
    "servers": [
      {
        "name": "files",
        "transport": "stdio",
        "command": "uv",
        "args": ["run", "python", "-m", "mcp_server_filesystem", "/tmp"],
        "env": {"FOO": "bar"},
        "enabled": true
      },
      ...
    ]
  }

The generate stage (P3.4 T3) consults this on every /run and injects
the discovered tools into the LLM call. Tool discovery is cached
in-process: the first call spawns each enabled server, fetches its
``listTools`` response, and reuses the connection until restart.

This module is the metadata store. ``runtime/mcp/client.py`` handles
the actual STDIO connections + RPC.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("runtime.agents.mcp")


_FILENAME = "mcp.json"
# stdio = subprocess (local/OSS); sse = Server-Sent Events; http =
# streamable HTTP (modern MCP HTTP transport). The latter two cover
# hosted MCP servers (Linear, Slack, etc).
_VALID_TRANSPORTS = ("stdio", "sse", "http")


@dataclass
class MCPServer:
    name: str                        # operator-chosen label, must be unique per agent
    transport: str = "stdio"         # "stdio" | "sse" | "http"
    command: str = ""                # for stdio: executable path or PATH name
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None        # for sse / http transports
    enabled: bool = True
    description: str = ""
    # Optional Bearer token for hosted MCP servers. Persisted as plain
    # text in mcp.json today — the per-agent dir already sits inside
    # the tenant's KMS-protected partition. A future tightening can
    # move this into the runtime/agents/secrets KMS store.
    auth_token: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "enabled": self.enabled,
            "description": self.description,
        }
        if self.url:
            d["url"] = self.url
        if self.auth_token:
            d["auth_token"] = self.auth_token
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServer":
        return cls(
            name=str(data.get("name", "")),
            transport=_normalize_transport(data.get("transport", "stdio")),
            command=str(data.get("command", "") or ""),
            args=[str(a) for a in (data.get("args") or [])],
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            url=data.get("url"),
            enabled=bool(data.get("enabled", True)),
            description=str(data.get("description", "") or ""),
            auth_token=data.get("auth_token") or None,
        )


# ---------------------------------------------------------------------------
# Path resolution — honors registry root (consistent with secrets/improvement)
# ---------------------------------------------------------------------------


def _path(agent_id: str, *, root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root) / agent_id / _FILENAME
    try:
        from runtime.agents import registry as _reg
        return _reg.agents_root() / agent_id / _FILENAME
    except Exception:
        return Path("agents") / agent_id / _FILENAME


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def load(agent_id: str, *, root: Optional[Path] = None) -> list[MCPServer]:
    """Return the list of configured MCP servers. Empty list when no
    file or unreadable."""
    path = _path(agent_id, root=root)
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, list):
            logger.warning("mcp.json at %s missing 'servers' array — empty", path)
            return []
        return [MCPServer.from_dict(s) for s in servers if isinstance(s, dict)]
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("failed to parse mcp.json at %s: %s", path, e)
        return []


def save(
    agent_id: str,
    servers: list[MCPServer],
    *,
    root: Optional[Path] = None,
) -> Path:
    """Persist the full list (replace, not merge). Atomic via .tmp rename."""
    path = _path(agent_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "servers": [s.to_dict() for s in servers],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Convenience: list/add/remove/update
# ---------------------------------------------------------------------------


def add_server(
    agent_id: str,
    server: MCPServer,
    *,
    root: Optional[Path] = None,
) -> list[MCPServer]:
    """Append a server. Raises ValueError on duplicate name."""
    existing = load(agent_id, root=root)
    if any(s.name == server.name for s in existing):
        raise ValueError(f"server_already_exists: {server.name}")
    existing.append(server)
    save(agent_id, existing, root=root)
    return existing


def update_server(
    agent_id: str,
    name: str,
    *,
    transport: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
    url: Optional[str] = None,
    enabled: Optional[bool] = None,
    description: Optional[str] = None,
    root: Optional[Path] = None,
) -> list[MCPServer]:
    """Partial update by name. Raises KeyError if not found."""
    existing = load(agent_id, root=root)
    target = next((s for s in existing if s.name == name), None)
    if target is None:
        raise KeyError(name)
    if transport is not None:
        target.transport = _normalize_transport(transport)
    if command is not None:
        target.command = command
    if args is not None:
        target.args = list(args)
    if env is not None:
        target.env = dict(env)
    if url is not None:
        target.url = url
    if enabled is not None:
        target.enabled = enabled
    if description is not None:
        target.description = description
    save(agent_id, existing, root=root)
    return existing


def remove_server(
    agent_id: str,
    name: str,
    *,
    root: Optional[Path] = None,
) -> list[MCPServer]:
    """Drop a server by name. No-op when missing."""
    existing = load(agent_id, root=root)
    filtered = [s for s in existing if s.name != name]
    save(agent_id, filtered, root=root)
    return filtered


def enabled_servers(
    agent_id: str,
    *,
    root: Optional[Path] = None,
) -> list[MCPServer]:
    """Enabled subset — what the generate stage actually spawns."""
    return [s for s in load(agent_id, root=root) if s.enabled]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_transport(raw: Any) -> str:
    v = str(raw or "stdio").strip().lower()
    return v if v in _VALID_TRANSPORTS else "stdio"
