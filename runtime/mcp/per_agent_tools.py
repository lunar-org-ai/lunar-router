"""Per-agent MCP tools — the surface the customer's local Claude Code
sees when it connects to ``/mcp/agents/<agent_id>``.

The tools are deliberately narrow: they expose the agent's *workspace*
(plan, state, files), its *evolution history* (Change Manifest per AHE
§3.3), and a single write tool to queue a task for the autonomous arm.
Everything tenant-wide (introspection, ledger, router health) stays on
the operator-only ``/mcp`` surface.

Tools read the active agent from :mod:`runtime.agent_context` — the
per-agent ASGI handler in :mod:`runtime.mcp.http` sets it from the URL
before invoking the session manager, so handlers don't need to thread
the id manually.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool


logger = logging.getLogger("runtime.mcp.per_agent_tools")


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------


TOOLS: list[Tool] = [
    Tool(
        name="read_plan",
        description=(
            "Read the agent's current plan from .opentracy/memory/plan.md. "
            "This is the long-form narrative the autonomous arm rewrites "
            "each turn to track what it's doing and what's next."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="read_state",
        description=(
            "Read the agent's structured state from .opentracy/memory/state.json. "
            "Returns {next_step, facts, blockers, last_turn_at} — the "
            "machine-readable counterpart to the plan."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="read_system_prompt",
        description=(
            "Read the evolved system prompt at .opentracy/system_prompt.md. "
            "This is the AHE NexAU 'system_prompt' component — the harness's "
            "current instructions for the autonomous engineer."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="list_workspace_files",
        description=(
            "List relative paths under the agent's workspace, sorted, capped "
            "at the given limit (default 200)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5000,
                    "default": 200,
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="read_workspace_file",
        description=(
            "Read the UTF-8 contents of one file under the agent's workspace. "
            "Path is relative to the workspace root. Binary files return "
            "a marker rather than raw bytes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="list_nexau_components",
        description=(
            "Snapshot of which AHE NexAU components currently exist in this "
            "agent's harness — system_prompt, tools, middleware, skills, "
            "subagents, memory. Empty lists mean 'minimal-seed', i.e. the "
            "evolution loop hasn't authored anything for that slot yet."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="list_manifest_history",
        description=(
            "Recent AHE Change Manifest entries (Decision Observability, §3.3) — "
            "each is an evolution-time edit paired with its claimed fixes, "
            "at-risk regressions, and the verification verdict from the "
            "next round (confirmed / regressed / rolled_back)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="read_pending_manifest",
        description=(
            "Read the pending Change Manifest (an edit that has been written "
            "but not yet verified by the next round). Returns null when "
            "there's nothing pending."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="list_recent_sessions",
        description=(
            "List the agent's most recent runs (turns). Each entry carries "
            "the trace id, request, success flag, duration, and a snippet "
            "of the reply. Use get_session_log to read the full log."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="send_task",
        description=(
            "Queue a task for the autonomous engineering arm. The task lands "
            "in .opentracy/inbox/<iso>.md inside the workspace; the next "
            "turn picks it up. Returns the inbox path so the caller can "
            "confirm delivery."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "minLength": 1, "maxLength": 8000},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _require_agent() -> str:
    """Pull the active agent id off the process-global pointer.

    The per-agent ASGI handler sets this from the URL before delegating
    to the session manager. When called outside that path the resolver
    falls back to ``_default`` per :func:`runtime.agent_context.get_active`
    — fine for OSS mode but worth a debug log so multi-tenant misuse
    shows up in traces.
    """
    from runtime.agent_context import get_active

    agent_id = get_active(default="")
    if not agent_id:
        logger.debug(
            "per-agent MCP tool called without agent_context — falling back to _default"
        )
        agent_id = "_default"
    return agent_id


def read_plan() -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    return {"plan_markdown": ws.read_plan()}


def read_state() -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    return {"state": ws.read_state()}


def read_system_prompt() -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    return {"system_prompt": ws.read_system_prompt()}


def list_workspace_files(limit: int = 200) -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    files = ws.list_files(max_files=limit)
    return {"files": files, "count": len(files)}


def read_workspace_file(path: str) -> dict[str, Any]:
    """Read a single workspace file. Refuses absolute paths or anything
    that tries to escape the workspace via ``..``."""
    from runtime.workspaces import get_workspace
    from pathlib import PurePosixPath

    ws = get_workspace(_require_agent())

    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts:
        return {"error": "path_traversal", "detail": f"refusing {path!r}"}

    target = ws.path.joinpath(*parts)
    try:
        target = target.resolve()
        # The resolved target must stay under the workspace root —
        # guards against symlinks pointing outside.
        target.relative_to(ws.path.resolve())
    except (ValueError, OSError) as exc:
        return {"error": "path_outside_workspace", "detail": str(exc)}

    if not target.is_file():
        return {"error": "not_found", "path": path}
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        size = target.stat().st_size
        return {"error": "binary", "path": path, "size": size}
    return {"path": path, "content": text, "size": len(text)}


def list_nexau_components() -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    return {"components": ws.list_nexau_components()}


def list_manifest_history(limit: int = 10) -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    return {"entries": ws.list_manifest_history(limit=limit)}


def read_pending_manifest() -> dict[str, Any]:
    from runtime.workspaces import get_workspace
    ws = get_workspace(_require_agent())
    pending = ws.read_pending_manifest()
    return {"pending": pending}


def list_recent_sessions(limit: int = 10) -> dict[str, Any]:
    """Recent turns for this agent.

    Uses the existing ``runtime.store.traces`` helpers so we don't
    duplicate trace-storage logic. The traces module already routes
    per-tenant via the ContextVar that's set by the per-agent ASGI
    handler upstream.
    """
    agent_id = _require_agent()
    try:
        from runtime.store import traces as traces_store
    except Exception as exc:
        return {"error": "traces_unavailable", "detail": str(exc)}

    fn = getattr(traces_store, "list_recent", None)
    if fn is None:
        return {"error": "traces_unavailable", "detail": "list_recent not found"}
    try:
        items = fn(agent_id=agent_id, limit=limit)
    except TypeError:
        # Older signature without explicit agent_id kwarg — fall back.
        items = fn(limit=limit)
    except Exception as exc:
        logger.warning("traces.list_recent failed: %s", exc)
        return {"error": "traces_failed", "detail": str(exc)}
    return {"sessions": list(items or [])}


def send_task(task: str) -> dict[str, Any]:
    """Write the task into ``.opentracy/inbox/<iso>.md``.

    The inbox folder is the contract with the per-turn engineer: the
    autonomous arm's system prompt instructs it to drain the inbox on
    each invocation. Keeping this as plain markdown files (one per
    task) keeps the on-disk shape diffable and inspectable from
    git/gcsfuse without parsing a queue format.
    """
    from datetime import datetime, timezone
    from runtime.workspaces import get_workspace

    text = (task or "").strip()
    if not text:
        return {"error": "empty_task"}

    ws = get_workspace(_require_agent())
    inbox = ws.path / ".opentracy" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    stamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
        .replace(":", "-")
    )
    path = inbox / f"{stamp}.md"
    path.write_text(text + "\n", encoding="utf-8")
    return {"queued_at": stamp, "path": str(path.relative_to(ws.path))}


HANDLERS: dict[str, Any] = {
    "read_plan": read_plan,
    "read_state": read_state,
    "read_system_prompt": read_system_prompt,
    "list_workspace_files": list_workspace_files,
    "read_workspace_file": read_workspace_file,
    "list_nexau_components": list_nexau_components,
    "list_manifest_history": list_manifest_history,
    "read_pending_manifest": read_pending_manifest,
    "list_recent_sessions": list_recent_sessions,
    "send_task": send_task,
}


# ---------------------------------------------------------------------------
# Dispatch + wiring
# ---------------------------------------------------------------------------


def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    handler = HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"unknown tool: {name!r}")]
    cleaned = {k: v for k, v in (arguments or {}).items() if v != ""}
    try:
        result = handler(**cleaned)
    except Exception as exc:
        logger.warning("per-agent mcp tool %s failed: %s", name, exc, exc_info=True)
        return [TextContent(type="text", text=f"error: {type(exc).__name__}: {exc}")]
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


def register(server: Server) -> None:
    """Attach the per-agent tool surface to an MCP Server.

    The HTTP lifespan instantiates a dedicated Server for the per-agent
    mount so the customer's Claude Code only sees these tools (not the
    tenant-wide introspection ones).
    """

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return call_tool(name, arguments)
