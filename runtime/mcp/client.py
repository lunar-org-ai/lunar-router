"""Sync wrapper around the async MCP STDIO client (P3.4).

Each enabled server in ``agents/<id>/mcp.json`` gets its own subprocess
that we keep around for the life of the runtime. ``list_tools_for_agent``
returns the combined catalog; ``call_tool`` dispatches by tool name
into the right server (we prefix tool names with the server name to
keep them unique across servers).

Design constraints:

  * The MCP SDK is async-first; the runtime's generate stage runs in a
    FastAPI threadpool. We bridge by running a single asyncio event
    loop in a dedicated background thread (started lazily on first use)
    and submitting coroutines via ``run_coroutine_threadsafe``.

  * Tool catalogs are cached for ``_DISCOVERY_TTL_S``. Re-discover when
    the operator adds/removes/edits a server (best-effort: caller
    invalidates via ``invalidate_cache(agent_id)``).

  * Errors during discovery never crash the runtime — a misbehaving
    server just disappears from the catalog for this turn.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger("runtime.mcp.client")


_DISCOVERY_TTL_S = 60.0  # re-discover catalog after this many seconds


@dataclass
class DiscoveredTool:
    """One tool projected for the LLM. ``qualified_name`` is
    ``<server>__<tool>`` so tools from different servers don't collide
    in the model's tool table."""

    server_name: str
    tool_name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Anthropic ``tools=`` parameter shape."""
        return {
            "name": self.qualified_name,
            "description": self.description,
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """OpenAI ``tools=`` parameter shape."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


# ---------------------------------------------------------------------------
# Background asyncio loop bridge
# ---------------------------------------------------------------------------


class _LoopBridge:
    """One asyncio loop per process, running on a daemon thread."""

    _lock = threading.Lock()
    _instance: Optional["_LoopBridge"] = None

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run, name="mcp-loop", daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def get(cls) -> "_LoopBridge":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(self, coro, *, timeout: float = 30.0):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Per-server session cache (agent_id -> server_name -> _Session)
# ---------------------------------------------------------------------------


@dataclass
class _CachedCatalog:
    tools: list[DiscoveredTool]
    fetched_at: float


_catalog_cache: dict[str, _CachedCatalog] = {}
_catalog_lock = threading.Lock()


def invalidate_cache(agent_id: Optional[str] = None) -> None:
    """Drop the cached catalog. Pass ``None`` to clear everything."""
    with _catalog_lock:
        if agent_id is None:
            _catalog_cache.clear()
        else:
            _catalog_cache.pop(agent_id, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_tools_for_agent(
    agent_id: str,
    *,
    force_refresh: bool = False,
    timeout_s: float = 15.0,
) -> list[DiscoveredTool]:
    """Combined catalog across the agent's enabled MCP servers.

    Cached for ``_DISCOVERY_TTL_S``. Pass ``force_refresh=True`` after
    the operator mutates the server list (the server endpoints do this).
    Errors per-server are logged + dropped from the catalog.
    """
    with _catalog_lock:
        cached = _catalog_cache.get(agent_id)
        if cached and not force_refresh and (time.time() - cached.fetched_at) < _DISCOVERY_TTL_S:
            return cached.tools

    from runtime.agents.mcp import enabled_servers

    servers = enabled_servers(agent_id)
    if not servers:
        with _catalog_lock:
            _catalog_cache[agent_id] = _CachedCatalog(tools=[], fetched_at=time.time())
        return []

    bridge = _LoopBridge.get()
    discovered: list[DiscoveredTool] = []
    for srv in servers:
        try:
            tools = bridge.run(
                _discover_tools_for_server(srv),
                timeout=timeout_s,
            )
        except Exception as e:
            logger.warning(
                "mcp server %s/%s discovery failed (%s)", agent_id, srv.name, e,
            )
            continue
        discovered.extend(tools)

    with _catalog_lock:
        _catalog_cache[agent_id] = _CachedCatalog(tools=discovered, fetched_at=time.time())
    return discovered


def call_tool(
    agent_id: str,
    qualified_name: str,
    arguments: dict[str, Any],
    *,
    timeout_s: float = 30.0,
) -> str:
    """Invoke a tool by its ``<server>__<tool>`` qualified name. Returns
    the textual result. Raises RuntimeError on transport/server failure.
    """
    from runtime.agents.mcp import enabled_servers

    if "__" not in qualified_name:
        raise ValueError(f"qualified_name must be '<server>__<tool>', got {qualified_name!r}")
    server_name, tool_name = qualified_name.split("__", 1)

    servers = enabled_servers(agent_id)
    server = next((s for s in servers if s.name == server_name), None)
    if server is None:
        raise RuntimeError(f"mcp_server_not_found: {server_name}")

    bridge = _LoopBridge.get()
    return bridge.run(
        _invoke_tool_for_server(server, tool_name, arguments),
        timeout=timeout_s,
    )


# ---------------------------------------------------------------------------
# Async MCP work
# ---------------------------------------------------------------------------


def _server_headers(server) -> dict[str, str]:
    """Build the HTTP headers for hosted (sse/http) transports."""
    headers: dict[str, str] = {}
    token = getattr(server, "auth_token", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _streams_ctx(server):
    """Return the async context manager that yields the MCP
    (read_stream, write_stream) pair for the chosen transport.

    Branches by ``server.transport``:
      * stdio → subprocess via ``stdio_client``
      * sse   → Server-Sent Events via ``sse_client``
      * http  → streamable HTTP via ``streamablehttp_client``

    The hosted transports honor ``server.auth_token`` as a
    ``Bearer`` header so private MCP servers (Linear/Notion/etc) work.
    """
    transport = (server.transport or "stdio").lower()

    if transport == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        import os

        if not server.command:
            raise RuntimeError("missing command for stdio transport")
        process_env = {**os.environ, **(server.env or {})}
        params = StdioServerParameters(
            command=server.command, args=list(server.args), env=process_env,
        )
        return stdio_client(params)

    if transport == "sse":
        from mcp.client.sse import sse_client

        if not server.url:
            raise RuntimeError("missing url for sse transport")
        return sse_client(server.url, headers=_server_headers(server))

    if transport == "http":
        from mcp.client.streamable_http import streamablehttp_client

        if not server.url:
            raise RuntimeError("missing url for http transport")
        return streamablehttp_client(server.url, headers=_server_headers(server))

    raise RuntimeError(f"transport_not_implemented: {transport}")


def _normalize_streams(streams):
    """Some transports yield ``(read, write)`` and some yield
    ``(read, write, get_session_id)``. Normalize to the pair the
    rest of the code path expects."""
    if isinstance(streams, tuple) and len(streams) >= 2:
        return streams[0], streams[1]
    raise RuntimeError("transport did not return a (read, write) stream pair")


async def _discover_tools_for_server(server) -> list[DiscoveredTool]:
    from mcp import ClientSession

    out: list[DiscoveredTool] = []
    async with _streams_ctx(server) as streams:
        read_stream, write_stream = _normalize_streams(streams)
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            resp = await session.list_tools()
            for t in resp.tools:
                schema = getattr(t, "inputSchema", None) or {}
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump()
                out.append(
                    DiscoveredTool(
                        server_name=server.name,
                        tool_name=t.name,
                        qualified_name=f"{server.name}__{t.name}",
                        description=getattr(t, "description", "") or "",
                        input_schema=schema,
                    )
                )
    return out


async def _invoke_tool_for_server(
    server,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    from mcp import ClientSession

    async with _streams_ctx(server) as streams:
        read_stream, write_stream = _normalize_streams(streams)
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            parts: list[str] = []
            for block in (getattr(result, "content", []) or []):
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "\n\n".join(parts) or "(tool returned no text content)"
