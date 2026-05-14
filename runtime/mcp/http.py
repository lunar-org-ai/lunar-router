"""HTTP/SSE transport for the OpenTracy MCP server (P16.2).

Mounts the existing introspection tools (``harness/introspection/tools``)
on two transports, gated by ``OPENTRACY_MULTI_TENANT=1``:

  - **Streamable HTTP** (modern, MCP spec 2024-11) at ``/mcp/``.
    Single endpoint, bidirectional POST + optional SSE response.
  - **SSE** (legacy) at ``/mcp/sse`` for the handshake stream and
    ``/mcp/messages/`` for the client→server POST channel.

Every request must carry ``Authorization: Bearer otrcy_live_<…>``.
The middleware resolves the token through the tenant registry, sets
the active tenant on the ContextVar, then delegates to the SDK
transport. Tool handlers (in ``harness.introspection.lib``) read the
tenant via the resolver chain in ``ledger.writer``, ``runtime.store.traces``,
etc, so each tenant only sees its own data.

Lifecycle: ``StreamableHTTPSessionManager`` requires its ``run()``
async context to be active for the duration of the server. The
caller (``runtime/server.py``) enters :func:`mcp_lifespan` from the
FastAPI lifespan and stashes the manager + SSE transport on the
shared state dict so the route handlers can reach them.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any, AsyncIterator

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from harness.introspection.tools import register


logger = logging.getLogger("runtime.mcp.http")


_TOKEN_PREFIX = "otrcy_live_"
_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _extract_token(headers: dict[str, str] | None) -> str | None:
    """Pull the Bearer payload off the incoming Authorization header."""
    if not headers:
        return None
    # ASGI normalizes header names to lowercase byte strings; in
    # Starlette's Request.headers they're already strings.
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    m = _BEARER_RE.match(raw)
    if not m:
        return None
    return m.group(1).strip()


def _resolve_bearer(token: str | None) -> str | None:
    """Return tenant_id for a valid otrcy_live_ token, else None."""
    if not token or not token.startswith(_TOKEN_PREFIX):
        return None
    from runtime.tenants.tokens import resolve_token
    return resolve_token(token)


async def _send_401(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
    """ASGI helper to emit a JSON 401 before the MCP handshake runs."""
    body = JSONResponse({"error": "unauthorized", "detail": reason}, status_code=401)
    await body(scope, receive, send)


def _scope_headers(scope: Scope) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_key, raw_value in scope.get("headers") or []:
        try:
            out[raw_key.decode("latin-1").lower()] = raw_value.decode("latin-1")
        except Exception:  # pragma: no cover — defensive
            continue
    return out


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def mcp_lifespan(state: dict[str, Any]) -> AsyncIterator[None]:
    """Initialize the MCP server + session manager + SSE transport
    and stash them on the shared state dict. Yields once setup
    completes; tears down on exit.

    Call this from the FastAPI lifespan AFTER the tenant bootstrap so
    the introspection tools can read tenant-scoped data from the get-go.
    """
    server: Server = Server("opentracy-introspection")
    register(server)

    session_manager = StreamableHTTPSessionManager(app=server, stateless=False)
    # The SSE transport sends messages to clients via SSE and receives
    # client POSTs at the message endpoint we mount below.
    sse = SseServerTransport("/mcp/messages/")

    state["mcp_server"] = server
    state["mcp_session_manager"] = session_manager
    state["mcp_sse"] = sse

    logger.info("mcp: lifecycle started — Streamable HTTP + SSE ready")
    async with session_manager.run():
        try:
            yield
        finally:
            logger.info("mcp: lifecycle stopped")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def _send_503(scope: Scope, receive: Receive, send: Send) -> None:
    """Lifespan hasn't initialized the MCP transports — OSS mode or a
    crash before mcp_lifespan ran."""
    body = JSONResponse(
        {"error": "mcp_disabled", "detail": "MCP HTTP transport not enabled on this server"},
        status_code=503,
    )
    await body(scope, receive, send)


def _make_streamable_handler(state: dict[str, Any]):
    """Streamable HTTP entry point — wraps the session manager with
    Bearer auth + tenant ContextVar setup."""

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        if "mcp_session_manager" not in state:
            await _send_503(scope, receive, send)
            return

        token = _extract_token(_scope_headers(scope))
        tenant_id = _resolve_bearer(token)
        if tenant_id is None:
            await _send_401(scope, receive, send, "invalid or missing tenant token")
            return

        from runtime.tenant_context import set_active

        set_active(tenant_id)
        try:
            await state["mcp_session_manager"].handle_request(scope, receive, send)
        finally:
            set_active(None)

    return handler


def _make_sse_handlers(state: dict[str, Any]):
    """SSE legacy transport. ``connect_sse`` is the GET endpoint that
    streams server→client events; ``handle_post_message`` is where
    the client POSTs its outbound calls."""

    async def connect(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        if "mcp_sse" not in state or "mcp_server" not in state:
            await _send_503(scope, receive, send)
            return

        token = _extract_token(_scope_headers(scope))
        tenant_id = _resolve_bearer(token)
        if tenant_id is None:
            await _send_401(scope, receive, send, "invalid or missing tenant token")
            return

        from runtime.tenant_context import set_active

        set_active(tenant_id)
        try:
            sse: SseServerTransport = state["mcp_sse"]
            server: Server = state["mcp_server"]
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )
        finally:
            set_active(None)

    async def post_message(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        if "mcp_sse" not in state:
            await _send_503(scope, receive, send)
            return

        token = _extract_token(_scope_headers(scope))
        tenant_id = _resolve_bearer(token)
        if tenant_id is None:
            await _send_401(scope, receive, send, "invalid or missing tenant token")
            return

        from runtime.tenant_context import set_active

        set_active(tenant_id)
        try:
            sse: SseServerTransport = state["mcp_sse"]
            await sse.handle_post_message(scope, receive, send)
        finally:
            set_active(None)

    return connect, post_message


# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------


def build_mcp_routes(state: dict[str, Any]) -> list:
    """Routes the FastAPI app mounts at ``/mcp``.

    Both transports share the same ``Server`` instance; the SDK's
    session manager handles request-level isolation via session ids.
    """
    streamable = _make_streamable_handler(state)
    sse_get, sse_post = _make_sse_handlers(state)
    return [
        # Streamable HTTP — the modern transport. Single endpoint that
        # handles both POST (client→server) and optional SSE response.
        Mount("/", app=streamable),
        # SSE legacy. Some MCP clients (older Claude Code builds) only
        # support this transport.
        Route("/sse", endpoint=sse_get, methods=["GET"]),
        Mount("/messages/", app=sse_post),
    ]
