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
from starlette.responses import JSONResponse, RedirectResponse, Response
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


def _resolve_bearer_with_scope(
    token: str | None,
) -> tuple[str, str | None] | None:
    """Return ``(tenant_id, agent_id)`` for a valid token, or None."""
    if not token or not token.startswith(_TOKEN_PREFIX):
        return None
    from runtime.tenants.tokens import resolve_token_with_scope
    return resolve_token_with_scope(token)


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
    """Initialize MCP servers + session managers + SSE transport.

    Two parallel servers, each with its own session manager so each
    surface exposes a different tool set:

      - ``mcp_server`` / ``mcp_session_manager`` — tenant-wide
        introspection (ledger, router health, lessons). Operator only.
      - ``mcp_agent_server`` / ``mcp_agent_session_manager`` — per-agent
        workspace + AHE Change Manifest tools. What the customer's
        local Claude Code sees via ``claude mcp add … /mcp/agents/<id>``.

    Call this from the FastAPI lifespan AFTER the tenant bootstrap so
    the introspection tools can read tenant-scoped data from the get-go.
    """
    server: Server = Server("opentracy-introspection")
    register(server)

    agent_server: Server = Server("opentracy-agent")
    from runtime.mcp.per_agent_tools import register as register_agent_tools
    register_agent_tools(agent_server)

    session_manager = StreamableHTTPSessionManager(app=server, stateless=False)
    agent_session_manager = StreamableHTTPSessionManager(
        app=agent_server, stateless=False,
    )
    # The SSE transport sends messages to clients via SSE and receives
    # client POSTs at the message endpoint we mount below. Tenant-wide
    # only — per-agent SSE isn't worth the complexity (Claude Code
    # speaks Streamable HTTP).
    sse = SseServerTransport("/mcp/messages/")

    state["mcp_server"] = server
    state["mcp_session_manager"] = session_manager
    state["mcp_agent_server"] = agent_server
    state["mcp_agent_session_manager"] = agent_session_manager
    state["mcp_sse"] = sse

    logger.info(
        "mcp: lifecycle started — Streamable HTTP (tenant + per-agent) + SSE ready"
    )
    async with session_manager.run(), agent_session_manager.run():
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


async def _send_400(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
    body = JSONResponse({"error": "bad_request", "detail": reason}, status_code=400)
    await body(scope, receive, send)


async def _send_403(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
    body = JSONResponse({"error": "forbidden", "detail": reason}, status_code=403)
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


def _make_per_agent_streamable_handler(state: dict[str, Any]):
    """Per-agent Streamable HTTP — same as the tenant-wide handler but
    also resolves the URL's ``agent_id`` from the mount path, enforces
    that the token's scope (if any) matches it, and sets the agent
    ContextVar in addition to the tenant one.

    Mounted at ``/mcp/agents/{agent_id}`` (see :func:`build_mcp_routes`).
    A tenant-wide token (``agent_id is None``) is allowed through so
    operators can hit per-agent paths without re-minting; an
    agent-scoped token whose ``agent_id`` disagrees with the URL is
    rejected with 403.

    Side effect: stamps the per-agent
    ``integrations/claude-code.json.last_used_at`` file so the live
    status overlay in the UI flips from "Not configured" to
    "Connected" on first MCP request — same pattern Slack/WhatsApp/Web
    use.
    """

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        # Prefer the dedicated per-agent manager (different tool surface);
        # fall back to the tenant one only if the lifespan hadn't been
        # upgraded yet (defensive — keeps existing tests passing during
        # the rollout).
        manager = (
            state.get("mcp_agent_session_manager")
            or state.get("mcp_session_manager")
        )
        if manager is None:
            await _send_503(scope, receive, send)
            return

        url_agent_id = (scope.get("path_params") or {}).get("agent_id")
        if not url_agent_id:
            # path_params should always be populated by the Route, but
            # belt + suspenders.
            await _send_400(scope, receive, send, "missing agent_id in URL")
            return

        token = _extract_token(_scope_headers(scope))
        scope_result = _resolve_bearer_with_scope(token)
        if scope_result is None:
            await _send_401(scope, receive, send, "invalid or missing tenant token")
            return
        tenant_id, token_agent_id = scope_result
        if token_agent_id is not None and token_agent_id != url_agent_id:
            await _send_403(
                scope,
                receive,
                send,
                f"token scoped to agent {token_agent_id!r}, URL targets {url_agent_id!r}",
            )
            return

        from runtime.agent_context import set_active as set_agent
        from runtime.tenant_context import set_active as set_tenant

        set_tenant(tenant_id)
        set_agent(url_agent_id)
        # Touch the live-status marker. Best-effort: any failure here
        # (e.g. fresh tenant without the integrations dir) should NOT
        # block the MCP request.
        try:
            _touch_claude_code_integration(tenant_id, url_agent_id)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("claude-code integration touch failed: %s", exc)

        try:
            await manager.handle_request(scope, receive, send)
        finally:
            set_agent(None)
            set_tenant(None)

    return handler


def _touch_claude_code_integration(tenant_id: str, agent_id: str) -> None:
    """Stamp ``agents/<id>/integrations/claude-code.json.last_used_at``
    so the UI's live status overlay flips to Connected."""
    import json as _json
    from datetime import datetime, timezone

    from runtime.tenants.registry import get_tenant_dir

    integ_dir = get_tenant_dir(tenant_id) / "agents" / agent_id / "integrations"
    integ_dir.mkdir(parents=True, exist_ok=True)
    path = integ_dir / "claude-code.json"
    now = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    payload: dict[str, Any] = {"last_used_at": now}
    if path.is_file():
        try:
            existing = _json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing["last_used_at"] = now
                payload = existing
        except _json.JSONDecodeError:
            pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


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

    All transports share the same ``Server`` instance; the SDK's
    session manager handles request-level isolation via session ids.

    Tenant-wide vs per-agent
    ------------------------

    Two parallel surfaces:

      - ``/mcp/`` (streamable + SSE) — tenant scope only. The token
        sets the tenant context; tools see the whole tenant catalog.
        Used for operator-level introspection.
      - ``/mcp/agents/<agent_id>/`` — same transport, but the URL pins
        the agent. Token must either be tenant-wide or scoped to the
        same agent. Tools see exactly one agent. This is the surface
        the customer's local Claude Code connects to via
        ``claude mcp add``.
    """
    streamable = _make_streamable_handler(state)
    per_agent_streamable = _make_per_agent_streamable_handler(state)
    sse_get, sse_post = _make_sse_handlers(state)

    async def _agent_no_slash_redirect(request: Request) -> Response:
        """Back-compat 308 for ``/mcp/agents/<id>`` (no trailing slash).

        Starlette's ``Mount("/agents/{agent_id}")`` compiles to a regex
        that requires the slash AFTER the captured param — without it,
        the URL falls through to ``Mount("/")`` and the client sees
        the wrong toolset. New install commands always include the
        slash; this redirect handles install lines minted before the
        fix shipped (mid-rollout) so they keep working."""
        agent_id = request.path_params.get("agent_id", "")
        return RedirectResponse(
            url=f"/mcp/agents/{agent_id}/",
            status_code=308,  # preserves method + body for POST
        )

    return [
        # No-trailing-slash compat. Must be REGISTERED before the
        # Mount below or Mount's pattern would shadow it (Mount with
        # path params still matches URLs that end exactly at the
        # param if the inner streamable transport doesn't insist on
        # path:path).
        Route(
            "/agents/{agent_id}",
            endpoint=_agent_no_slash_redirect,
            methods=["GET", "POST", "DELETE"],
        ),
        # Per-agent route — Mount (not Route) because the handler is a
        # raw ASGI callable, not a FastAPI endpoint. Mount with a path
        # pattern populates ``scope['path_params']['agent_id']`` and
        # strips the matched prefix before delegating, exactly what
        # the streamable session manager wants. Must be registered
        # before the catch-all Mount("/") below or it would shadow.
        Mount("/agents/{agent_id}", app=per_agent_streamable),
        # SSE legacy. Some MCP clients (older Claude Code builds) only
        # support this transport. Tenant-wide only for now — per-agent
        # SSE adds complexity for marginal benefit (modern Claude Code
        # supports Streamable HTTP).
        Route("/sse", endpoint=sse_get, methods=["GET"]),
        Mount("/messages/", app=sse_post),
        # Streamable HTTP — tenant-wide. Single endpoint that handles
        # both POST (client→server) and optional SSE response.
        Mount("/", app=streamable),
    ]
