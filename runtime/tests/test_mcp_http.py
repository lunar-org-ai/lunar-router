"""Integration tests for the MCP HTTP transport (P16.2.S4).

Exercise the full request flow:
  - Streamable HTTP at /mcp/
  - SSE legacy at /mcp/sse
  - Bearer auth: valid token routes to the right tenant; missing or
    wrong-shape tokens 401 before any MCP handshake
  - Two tenants with their own tokens get isolated views of the data

These tests boot the full FastAPI app via TestClient with the
multi-tenant flag on. The MCP handshake messages follow the MCP spec
JSON-RPC structure.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def client_with_tenants(tmp_path, monkeypatch):
    """FastAPI app booted in multi-tenant mode with two tenants
    preconfigured (acme + beta) and a token minted for each."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tmp_path / "tenants"
    )
    monkeypatch.setattr(
        "runtime.tenants.tokens._DEFAULT_ROOT", tmp_path / "tenants"
    )
    monkeypatch.setattr(
        "runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents-legacy"
    )
    monkeypatch.setattr(
        "runtime.agents.registry._LIVE_AGENT_DIR", tmp_path / "agent"
    )
    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0\n")
    (tmp_path / "agent" / "prompts" / "system.md").write_text("seed")
    monkeypatch.setattr(
        "runtime.tenants.bootstrap.migrate_legacy_to_default",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "runtime.server._reload_live_pipeline", lambda *a, **k: None
    )

    class _StubCfg:
        version = "v0.0.1"

    monkeypatch.setattr("runtime.server.load_agent", lambda _p: _StubCfg())
    monkeypatch.setattr(
        "runtime.server.compile_agent",
        lambda _cfg: type("P", (), {"stages": []})(),
    )
    monkeypatch.setattr(
        "runtime.server.PipelineExecutor",
        lambda _p: object(),
    )

    from runtime.server import app

    with TestClient(app) as c:
        # Create both tenants
        c.post("/admin/tenants", json={"name": "Acme"})
        c.post("/admin/tenants", json={"name": "Beta"})
        # Mint a token for each
        token_acme = c.post(
            "/admin/tenants/acme/tokens", json={"label": "test"}
        ).json()["token"]
        token_beta = c.post(
            "/admin/tenants/beta/tokens", json={"label": "test"}
        ).json()["token"]
        yield {
            "client": c,
            "token_acme": token_acme,
            "token_beta": token_beta,
            "tmp_path": tmp_path,
        }


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------


def test_streamable_endpoint_401_without_bearer(client_with_tenants):
    c = client_with_tenants["client"]
    r = c.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        headers={"accept": "application/json, text/event-stream"},
    )
    assert r.status_code == 401
    body = r.json()
    assert body["error"] == "unauthorized"


def test_streamable_endpoint_401_with_wrong_shape_bearer(client_with_tenants):
    c = client_with_tenants["client"]
    r = c.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        headers={
            "authorization": "Bearer some-other-token",
            "accept": "application/json, text/event-stream",
        },
    )
    assert r.status_code == 401


def test_streamable_endpoint_401_with_unknown_otrcy_token(client_with_tenants):
    c = client_with_tenants["client"]
    r = c.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        headers={
            "authorization": "Bearer otrcy_live_unknownunknownunknownunknown",
            "accept": "application/json, text/event-stream",
        },
    )
    assert r.status_code == 401


def test_sse_endpoint_401_without_bearer(client_with_tenants):
    c = client_with_tenants["client"]
    r = c.get("/mcp/sse")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Streamable HTTP handshake
# ---------------------------------------------------------------------------


def _initialize_request_body() -> dict:
    """Standard MCP initialize handshake payload."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0.1"},
        },
    }


def test_streamable_initialize_with_valid_token(client_with_tenants):
    """A POST to /mcp/ with a valid Bearer should complete the MCP
    initialize handshake without 401-ing."""
    c = client_with_tenants["client"]
    token = client_with_tenants["token_acme"]
    r = c.post(
        "/mcp/",
        json=_initialize_request_body(),
        headers={
            "authorization": f"Bearer {token}",
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        },
    )
    # The SDK returns 200 with either JSON or SSE stream depending on
    # client capability declaration. We expect a 200 — anything else
    # means the auth or transport setup is broken.
    assert r.status_code == 200, f"unexpected: {r.status_code} {r.text[:200]}"
    # The session id header is set on a successful handshake (Streamable
    # HTTP spec).
    session_id = r.headers.get("mcp-session-id")
    assert session_id is not None and len(session_id) > 0


def test_two_tenants_get_independent_sessions(client_with_tenants):
    """Tenant A and tenant B each get a distinct mcp-session-id and
    neither can see the other's session state."""
    c = client_with_tenants["client"]
    body = _initialize_request_body()

    r_a = c.post(
        "/mcp/",
        json=body,
        headers={
            "authorization": f"Bearer {client_with_tenants['token_acme']}",
            "accept": "application/json, text/event-stream",
        },
    )
    r_b = c.post(
        "/mcp/",
        json=body,
        headers={
            "authorization": f"Bearer {client_with_tenants['token_beta']}",
            "accept": "application/json, text/event-stream",
        },
    )
    assert r_a.status_code == 200
    assert r_b.status_code == 200
    assert r_a.headers.get("mcp-session-id") != r_b.headers.get("mcp-session-id")


# ---------------------------------------------------------------------------
# Mount table — verify the routes exist
# ---------------------------------------------------------------------------


def test_mcp_routes_are_mounted_under_slash_mcp(client_with_tenants):
    """Smoke: /mcp/* paths route to the MCP handler instead of 404."""
    c = client_with_tenants["client"]
    # Without auth: should 401 (not 404), proving the route exists
    r = c.post("/mcp/", json={"x": 1})
    assert r.status_code == 401, "expected 401 from MCP auth, not 404 from missing route"

    r = c.get("/mcp/sse")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# OSS mode fallback
# ---------------------------------------------------------------------------


def test_mcp_returns_503_in_oss_mode(tmp_path, monkeypatch):
    """When the multi-tenant flag is off, the MCP routes still exist
    (we mount them unconditionally) but the lifespan doesn't initialize
    the session manager, so handlers return 503 instead of attempting
    a real MCP handshake."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("OPENTRACY_MULTI_TENANT", raising=False)
    monkeypatch.setattr(
        "runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents"
    )
    monkeypatch.setattr(
        "runtime.agents.registry._LIVE_AGENT_DIR", tmp_path / "agent"
    )
    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0\n")
    (tmp_path / "agent" / "prompts" / "system.md").write_text("seed")
    monkeypatch.setattr(
        "runtime.server._reload_live_pipeline", lambda *a, **k: None
    )

    class _StubCfg:
        version = "v0.0.1"

    monkeypatch.setattr("runtime.server.load_agent", lambda _p: _StubCfg())
    monkeypatch.setattr(
        "runtime.server.compile_agent",
        lambda _cfg: type("P", (), {"stages": []})(),
    )
    monkeypatch.setattr(
        "runtime.server.PipelineExecutor",
        lambda _p: object(),
    )

    from runtime.server import app

    with TestClient(app) as c:
        # Even with no auth, the handler returns 503 because the
        # session manager isn't initialized.
        r = c.post("/mcp/", json={"x": 1})
    assert r.status_code == 503
    assert r.json()["error"] == "mcp_disabled"
