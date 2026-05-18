"""Tests for per-agent MCP scoping.

Exercises:
  - Token mint with ``agent_id`` scope.
  - ``resolve_token_with_scope`` returning the right tuple.
  - The per-agent ASGI handler enforcing URL ↔ token scope agreement.
  - The integration-touch side effect that flips the live-status pill.

The actual MCP session manager is replaced with a recording stub —
this isolates the auth + scope logic from the heavy MCP SDK pathway
which has its own dedicated tests (test_mcp_http.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime.mcp.http import (
    _make_per_agent_streamable_handler,
    _touch_claude_code_integration,
)
from runtime.tenants.tokens import (
    mint_token,
    resolve_token,
    resolve_token_with_scope,
)


# ---------------------------------------------------------------------------
# Token scope
# ---------------------------------------------------------------------------


@pytest.fixture
def tenants_root(tmp_path):
    (tmp_path / "acme").mkdir()
    return tmp_path


def test_mint_token_defaults_to_tenant_wide(tenants_root):
    plaintext, record = mint_token("acme", "operator-key", root=tenants_root)
    assert record.agent_id is None
    assert resolve_token(plaintext, root=tenants_root) == "acme"
    assert resolve_token_with_scope(plaintext, root=tenants_root) == ("acme", None)


def test_mint_token_with_agent_id_persists_scope(tenants_root):
    plaintext, record = mint_token(
        "acme", "claude-code-default", agent_id="_default", root=tenants_root,
    )
    assert record.agent_id == "_default"

    # Round-trips on resolve.
    tenant, agent = resolve_token_with_scope(plaintext, root=tenants_root)
    assert tenant == "acme"
    assert agent == "_default"

    # Persisted in the per-tenant tokens.json so subsequent processes
    # reload the same scope.
    on_disk = json.loads((tenants_root / "acme" / "tokens.json").read_text())
    assert on_disk["tokens"][0]["agent_id"] == "_default"


def test_resolve_token_drops_scope_for_legacy_callers(tenants_root):
    plaintext, _ = mint_token(
        "acme", "claude-code", agent_id="_default", root=tenants_root,
    )
    # The plain resolve_token wrapper hides the agent_id so existing
    # tenant-wide call sites don't need to learn the new shape.
    assert resolve_token(plaintext, root=tenants_root) == "acme"


def test_unknown_token_returns_none(tenants_root):
    assert resolve_token_with_scope("otrcy_live_bogus", root=tenants_root) is None
    assert resolve_token_with_scope("", root=tenants_root) is None
    assert resolve_token_with_scope("not-a-bearer", root=tenants_root) is None


def test_old_tokens_json_without_agent_id_field_still_loads(tmp_path):
    """Back-compat: tokens minted before P16.4 don't carry agent_id."""
    tenant_dir = tmp_path / "acme"
    tenant_dir.mkdir()
    (tenant_dir / "tokens.json").write_text(json.dumps({"tokens": [
        {
            "hash": "a" * 64,
            "label": "legacy",
            "created_at": "2025-01-01T00:00:00Z",
            "last_used_at": None,
        },
    ]}))
    # Manually populate the index because we side-stepped mint_token.
    (tmp_path / "_tokens_index.json").write_text(json.dumps({"a" * 64: "acme"}))

    # Forge a plaintext whose hash matches what we wrote — easier than
    # injecting it via a mock. Use the production hash helper.
    import hashlib

    # We can't reverse a hash so synthesize a token whose hash we DO
    # know by pre-computing the legacy entry from a real plaintext.
    plaintext_seed = "otrcy_live_legacysample"
    legacy_hash = hashlib.sha256(plaintext_seed.encode("utf-8")).hexdigest()
    (tenant_dir / "tokens.json").write_text(json.dumps({"tokens": [
        {
            "hash": legacy_hash,
            "label": "legacy",
            "created_at": "2025-01-01T00:00:00Z",
            "last_used_at": None,
        },
    ]}))
    (tmp_path / "_tokens_index.json").write_text(json.dumps({legacy_hash: "acme"}))

    tenant, agent = resolve_token_with_scope(plaintext_seed, root=tmp_path)
    assert tenant == "acme"
    assert agent is None


# ---------------------------------------------------------------------------
# Per-agent ASGI handler
# ---------------------------------------------------------------------------


def _scope_for(*, path: str, agent_id: str, token: str | None) -> dict:
    headers: list[tuple[bytes, bytes]] = []
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode("latin-1")))
    return {
        "type": "http",
        "path": path,
        "method": "POST",
        "headers": headers,
        "path_params": {"agent_id": agent_id},
    }


async def _run_handler(handler, scope):
    """Drive the ASGI handler with a stub receive/send and capture the
    response start frame (status + headers) so tests can assert on it."""
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    receive = AsyncMock(return_value={"type": "http.request", "body": b""})
    await handler(scope, receive, send)
    return sent


@pytest.fixture
def fake_state():
    """Stub MCP session manager: records calls so we can assert the
    handler delegated AFTER passing the auth/scope gates."""
    session_manager = MagicMock()
    handled: list[dict] = []

    async def _handle_request(scope, receive, send):
        handled.append({"path": scope.get("path")})
        # Acknowledge the response so send queue isn't empty.
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    session_manager.handle_request = _handle_request
    return {"mcp_session_manager": session_manager, "_handled": handled}


@pytest.mark.asyncio
async def test_per_agent_handler_rejects_missing_token(fake_state):
    handler = _make_per_agent_streamable_handler(fake_state)
    sent = await _run_handler(
        handler,
        _scope_for(path="/mcp/agents/_default", agent_id="_default", token=None),
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 401
    assert fake_state["_handled"] == []


@pytest.mark.asyncio
async def test_per_agent_handler_rejects_unknown_token(fake_state):
    handler = _make_per_agent_streamable_handler(fake_state)
    sent = await _run_handler(
        handler,
        _scope_for(
            path="/mcp/agents/_default",
            agent_id="_default",
            token="otrcy_live_bogus",
        ),
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 401


@pytest.mark.asyncio
async def test_per_agent_handler_accepts_tenant_wide_token(
    fake_state, tenants_root, monkeypatch,
):
    """An operator's tenant-wide token works on per-agent endpoints —
    they have authority over every agent in the tenant."""
    monkeypatch.setattr("runtime.tenants.tokens._DEFAULT_ROOT", tenants_root)
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tenants_root,
    )
    plaintext, _ = mint_token("acme", "operator", root=tenants_root)
    (tenants_root / "acme" / "agents" / "_default").mkdir(parents=True)

    handler = _make_per_agent_streamable_handler(fake_state)
    sent = await _run_handler(
        handler,
        _scope_for(path="/mcp/agents/_default", agent_id="_default", token=plaintext),
    )

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200
    assert fake_state["_handled"] == [{"path": "/mcp/agents/_default"}]


@pytest.mark.asyncio
async def test_per_agent_handler_accepts_matching_scoped_token(
    fake_state, tenants_root, monkeypatch,
):
    monkeypatch.setattr("runtime.tenants.tokens._DEFAULT_ROOT", tenants_root)
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tenants_root,
    )
    plaintext, _ = mint_token(
        "acme", "claude-code", agent_id="sales-bot", root=tenants_root,
    )
    (tenants_root / "acme" / "agents" / "sales-bot").mkdir(parents=True)

    handler = _make_per_agent_streamable_handler(fake_state)
    sent = await _run_handler(
        handler,
        _scope_for(path="/mcp/agents/sales-bot", agent_id="sales-bot", token=plaintext),
    )

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200


@pytest.mark.asyncio
async def test_per_agent_handler_rejects_wrong_agent_scope(
    fake_state, tenants_root, monkeypatch,
):
    """The core isolation guarantee — a token scoped to agent A cannot
    drive agent B even within the same tenant."""
    monkeypatch.setattr("runtime.tenants.tokens._DEFAULT_ROOT", tenants_root)
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tenants_root,
    )
    plaintext, _ = mint_token(
        "acme", "claude-code-A", agent_id="agent-a", root=tenants_root,
    )

    handler = _make_per_agent_streamable_handler(fake_state)
    sent = await _run_handler(
        handler,
        _scope_for(path="/mcp/agents/agent-b", agent_id="agent-b", token=plaintext),
    )

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 403
    # Session manager must NOT have been reached.
    assert fake_state["_handled"] == []


@pytest.mark.asyncio
async def test_per_agent_handler_stamps_integration_marker(
    fake_state, tenants_root, monkeypatch,
):
    """First successful MCP call should write
    ``integrations/claude-code.json`` so the UI's live-status overlay
    can flip the agent's pill from 'Not configured' to 'Connected'."""
    monkeypatch.setattr("runtime.tenants.tokens._DEFAULT_ROOT", tenants_root)
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tenants_root,
    )
    plaintext, _ = mint_token(
        "acme", "claude-code", agent_id="sales-bot", root=tenants_root,
    )
    (tenants_root / "acme" / "agents" / "sales-bot").mkdir(parents=True)

    handler = _make_per_agent_streamable_handler(fake_state)
    await _run_handler(
        handler,
        _scope_for(path="/mcp/agents/sales-bot", agent_id="sales-bot", token=plaintext),
    )

    marker = (
        tenants_root / "acme" / "agents" / "sales-bot"
        / "integrations" / "claude-code.json"
    )
    assert marker.is_file()
    data = json.loads(marker.read_text())
    assert "last_used_at" in data
    assert data["last_used_at"].endswith("Z")


# ---------------------------------------------------------------------------
# Integration-touch helper (also called directly for the e2e smoke)
# ---------------------------------------------------------------------------


def test_touch_claude_code_integration_creates_and_updates(
    tenants_root, monkeypatch,
):
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tenants_root,
    )
    (tenants_root / "acme" / "agents" / "demo").mkdir(parents=True)

    _touch_claude_code_integration("acme", "demo")
    marker = (
        tenants_root / "acme" / "agents" / "demo"
        / "integrations" / "claude-code.json"
    )
    assert marker.is_file()
    first = json.loads(marker.read_text())["last_used_at"]

    # Pre-existing fields must survive subsequent touches.
    payload = json.loads(marker.read_text())
    payload["operator_note"] = "added Slack alert"
    marker.write_text(json.dumps(payload))

    import time
    time.sleep(1.1)  # iso second resolution
    _touch_claude_code_integration("acme", "demo")
    after = json.loads(marker.read_text())
    assert after["operator_note"] == "added Slack alert"
    assert after["last_used_at"] != first
