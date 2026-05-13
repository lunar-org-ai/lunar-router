"""Tests for the public API channel (P3.3.1).

Exercises connect/rotate/disconnect on /agents/<id>/channels/api and
the public /api/<id>/chat endpoint with bearer auth.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents")
    monkeypatch.setattr("runtime.agents.registry._LIVE_AGENT_DIR", tmp_path / "agent")

    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0.0.1\n")
    (tmp_path / "agent" / "prompts" / "system.md").write_text("seed")
    (tmp_path / "agent" / "pipeline").mkdir()
    (tmp_path / "agent" / "pipeline" / "route.yaml").write_text(
        "stage: route\nknobs:\n  small: claude-haiku-4-5\n"
    )

    monkeypatch.setattr("runtime.server._reload_live_pipeline", lambda agent_id=None: None)

    # Stub the pipeline so /api/<id>/chat doesn't try to call the real
    # generate stage (which would hit Anthropic).
    class _StubCfg: version = "v0.0.1"
    class _StubPipeline: stages = []
    class _StubRecord:
        response = "stub answer"
        duration_ms = 1.0
        success = True
        error = None
        stages = []
        agent_version = "v0.0.1"
        request = "what"
    class _StubExecutor:
        def run(self, request, history=None):
            return None, _StubRecord()
    monkeypatch.setattr("runtime.server.load_agent", lambda _p: _StubCfg())
    monkeypatch.setattr("runtime.server.compile_agent", lambda _cfg: _StubPipeline())
    monkeypatch.setattr("runtime.server.PipelineExecutor", lambda _p: _StubExecutor())
    monkeypatch.setattr("runtime.executor.tracing.write_trace", lambda rec, **kw: "stub-trace-id")

    from runtime.server import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Status / connect / rotate / disconnect
# ---------------------------------------------------------------------------


def test_get_returns_disconnected_initially(client):
    r = client.get("/agents/_default/channels/api")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is False


def test_connect_mints_token(client):
    r = client.post("/agents/_default/channels/api/connect")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["connected"] is True
    assert body["token"].startswith("ot_")
    assert "…" in body["token_mask"]
    assert body["public_url"].endswith("/api/_default/chat")


def test_connect_twice_returns_409(client):
    client.post("/agents/_default/channels/api/connect")
    r = client.post("/agents/_default/channels/api/connect")
    assert r.status_code == 409


def test_rotate_replaces_token(client):
    first = client.post("/agents/_default/channels/api/connect").json()
    second = client.post("/agents/_default/channels/api/rotate").json()
    assert first["token"] != second["token"]
    assert second["connected"] is True


def test_disconnect_returns_204(client):
    client.post("/agents/_default/channels/api/connect")
    r = client.delete("/agents/_default/channels/api")
    assert r.status_code == 204
    # Subsequent GET shows disconnected
    body = client.get("/agents/_default/channels/api").json()
    assert body["connected"] is False


# ---------------------------------------------------------------------------
# Public /api/<id>/chat endpoint
# ---------------------------------------------------------------------------


def test_chat_requires_bearer_token(client):
    """Missing Authorization header → 401."""
    r = client.post("/api/_default/chat", json={"request": "hi"})
    assert r.status_code == 401


def test_chat_rejects_invalid_token(client):
    client.post("/agents/_default/channels/api/connect")
    r = client.post(
        "/api/_default/chat",
        headers={"authorization": "Bearer wrong-token"},
        json={"request": "hi"},
    )
    assert r.status_code == 401


def test_chat_accepts_valid_token(client):
    connect = client.post("/agents/_default/channels/api/connect").json()
    token = connect["token"]
    r = client.post(
        "/api/_default/chat",
        headers={"authorization": f"Bearer {token}"},
        json={"request": "hello"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["response"] == "stub answer"
    assert body["trace_id"]


def test_chat_unknown_agent_404(client):
    r = client.post(
        "/api/no-such-agent/chat",
        headers={"authorization": "Bearer something"},
        json={"request": "hi"},
    )
    # Agent missing → 404 before token check
    assert r.status_code == 404
