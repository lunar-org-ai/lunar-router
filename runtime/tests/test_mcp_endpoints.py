"""Server endpoints for MCP CRUD (P3.4)."""

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

    monkeypatch.setattr("runtime.server._reload_live_pipeline", lambda agent_id=None: None)
    class _StubCfg: version = "v0.0.1"
    monkeypatch.setattr("runtime.server.load_agent", lambda _p: _StubCfg())
    monkeypatch.setattr("runtime.server.compile_agent", lambda _cfg: type("P", (), {"stages": []})())
    monkeypatch.setattr("runtime.server.PipelineExecutor", lambda _p: object())

    from runtime.server import app
    with TestClient(app) as c:
        yield c


def test_list_empty(client):
    r = client.get("/agents/_default/mcp")
    assert r.status_code == 200
    assert r.json()["servers"] == []


def test_add_then_list(client):
    r = client.post(
        "/agents/_default/mcp",
        json={"name": "fs", "command": "npx", "args": ["-y", "@mcp/fs", "/tmp"]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert [s["name"] for s in body["servers"]] == ["fs"]
    assert body["servers"][0]["transport"] == "stdio"


def test_add_duplicate_409(client):
    client.post("/agents/_default/mcp", json={"name": "fs", "command": "npx"})
    r = client.post("/agents/_default/mcp", json={"name": "fs", "command": "other"})
    assert r.status_code == 409


def test_add_missing_name_400(client):
    r = client.post("/agents/_default/mcp", json={"name": "", "command": "npx"})
    assert r.status_code == 400


def test_patch_updates(client):
    client.post("/agents/_default/mcp", json={"name": "fs", "command": "npx", "enabled": True})
    r = client.patch("/agents/_default/mcp/fs", json={"enabled": False, "description": "off"})
    assert r.status_code == 200
    server = r.json()["servers"][0]
    assert server["enabled"] is False
    assert server["description"] == "off"


def test_patch_unknown_server_404(client):
    r = client.patch("/agents/_default/mcp/ghost", json={"enabled": False})
    assert r.status_code == 404


def test_delete(client):
    client.post("/agents/_default/mcp", json={"name": "fs", "command": "npx"})
    r = client.delete("/agents/_default/mcp/fs")
    assert r.status_code == 204
    listing = client.get("/agents/_default/mcp").json()
    assert listing["servers"] == []


def test_discover_tools_empty_when_no_servers(client, monkeypatch):
    """The discovery endpoint hits the live MCP client; with no servers
    it returns an empty catalog (no subprocess spawned)."""
    r = client.get("/agents/_default/mcp/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["tools"] == []


def test_unknown_agent_404(client):
    assert client.get("/agents/no-such/mcp").status_code == 404
    assert client.post("/agents/no-such/mcp", json={"name": "x", "command": "y"}).status_code == 404
    assert client.delete("/agents/no-such/mcp/x").status_code == 404
