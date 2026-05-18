"""Server endpoints for per-agent improvement config (P3.2)."""

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
    class _Stub: version = "v0.0.1"
    monkeypatch.setattr("runtime.server.load_agent", lambda _p: _Stub())
    monkeypatch.setattr("runtime.server.compile_agent", lambda _cfg: type("P", (), {"stages": []})())
    monkeypatch.setattr("runtime.server.PipelineExecutor", lambda _p: object())

    from runtime.server import app
    with TestClient(app) as c:
        yield c


def test_get_improvement_returns_defaults(client):
    r = client.get("/agents/_default/improvement")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["transport"] == "auto"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["cadence_minutes"] == 30


def test_put_improvement_persists_changes(client):
    r = client.put(
        "/agents/_default/improvement",
        json={
            "enabled": False,
            "transport": "claude_code_cli",
            "model": "claude-opus-4-7",
            "cadence_minutes": 60,
            "notes": "audit week",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["transport"] == "claude_code_cli"
    assert body["model"] == "claude-opus-4-7"
    assert body["cadence_minutes"] == 60
    assert body["notes"] == "audit week"

    # Read back
    body2 = client.get("/agents/_default/improvement").json()
    assert body2["transport"] == "claude_code_cli"


def test_put_improvement_partial_body(client):
    """Only listed fields update; others stay at current values."""
    # First set a full state
    client.put("/agents/_default/improvement", json={
        "enabled": True,
        "transport": "anthropic_api",
        "model": "claude-haiku-4-5",
        "cadence_minutes": 45,
    })
    # Then patch only enabled
    r = client.put("/agents/_default/improvement", json={"enabled": False})
    body = r.json()
    assert body["enabled"] is False
    # Other fields preserved
    assert body["transport"] == "anthropic_api"
    assert body["model"] == "claude-haiku-4-5"
    assert body["cadence_minutes"] == 45


def test_put_improvement_unknown_agent_404(client):
    r = client.put("/agents/no-such/improvement", json={"enabled": True})
    assert r.status_code == 404


def test_get_improvement_unknown_agent_404(client):
    r = client.get("/agents/no-such/improvement")
    assert r.status_code == 404


def test_put_improvement_invalid_transport_normalizes(client):
    """Unknown transport names get coerced to 'auto'."""
    r = client.put("/agents/_default/improvement", json={"transport": "telepathy"})
    assert r.status_code == 200
    assert r.json()["transport"] == "auto"
