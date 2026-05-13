"""Server-level tests for the /agents endpoints (P2.0)."""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    # Redirect the registry to tmp so tests don't share state with the
    # real ``agents/`` dir.
    monkeypatch.setattr("runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents")
    monkeypatch.setattr("runtime.agents.registry._LIVE_AGENT_DIR", tmp_path / "agent")

    # Seed a tiny live agent dir so bootstrap has something to migrate.
    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0.0.1\n")
    (tmp_path / "agent" / "prompts" / "system.md").write_text("seed prompt")

    # Avoid the server's lifespan from compiling the real agent — we only
    # want to exercise the /agents/* surface here. Patch _reload_live_pipeline
    # so the activate endpoint doesn't try to spin up the executor.
    monkeypatch.setattr("runtime.server._reload_live_pipeline", lambda: None)

    from runtime.server import app
    # We bypass the lifespan by using TestClient(app) without a context;
    # FastAPI runs lifespan on enter, which tries to load agent.yaml from
    # the current directory. Patch load_agent to return a stub instead.
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

    with TestClient(app) as c:
        yield c


def test_list_returns_default_after_bootstrap(client):
    r = client.get("/agents")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == "_default"
    assert any(a["id"] == "_default" for a in body["agents"])


def test_create_then_list_then_get(client):
    r = client.post(
        "/agents",
        json={
            "name": "support",
            "prompt": "You support customers.",
            "model": "claude-haiku-4-5",
            "channels": ["web"],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "support"
    assert body["model"] == "claude-haiku-4-5"
    assert body["is_active"] is False  # we didn't pass activate=true

    r2 = client.get("/agents")
    ids = {a["id"] for a in r2.json()["agents"]}
    assert ids == {"_default", "support"}

    r3 = client.get("/agents/support")
    assert r3.status_code == 200
    assert r3.json()["name"] == "support"


def test_create_with_activate_switches_live(client):
    r = client.post(
        "/agents",
        json={
            "name": "new-active",
            "prompt": "I am the new active agent.",
            "activate": True,
        },
    )
    assert r.status_code == 201

    r2 = client.get("/agents")
    body = r2.json()
    assert body["active"] == "new-active"
    new_meta = next(a for a in body["agents"] if a["id"] == "new-active")
    assert new_meta["is_active"] is True


def test_activate_unknown_returns_404(client):
    r = client.post("/agents/no-such-agent/activate")
    assert r.status_code == 404


def test_delete_active_returns_409(client):
    r = client.delete("/agents/_default")
    # _default cannot be deleted at all
    assert r.status_code == 409


def test_delete_inactive_returns_204(client):
    client.post(
        "/agents",
        json={"name": "kill-me", "prompt": "x"},
    )
    r = client.delete("/agents/kill-me")
    assert r.status_code == 204
    # Gone from listing
    listing = client.get("/agents").json()
    assert all(a["id"] != "kill-me" for a in listing["agents"])
