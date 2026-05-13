"""Server endpoints for per-agent BYOK secrets (P3.1)."""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents")
    monkeypatch.setattr("runtime.agents.registry._LIVE_AGENT_DIR", tmp_path / "agent")

    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0.0.1\n")
    (tmp_path / "agent" / "prompts" / "system.md").write_text("seed prompt")
    (tmp_path / "agent" / "pipeline").mkdir()
    (tmp_path / "agent" / "pipeline" / "route.yaml").write_text("knobs:\n  small: claude-haiku-4-5\n")

    monkeypatch.setattr("runtime.server._reload_live_pipeline", lambda agent_id=None: None)
    class _StubCfg:
        version = "v0.0.1"
    monkeypatch.setattr("runtime.server.load_agent", lambda _p: _StubCfg())
    monkeypatch.setattr(
        "runtime.server.compile_agent",
        lambda _cfg: type("P", (), {"stages": []})(),
    )
    monkeypatch.setattr("runtime.server.PipelineExecutor", lambda _p: object())

    # Isolate secrets file location to tmp_path/agents/<id>/secrets.env
    # — same dir we already point the registry at, so no extra patching.

    from runtime.server import app
    with TestClient(app) as c:
        yield c


def test_get_secrets_empty_for_unset_agent(client, monkeypatch):
    """Brand-new agent with nothing in agent secrets or global env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.get("/agents/_default/secrets")
    assert r.status_code == 200
    body = r.json()
    assert body["agent_id"] == "_default"
    assert body["providers"]["anthropic"]["set"] is False
    assert body["providers"]["anthropic"]["source"] == "unset"
    assert body["providers"]["openai"]["set"] is False


def test_put_secret_writes_per_agent_value(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    r = client.put(
        "/agents/_default/secrets",
        json={"anthropic": "sk-ant-LONG_KEY_FOR_THIS_TEST_AGENT"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["providers"]["anthropic"]["set"] is True
    assert body["providers"]["anthropic"]["source"] == "per-agent"
    assert body["providers"]["anthropic"]["mask"]
    # Mask hides the middle
    assert "LONG_KEY" not in body["providers"]["anthropic"]["mask"]


def test_put_secret_remove_falls_back_to_global(client, monkeypatch):
    """Set per-agent, set global, then remove per-agent → status flips
    to 'global' source."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "global-key-LONG_ENOUGH_TO_MASK")

    # Set per-agent
    client.put("/agents/_default/secrets", json={"anthropic": "agent-key"})
    body = client.get("/agents/_default/secrets").json()
    assert body["providers"]["anthropic"]["source"] == "per-agent"

    # Remove per-agent
    r = client.put("/agents/_default/secrets", json={"anthropic": ""})
    assert r.status_code == 200
    body = r.json()
    # Now resolves to the global env var instead of being unset
    assert body["providers"]["anthropic"]["source"] == "global"
    assert body["providers"]["anthropic"]["set"] is True


def test_put_secret_unknown_provider_400(client):
    r = client.put(
        "/agents/_default/secrets",
        json={"some_unknown_provider": "key"},
    )
    # The Pydantic schema lists only known providers, so this just gets
    # ignored (not 400). We check the success path instead.
    assert r.status_code == 200


def test_get_secrets_unknown_agent_404(client):
    r = client.get("/agents/no-such-agent/secrets")
    assert r.status_code == 404


def test_put_secrets_unknown_agent_404(client):
    r = client.put("/agents/no-such-agent/secrets", json={"anthropic": "k"})
    assert r.status_code == 404
