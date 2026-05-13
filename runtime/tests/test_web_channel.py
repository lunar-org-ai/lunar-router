"""Tests for the Web Widget channel (P3.5).

Exercises the operator-facing CRUD on /agents/<id>/channels/web plus
the public /widget/<widget_id>/message endpoint with origin pinning.
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


# ─── Operator-facing CRUD ─────────────────────────────────────────────────


def test_get_returns_disconnected_initially(client):
    r = client.get("/agents/_default/channels/web")
    assert r.status_code == 200
    assert r.json()["connected"] is False


def test_connect_mints_widget_id_and_secret(client):
    r = client.post("/agents/_default/channels/web/connect")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["connected"] is True
    assert body["widget_id"].startswith("w_")
    assert body["signing_secret"].startswith("whsec_")
    assert "…" in body["signing_secret_mask"]
    # Embed URL routes through /widget/<id>/v1.js
    assert body["embed_url"].endswith(f"/widget/{body['widget_id']}/v1.js")
    assert body["message_url"].endswith(f"/widget/{body['widget_id']}/message")


def test_connect_twice_returns_409(client):
    client.post("/agents/_default/channels/web/connect")
    r = client.post("/agents/_default/channels/web/connect")
    assert r.status_code == 409


def test_get_after_connect_returns_mask_not_secret(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    status = client.get("/agents/_default/channels/web").json()
    assert status["connected"] is True
    assert status["widget_id"] == fresh["widget_id"]
    # The raw secret is never returned on GET — only the mask.
    assert "signing_secret" not in status
    assert "…" in status["signing_secret_mask"]


def test_rotate_secret_replaces_only_secret(client):
    first = client.post("/agents/_default/channels/web/connect").json()
    rotated = client.post("/agents/_default/channels/web/rotate-secret").json()
    assert rotated["widget_id"] == first["widget_id"]  # widget_id stable
    assert rotated["signing_secret"] != first["signing_secret"]  # secret changed


def test_rotate_before_connect_returns_404(client):
    r = client.post("/agents/_default/channels/web/rotate-secret")
    assert r.status_code == 404


def test_patch_normalizes_and_dedupes_domains(client):
    client.post("/agents/_default/channels/web/connect")
    r = client.patch(
        "/agents/_default/channels/web",
        json={"allowed_domains": [
            "https://acme.com/path",   # protocol + path stripped
            "ACME.com",                # case folded → dedupes with above
            "  help.acme.com  ",       # trimmed
            "store.acme.com:8080",     # port stripped
        ]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["allowed_domains"] == ["acme.com", "help.acme.com", "store.acme.com"]


def test_patch_updates_settings(client):
    client.post("/agents/_default/channels/web/connect")
    r = client.patch(
        "/agents/_default/channels/web",
        json={"settings": {
            "position": "bl",
            "shape": "pill",
            "accent": "plum",
            "greeting": "Hi!",
            "welcome": "Welcome",
            "fallback": "Sorry",
            "show_greeting": False,
            "require_email": True,
            "pill_label": "Help",
        }},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings"]["position"] == "bl"
    assert body["settings"]["shape"] == "pill"


def test_disconnect_returns_204(client):
    client.post("/agents/_default/channels/web/connect")
    r = client.delete("/agents/_default/channels/web")
    assert r.status_code == 204
    assert client.get("/agents/_default/channels/web").json()["connected"] is False


# ─── Public /widget/<id>/message — origin pinning + routing ───────────────


def test_message_unknown_widget_returns_404(client):
    r = client.post("/widget/w_ghost/message", json={"message": "hi"})
    assert r.status_code == 404


def test_message_allows_localhost_when_no_domains_pinned(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    r = client.post(
        f"/widget/{fresh['widget_id']}/message",
        headers={"origin": "http://localhost:5173"},
        json={"message": "hello"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"] == "stub answer"
    assert body["session"].startswith("web_")
    assert body["trace_id"]


def test_message_rejects_disallowed_origin(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    client.patch(
        "/agents/_default/channels/web",
        json={"allowed_domains": ["acme.com"]},
    )
    r = client.post(
        f"/widget/{fresh['widget_id']}/message",
        headers={"origin": "https://evil.com"},
        json={"message": "hi"},
    )
    assert r.status_code == 403


def test_message_accepts_listed_origin(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    client.patch(
        "/agents/_default/channels/web",
        json={"allowed_domains": ["acme.com"]},
    )
    r = client.post(
        f"/widget/{fresh['widget_id']}/message",
        headers={"origin": "https://acme.com"},
        json={"message": "hi"},
    )
    assert r.status_code == 200


def test_message_subdomain_wildcard(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    client.patch(
        "/agents/_default/channels/web",
        json={"allowed_domains": ["*.acme.com"]},
    )
    r = client.post(
        f"/widget/{fresh['widget_id']}/message",
        headers={"origin": "https://help.acme.com"},
        json={"message": "hi"},
    )
    assert r.status_code == 200


def test_message_preserves_session_across_turns(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    first = client.post(
        f"/widget/{fresh['widget_id']}/message",
        headers={"origin": "http://localhost:5173"},
        json={"message": "hi"},
    ).json()
    second = client.post(
        f"/widget/{fresh['widget_id']}/message",
        headers={"origin": "http://localhost:5173"},
        json={"message": "again", "session": first["session"]},
    ).json()
    assert second["session"] == first["session"]


def test_options_preflight_returns_204(client):
    r = client.options(
        "/widget/w_anything/message",
        headers={"origin": "https://acme.com"},
    )
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == "https://acme.com"


# ─── Embed JS ─────────────────────────────────────────────────────────────


def test_embed_js_for_unknown_widget(client):
    r = client.get("/widget/w_ghost/v1.js")
    assert r.status_code == 404
    assert "application/javascript" in r.headers["content-type"]


def test_embed_js_includes_widget_id_and_endpoint(client):
    fresh = client.post("/agents/_default/channels/web/connect").json()
    r = client.get(f"/widget/{fresh['widget_id']}/v1.js")
    assert r.status_code == 200
    assert "application/javascript" in r.headers["content-type"]
    body = r.text
    assert fresh["widget_id"] in body
    assert "/widget/" in body
    assert "/message" in body
