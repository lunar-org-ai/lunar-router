"""Server-level tests for the /admin/tenants/* endpoints (P16.1)."""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Spawn the FastAPI app with all storage roots redirected to
    ``tmp_path`` and the lifespan's heavy work stubbed out — we only
    want to exercise the /admin/tenants/* surface."""
    from fastapi.testclient import TestClient

    # Multi-tenant mode ON for these tests — that's the whole point of
    # the /admin/tenants/* surface. OSS local mode (flag off) doesn't
    # exercise these routes.
    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")

    # Re-route every storage helper to tmp_path so the test doesn't
    # touch the operator's real filesystem.
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tmp_path / "tenants"
    )
    monkeypatch.setattr(
        "runtime.tenants.tokens._DEFAULT_ROOT", tmp_path / "tenants"
    )
    monkeypatch.setattr(
        "runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents"
    )
    monkeypatch.setattr(
        "runtime.agents.registry._LIVE_AGENT_DIR", tmp_path / "agent"
    )

    # Seed a minimal live agent dir so the agent bootstrap step has
    # something to copy from.
    (tmp_path / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0.0.1\n")
    (tmp_path / "agent" / "prompts" / "system.md").write_text("seed")

    # Stub the heavy lifespan steps so the test isn't gated on the
    # full pipeline loader.
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
        yield c


# ---------------------------------------------------------------------------
# Tenants CRUD
# ---------------------------------------------------------------------------


def test_list_returns_default_after_bootstrap(client):
    r = client.get("/admin/tenants")
    assert r.status_code == 200, r.text
    body = r.json()
    ids = {t["id"] for t in body["tenants"]}
    assert "_default" in ids


def test_create_then_list_then_delete(client):
    r = client.post(
        "/admin/tenants",
        json={"name": "Acme Corp", "description": "Test"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "acme-corp"
    assert body["description"] == "Test"

    r2 = client.get("/admin/tenants")
    ids = {t["id"] for t in r2.json()["tenants"]}
    assert ids == {"_default", "acme-corp"}

    r3 = client.delete("/admin/tenants/acme-corp")
    assert r3.status_code == 204

    r4 = client.get("/admin/tenants")
    ids = {t["id"] for t in r4.json()["tenants"]}
    assert ids == {"_default"}


def test_create_with_explicit_slug(client):
    r = client.post(
        "/admin/tenants",
        json={"name": "Anything", "slug": "beta-co"},
    )
    assert r.status_code == 201
    assert r.json()["id"] == "beta-co"


def test_create_rejects_reserved_slug(client):
    r = client.post(
        "/admin/tenants", json={"name": "X", "slug": "_default"}
    )
    assert r.status_code == 400
    assert "reserved" in r.json()["detail"]


def test_create_rejects_bad_slug(client):
    r = client.post(
        "/admin/tenants", json={"name": "X", "slug": "UPPER"}
    )
    assert r.status_code == 400
    assert "invalid slug" in r.json()["detail"]


def test_delete_default_is_refused(client):
    r = client.delete("/admin/tenants/_default")
    assert r.status_code == 400


def test_delete_unknown_returns_404(client):
    r = client.delete("/admin/tenants/no-such-thing")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_mint_token_then_list(client):
    client.post("/admin/tenants", json={"name": "Acme"})

    r = client.post("/admin/tenants/acme/tokens", json={"label": "prod"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["token"].startswith("otrcy_live_")
    assert body["display"] == "show_once"
    assert body["record"]["label"] == "prod"

    r2 = client.get("/admin/tenants/acme/tokens")
    assert r2.status_code == 200
    assert len(r2.json()["tokens"]) == 1
    assert r2.json()["tokens"][0]["label"] == "prod"
    # plaintext never returned by list
    assert "token" not in r2.json()["tokens"][0]


def test_mint_token_for_unknown_tenant_returns_404(client):
    r = client.post("/admin/tenants/no-such/tokens", json={"label": "x"})
    assert r.status_code == 404


def test_revoke_token(client):
    client.post("/admin/tenants", json={"name": "Acme"})
    mint = client.post(
        "/admin/tenants/acme/tokens", json={"label": "prod"}
    ).json()
    hash_prefix = mint["record"]["hash_prefix"]

    r = client.delete(f"/admin/tenants/acme/tokens/{hash_prefix}")
    assert r.status_code == 204

    r2 = client.get("/admin/tenants/acme/tokens")
    assert r2.json()["tokens"] == []


def test_revoke_unknown_token_returns_404(client):
    client.post("/admin/tenants", json={"name": "Acme"})
    r = client.delete("/admin/tenants/acme/tokens/000000000000")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def test_resolve_token_returns_tenant_id(client):
    client.post("/admin/tenants", json={"name": "Acme"})
    plaintext = client.post(
        "/admin/tenants/acme/tokens", json={"label": "x"}
    ).json()["token"]

    r = client.post("/admin/tokens/resolve", json={"token": plaintext})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "acme"


def test_resolve_unknown_token_401(client):
    r = client.post(
        "/admin/tokens/resolve", json={"token": "otrcy_live_unknown"}
    )
    assert r.status_code == 401


def test_resolve_garbage_token_401(client):
    r = client.post("/admin/tokens/resolve", json={"token": "not-a-token"})
    assert r.status_code == 401


def test_token_isolation_between_tenants(client):
    """A token minted to acme must never resolve to beta — even if both
    tenants exist."""
    client.post("/admin/tenants", json={"name": "Acme"})
    client.post("/admin/tenants", json={"name": "Beta"})

    acme_token = client.post(
        "/admin/tenants/acme/tokens", json={"label": "x"}
    ).json()["token"]
    beta_token = client.post(
        "/admin/tenants/beta/tokens", json={"label": "x"}
    ).json()["token"]

    r1 = client.post("/admin/tokens/resolve", json={"token": acme_token})
    r2 = client.post("/admin/tokens/resolve", json={"token": beta_token})
    assert r1.json()["tenant_id"] == "acme"
    assert r2.json()["tenant_id"] == "beta"
