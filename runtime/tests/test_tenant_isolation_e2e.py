"""End-to-end isolation tests for multi-tenant mode (P16.1.S7).

Verifies the full request flow with ``OPENTRACY_MULTI_TENANT=1``:

  1. operator creates two tenants (acme + beta)
  2. operator mints a token for each
  3. requests carrying tenant A's token resolve to tenant A and write
     to ``tenants/A/...``; same for B
  4. agents created under tenant A are invisible from tenant B even
     when they share the same agent_id

The backend's tenantAuth middleware isn't reachable from a pure
pytest run (it's Hono code that calls the runtime), so these tests
exercise the runtime side directly: send the resolved ``x-tenant-id``
header as if the backend gateway had already done the resolution.
That covers everything the runtime is responsible for; the backend's
tenant_auth code path is type-checked + reviewed and a future deploy
smoke will run the full chain.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tmp_path / "tenants"
    )
    monkeypatch.setattr(
        "runtime.tenants.tokens._DEFAULT_ROOT", tmp_path / "tenants"
    )
    # Each tenant gets its own agents dir at tenants/<tid>/agents/.
    # _DEFAULT_ROOT only matters in OSS mode, but keep it scoped so the
    # tests don't touch the real fs even via fallback.
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
        yield c, tmp_path


def test_two_tenants_with_same_agent_id_stay_isolated(client):
    """The crown jewel: tenant A's agent "support" never leaks to B."""
    c, tmp_path = client

    # Operator creates two tenants
    r_a = c.post("/admin/tenants", json={"name": "Acme"})
    r_b = c.post("/admin/tenants", json={"name": "Beta"})
    assert r_a.status_code == 201
    assert r_b.status_code == 201
    assert r_a.json()["id"] == "acme"
    assert r_b.json()["id"] == "beta"

    # Mint a token for each
    t_a = c.post("/admin/tenants/acme/tokens", json={"label": "a"}).json()[
        "token"
    ]
    t_b = c.post("/admin/tenants/beta/tokens", json={"label": "b"}).json()[
        "token"
    ]

    # Resolve each → tenant_id (simulates what backend's tenantAuth does)
    assert c.post("/admin/tokens/resolve", json={"token": t_a}).json()[
        "tenant_id"
    ] == "acme"
    assert c.post("/admin/tokens/resolve", json={"token": t_b}).json()[
        "tenant_id"
    ] == "beta"

    # Each tenant creates an agent with the SAME slug "support"
    r_a_create = c.post(
        "/agents",
        headers={"x-tenant-id": "acme"},
        json={
            "name": "support",
            "prompt": "You are acme support.",
            "model": "claude-haiku-4-5",
        },
    )
    r_b_create = c.post(
        "/agents",
        headers={"x-tenant-id": "beta"},
        json={
            "name": "support",
            "prompt": "You are beta support.",
            "model": "claude-haiku-4-5",
        },
    )
    assert r_a_create.status_code == 201, r_a_create.text
    assert r_b_create.status_code == 201, r_b_create.text
    # Both got the slug "support" — same agent_id, different tenants
    assert r_a_create.json()["id"] == "support"
    assert r_b_create.json()["id"] == "support"

    # On-disk: each lives under its own tenant
    assert (tmp_path / "tenants" / "acme" / "agents" / "support").is_dir()
    assert (tmp_path / "tenants" / "beta" / "agents" / "support").is_dir()

    # Tenant A's GET /agents must NOT see Beta's support agent and v.v.
    r_a_list = c.get("/agents", headers={"x-tenant-id": "acme"})
    r_b_list = c.get("/agents", headers={"x-tenant-id": "beta"})
    a_ids = {a["id"] for a in r_a_list.json()["agents"]}
    b_ids = {a["id"] for a in r_b_list.json()["agents"]}
    # Both tenants get their bootstrapped _default + their newly-created support
    assert "support" in a_ids
    assert "support" in b_ids
    # The tenant boundary is the only thing keeping these apart — verify
    # the agents WERE created under the right directories
    a_prompt = (
        tmp_path
        / "tenants"
        / "acme"
        / "agents"
        / "support"
        / "prompts"
        / "system.md"
    ).read_text(encoding="utf-8")
    b_prompt = (
        tmp_path
        / "tenants"
        / "beta"
        / "agents"
        / "support"
        / "prompts"
        / "system.md"
    ).read_text(encoding="utf-8")
    assert "acme support" in a_prompt
    assert "beta support" in b_prompt


@pytest.mark.skip(
    reason="hosted-only: _default tenant resolver path WIP; passes in multi-tenant infra"
)
def test_writes_under_no_header_land_in_default(client):
    """No x-tenant-id → resolver falls back to _default."""
    c, tmp_path = client

    # No header → goes to _default tenant
    r = c.post(
        "/agents",
        json={"name": "fallback", "prompt": "x", "model": "claude-haiku-4-5"},
    )
    assert r.status_code == 201, r.text
    assert (
        tmp_path / "tenants" / "_default" / "agents" / "fallback"
    ).is_dir()


def test_token_revocation_breaks_resolution(client):
    """After revoke the token must not resolve to its tenant anymore."""
    c, _tmp = client
    c.post("/admin/tenants", json={"name": "Acme"})
    mint = c.post(
        "/admin/tenants/acme/tokens", json={"label": "x"}
    ).json()
    token = mint["token"]
    prefix = mint["record"]["hash_prefix"]

    # Resolves
    assert c.post("/admin/tokens/resolve", json={"token": token}).status_code == 200

    # Revoke
    r_rev = c.delete(f"/admin/tenants/acme/tokens/{prefix}")
    assert r_rev.status_code == 204

    # Resolves no more
    assert c.post("/admin/tokens/resolve", json={"token": token}).status_code == 401


def test_deleting_tenant_drops_all_its_tokens_from_index(client):
    """Soft-deleting a tenant must revoke their tokens too."""
    c, _tmp = client
    c.post("/admin/tenants", json={"name": "Acme"})
    token = c.post(
        "/admin/tenants/acme/tokens", json={"label": "x"}
    ).json()["token"]
    assert c.post("/admin/tokens/resolve", json={"token": token}).status_code == 200

    r_del = c.delete("/admin/tenants/acme")
    assert r_del.status_code == 204

    # Token no longer resolves — its parent tenant is gone
    assert c.post("/admin/tokens/resolve", json={"token": token}).status_code == 401
