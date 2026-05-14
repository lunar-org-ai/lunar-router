"""Tests for the ASGI tenant middleware (P16.1.S5)."""

from __future__ import annotations

import pytest


@pytest.fixture
def app_with_probe(tmp_path, monkeypatch):
    """Spawn the FastAPI app + add a probe endpoint that surfaces the
    tenant context at request time. The probe is registered on a fresh
    app instance to avoid module-import side effects."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")
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

    from runtime import tenant_context
    from runtime.server import app

    @app.get("/_test/whoami")
    def _whoami():
        return {"active": tenant_context.get_active()}

    with TestClient(app) as c:
        yield c, tenant_context


# ---------------------------------------------------------------------------
# Flag ON
# ---------------------------------------------------------------------------


def test_header_sets_active_tenant_for_request_duration(app_with_probe):
    c, _tc = app_with_probe
    r = c.get("/_test/whoami", headers={"x-tenant-id": "acme"})
    assert r.status_code == 200
    assert r.json()["active"] == "acme"


def test_missing_header_falls_back_to_default(app_with_probe):
    c, _tc = app_with_probe
    r = c.get("/_test/whoami")
    assert r.status_code == 200
    # No header set, no previous explicit active → resolver returns _default
    assert r.json()["active"] == "_default"


def test_middleware_restores_previous_after_request(app_with_probe):
    """The tenant pointer must not leak between requests."""
    c, tc = app_with_probe
    c.get("/_test/whoami", headers={"x-tenant-id": "acme"})
    # After the request, the global should be back to its pre-request state.
    assert tc._active is None


def test_consecutive_requests_each_pick_their_own_tenant(app_with_probe):
    c, _tc = app_with_probe
    r1 = c.get("/_test/whoami", headers={"x-tenant-id": "acme"})
    r2 = c.get("/_test/whoami", headers={"x-tenant-id": "beta"})
    assert r1.json()["active"] == "acme"
    assert r2.json()["active"] == "beta"


def test_empty_header_value_is_ignored(app_with_probe):
    """An explicitly empty x-tenant-id should NOT clobber whatever
    was previously set (e.g. an inner test-side ``set_active``)."""
    c, tc = app_with_probe
    tc.set_active("pre-set")
    try:
        r = c.get("/_test/whoami", headers={"x-tenant-id": "  "})
        assert r.json()["active"] == "pre-set"
    finally:
        tc.set_active(None)


# ---------------------------------------------------------------------------
# Flag OFF — OSS mode
# ---------------------------------------------------------------------------


def test_middleware_inert_in_oss_mode(tmp_path, monkeypatch):
    """When the multi-tenant flag is unset, the middleware must be a
    pure pass-through. Tenant header is ignored, context unchanged."""
    from fastapi.testclient import TestClient

    # Don't set OPENTRACY_MULTI_TENANT — default OSS behavior.
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

    from runtime import tenant_context
    from runtime.server import app

    @app.get("/_test/whoami_oss")
    def _whoami_oss():
        # Mirror the same probe in a different path so it can be
        # registered alongside the multi-tenant probe.
        return {"active": tenant_context._active}

    with TestClient(app) as c:
        # Even with header set, the middleware doesn't touch context
        r = c.get("/_test/whoami_oss", headers={"x-tenant-id": "acme"})
    assert r.status_code == 200
    assert r.json()["active"] is None
