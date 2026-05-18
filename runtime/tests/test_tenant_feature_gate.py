"""Tests for the OSS-vs-infra feature gate (P16.1).

The OSS local distribution must keep working with zero multi-tenant
machinery active. The hosted/infra distribution flips the flag and
gets tenant routing, migration on boot, etc. These tests pin both
modes.
"""

from __future__ import annotations

import pytest

from runtime.tenants.feature import is_multi_tenant_enabled


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPENTRACY_MULTI_TENANT", raising=False)
    yield


def test_flag_off_by_default():
    assert is_multi_tenant_enabled() is False


def test_flag_on_when_env_set_to_one(monkeypatch):
    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")
    assert is_multi_tenant_enabled() is True


@pytest.mark.parametrize("val", ["0", "true", "yes", "TRUE", "", " "])
def test_flag_only_accepts_literal_one(monkeypatch, val):
    """No truthy-coercion — only the exact string "1" enables.

    Keeps the behavior explicit and prevents accidental activation
    from a value like "true" that the operator thought was a boolean.
    """
    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", val)
    assert is_multi_tenant_enabled() is False


# ---------------------------------------------------------------------------
# Resolver routing
# ---------------------------------------------------------------------------


def test_agents_root_uses_legacy_when_flag_off(monkeypatch, tmp_path):
    """OSS mode: agents_root() returns the legacy _DEFAULT_ROOT and
    ignores tenant_context entirely."""
    monkeypatch.setattr(
        "runtime.agents.registry._DEFAULT_ROOT", tmp_path / "agents"
    )
    # Even with a tenant active, OSS mode bypasses the tenant routing.
    from runtime import tenant_context
    from runtime.agents.registry import agents_root

    tenant_context.set_active("acme")
    try:
        assert agents_root() == tmp_path / "agents"
    finally:
        tenant_context.set_active(None)


def test_agents_root_uses_tenant_path_when_flag_on(monkeypatch, tmp_path):
    """Infra mode: agents_root() routes through tenants/<active>/agents/."""
    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tmp_path / "tenants"
    )
    from runtime import tenant_context
    from runtime.agents.registry import agents_root

    tenant_context.set_active("acme")
    try:
        assert agents_root() == tmp_path / "tenants" / "acme" / "agents"
    finally:
        tenant_context.set_active(None)


def test_agents_root_falls_back_to_default_tenant_when_no_active(
    monkeypatch, tmp_path
):
    """Infra mode with no per-request tenant set → _default tenant.
    Background tasks + boot-time lookups land here."""
    monkeypatch.setenv("OPENTRACY_MULTI_TENANT", "1")
    monkeypatch.setattr(
        "runtime.tenants.registry._DEFAULT_ROOT", tmp_path / "tenants"
    )
    from runtime import tenant_context
    from runtime.agents.registry import agents_root

    assert tenant_context._active is None
    assert agents_root() == tmp_path / "tenants" / "_default" / "agents"
