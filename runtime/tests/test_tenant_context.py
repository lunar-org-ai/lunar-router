"""Tests for the process-wide tenant pointer (P16.1)."""

from __future__ import annotations

import os

import pytest

from runtime import tenant_context


@pytest.fixture(autouse=True)
def _reset_active_tenant(monkeypatch):
    """Wipe the module-global pointer + env var between tests so each
    case sees a clean process state."""
    tenant_context.set_active(None)
    monkeypatch.delenv(tenant_context._ENV_VAR, raising=False)
    yield
    tenant_context.set_active(None)


def test_default_resolves_to_underscore_default():
    assert tenant_context.get_active() == "_default"


def test_explicit_default_argument_wins_over_underscore_default():
    assert tenant_context.get_active(default="acme") == "acme"


def test_set_active_overrides_default():
    tenant_context.set_active("acme-corp")
    assert tenant_context.get_active() == "acme-corp"
    # Even if a custom default is passed, the explicit set_active wins.
    assert tenant_context.get_active(default="other") == "acme-corp"


def test_set_active_none_clears_back_to_default():
    tenant_context.set_active("acme-corp")
    tenant_context.set_active(None)
    assert tenant_context.get_active() == "_default"


def test_env_var_overrides_default_but_not_set_active(monkeypatch):
    monkeypatch.setenv(tenant_context._ENV_VAR, "from-env")
    # No set_active → env wins over default
    assert tenant_context.get_active() == "from-env"
    # set_active still wins over env
    tenant_context.set_active("from-call")
    assert tenant_context.get_active() == "from-call"


def test_empty_env_var_is_ignored(monkeypatch):
    monkeypatch.setenv(tenant_context._ENV_VAR, "   ")
    assert tenant_context.get_active() == "_default"
