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


# ---------------------------------------------------------------------------
# Concurrent isolation (P16.2 — ContextVar migration)
# ---------------------------------------------------------------------------


def test_concurrent_asyncio_tasks_each_have_their_own_active():
    """The whole point of the ContextVar migration: two tasks running
    in parallel must each see their own ``set_active(...)`` value, not
    leak through the module global the way P16.1's threading.Lock impl
    did."""
    import asyncio

    results: dict[str, list[str]] = {"a": [], "b": []}

    async def task(name: str, tenant: str, results_key: str):
        tenant_context.set_active(tenant)
        # Yield repeatedly to let the other task run in between
        for _ in range(5):
            results[results_key].append(tenant_context.get_active())
            await asyncio.sleep(0)

    async def run():
        await asyncio.gather(
            task("alpha", "acme", "a"),
            task("beta", "beta", "b"),
        )

    asyncio.run(run())

    # Each task only ever sees its own tenant — no cross-contamination.
    assert set(results["a"]) == {"acme"}
    assert set(results["b"]) == {"beta"}


def test_backcompat_read_of_dunder_active():
    """A handful of P16.1-era callers read ``tenant_context._active``
    directly. The shim must keep returning the current binding."""
    tenant_context.set_active("acme")
    try:
        assert tenant_context._active == "acme"
        tenant_context.set_active(None)
        assert tenant_context._active is None
    finally:
        tenant_context.set_active(None)
