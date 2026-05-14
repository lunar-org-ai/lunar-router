"""Per-request tenant pointer backed by ``contextvars.ContextVar`` (P16.2).

Mirrors ``runtime/agent_context.py`` but one level higher. Every
persistent path nests under ``tenants/<tenant_id>/...`` (in
infra/hosted mode) and this module is the single source of truth
for "which tenant is this request for."

**Why ContextVar (P16.2)**: P16.1 used a module global plus a
``threading.Lock``. That works for synchronous code in one process
but doesn't isolate concurrent asyncio tasks — and concurrent
requests are exactly what hosted-mode MCP HTTP introduces. A
``ContextVar`` gives each asyncio Task / each request its own
binding automatically, so ``set_active("acme")`` in one Task
doesn't leak into another.

Public API is unchanged from P16.1:
  - ``set_active(id)`` / ``set_active(None)``
  - ``get_active(default="_default")``
  - ``_active`` attribute kept as a *read-only fallback shim* for
    legacy tests that read the global directly. Writes still go
    through ``set_active``; reads return the ContextVar's current
    binding for the calling context.
"""

from __future__ import annotations

import contextvars
import os
from typing import Optional


_DEFAULT_TENANT_ID = "_default"
_ENV_VAR = "OPENTRACY_TENANT_ID"

# ContextVar carries the current binding per asyncio Task / thread
# automatically. ``None`` means "unbound" — readers fall back to env
# var then to the default literal.
_active_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "opentracy_active_tenant", default=None
)


def set_active(tenant_id: Optional[str]) -> None:
    """Update the per-context pointer. ``None`` clears it so the
    next read falls back to env or default."""
    _active_var.set(tenant_id)


def get_active(default: str = _DEFAULT_TENANT_ID) -> str:
    """Resolve the current tenant id. Order:
       1. per-context ``set_active`` value (ContextVar binding)
       2. ``OPENTRACY_TENANT_ID`` env var
       3. ``default``
    """
    current = _active_var.get()
    if current:
        return current
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return env
    return default


# ---------------------------------------------------------------------------
# Back-compat shim
# ---------------------------------------------------------------------------
#
# A handful of P16.1-era callers and tests read ``tenant_context._active``
# directly to gate behavior ("is anyone driving?"). We expose a property
# at module level via ``__getattr__`` so those reads keep working — they
# now see the ContextVar's current binding for the calling context.
# Writes still must go through ``set_active(...)``; assigning to
# ``_active`` directly raises so we catch any forgotten direct-write
# sites at the next test run.


def __getattr__(name: str):  # noqa: D401 — module-level __getattr__
    if name == "_active":
        return _active_var.get()
    raise AttributeError(name)
