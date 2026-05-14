"""Process-wide active-tenant pointer (P16.1).

Mirrors ``runtime/agent_context.py`` but one level higher. Every
persistent path now nests under ``tenants/<tenant_id>/...`` and this
module is the single source of truth for "which tenant are we serving
right now."

Set on three paths:
  - server lifespan after ``ensure_bootstrapped()`` resolves the
    tenant registry (defaults to ``_default``)
  - the ASGI tenant middleware after extracting ``x-tenant-id`` from
    the incoming request
  - ``OPENTRACY_TENANT_ID`` env var (tests + CLI override)

Reads always succeed: if nothing was set and no env override exists,
the resolver returns ``_default`` so writes still land in a partition
that ``ensure_bootstrapped()`` has prepared.
"""

from __future__ import annotations

import os
import threading
from typing import Optional


_DEFAULT_TENANT_ID = "_default"
_ENV_VAR = "OPENTRACY_TENANT_ID"

_lock = threading.Lock()
_active: Optional[str] = None


def set_active(tenant_id: Optional[str]) -> None:
    """Update the process-global pointer. ``None`` clears it so the
    next read falls back to env or default — useful for tests."""
    global _active
    with _lock:
        _active = tenant_id


def get_active(default: str = _DEFAULT_TENANT_ID) -> str:
    """Resolve the current tenant id. Order:
       1. process-global ``set_active`` value
       2. ``OPENTRACY_TENANT_ID`` env var
       3. ``default``
    """
    if _active:
        return _active
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return env
    return default
