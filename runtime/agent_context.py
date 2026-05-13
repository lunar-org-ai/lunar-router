"""Process-wide active-agent pointer (P2.1).

Every storage module that previously wrote to a flat path
(``ledger/entries/...``, ``traces/raw/...``) now writes to a partition
under the currently-active agent (``ledger/<agent_id>/entries/...``).
This module is the single source of truth for "which agent are we
writing as right now."

Set on three paths:
  - server lifespan after ``ensure_bootstrapped()`` resolves the
    registry's active pointer
  - ``/agents/<id>/activate`` endpoint after the live ``agent/`` dir
    is swapped
  - ``OPENTRACY_AGENT_ID`` env var (tests + CLI override)

Reads always succeed: if nothing was set and no env override exists,
the resolver returns ``_default`` so writes still land in a partition
that ``ensure_bootstrapped()`` has prepared.
"""

from __future__ import annotations

import os
import threading
from typing import Optional


_DEFAULT_AGENT_ID = "_default"
_ENV_VAR = "OPENTRACY_AGENT_ID"

_lock = threading.Lock()
_active: Optional[str] = None


def set_active(agent_id: Optional[str]) -> None:
    """Update the process-global pointer. ``None`` clears it so the
    next read falls back to env or default — useful for tests."""
    global _active
    with _lock:
        _active = agent_id


def get_active(default: str = _DEFAULT_AGENT_ID) -> str:
    """Resolve the current agent id. Order:
       1. process-global ``set_active`` value
       2. ``OPENTRACY_AGENT_ID`` env var
       3. ``default``
    """
    if _active:
        return _active
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return env
    return default
