"""Action interface + registry.

Actions are plug-in-style: a new module anywhere under `actions/` calls
`register_action("name", fn)` at import time and it becomes callable by
any recipe. Mirrors the AgentRegistry pattern without needing a scan:
Python's import machinery does the discovery when the package is
imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


ActionFn = Callable[[dict[str, Any], Any, str], Awaitable["ActionResult"]]


@dataclass
class ActionResult:
    """Outcome of one action invocation.

    `ledger_entry_id` is the id of the `action` ledger row the executor
    (or the action itself) wrote. Recipes chain subsequent steps off it.
    """

    outcome: str  # "ok" | "failed" | "skipped" | "rolled_back"
    data: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_ms: int = 0
    ledger_entry_id: Optional[str] = None


_REGISTRY: dict[str, ActionFn] = {}


def register_action(name: str, fn: Optional[ActionFn] = None):
    """Register an action. Usable as a decorator or a direct call.

    Duplicate registrations overwrite — useful for tests that want to
    swap a stub in, a no-op for production since each module registers
    exactly once at import.
    """

    def _decorate(f: ActionFn) -> ActionFn:
        _REGISTRY[name] = f
        return f

    if fn is not None:
        return _decorate(fn)
    return _decorate


def get_action(name: str) -> Optional[ActionFn]:
    return _REGISTRY.get(name)


def list_actions() -> list[str]:
    return sorted(_REGISTRY.keys())
