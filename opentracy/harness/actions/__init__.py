"""Harness actions — code-level side effects dispatched at the end of a
task recipe.

An agent reasons. An action does. Actions are the only layer that
writes to ClickHouse, touches weights, or calls external systems —
keeping reasoning and side effects separate makes recipes testable
without stubbing the whole world.

Each action is a module-level coroutine with the signature:

    async def execute(
        inputs: dict[str, Any],
        ledger: LedgerStore,
        parent_id: str,
    ) -> ActionResult

and registers itself via `@register_action("name")`. Recipes reference
actions by name, same pattern as agents.
"""

from .base import ActionResult, get_action, list_actions, register_action
from . import fetch_cost_summary  # noqa: F401 — import registers the action
from . import fetch_dataset_samples  # noqa: F401 — import registers the action
from . import queue_training  # noqa: F401 — import registers the action
from . import run_clustering  # noqa: F401 — import registers the action
from . import run_eval  # noqa: F401 — import registers the action

__all__ = [
    "ActionResult",
    "get_action",
    "list_actions",
    "register_action",
]
