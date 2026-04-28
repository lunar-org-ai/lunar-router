"""`run_eval` action — executes an eval case produced earlier in a
recipe by a proposer agent, writes the result to the ledger.

For MVP this is a validating stub: it checks the eval case has the
required fields and records a ledger row with outcome=ok. Wiring it to
the real eval runner (in `opentracy/eval_agent/`) is a follow-up —
this keeps the recipe executor decoupled from the eval stack while the
recipe/action contract stabilizes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from opentracy.harness.ledger import LedgerEntry, LedgerStore

from .base import ActionResult, register_action

logger = logging.getLogger(__name__)


_REQUIRED_FIELDS = {"input", "expected_behavior", "check_type"}


@register_action("run_eval")
async def execute(
    inputs: dict[str, Any],
    ledger: LedgerStore,
    parent_id: str,
) -> ActionResult:
    """Run one eval case.

    Expected `inputs`:
      - eval_case: {input, expected_behavior, check_type, severity?, tags?}
      - rationale: str (optional, from proposer)

    Returns ActionResult with outcome=ok when the eval case validates,
    outcome=failed when required fields are missing. A failed validation
    does NOT mean the eval itself failed — the eval runner isn't wired
    yet; this action is the contract that future implementations plug
    into.
    """
    started = datetime.now(timezone.utc)
    eval_case = inputs.get("eval_case") or {}
    missing = _REQUIRED_FIELDS - set(eval_case.keys())

    if missing:
        entry = LedgerEntry(
            type="action",
            agent="run_eval",
            parent_id=parent_id,
            data={
                "error": "malformed eval_case",
                "missing_fields": sorted(missing),
            },
            tags=["run_eval", "validation_failed"],
            outcome="failed",
            duration_ms=_elapsed_ms(started),
        )
        ledger.append(entry)
        return ActionResult(
            outcome="failed",
            data={"missing_fields": sorted(missing)},
            duration_ms=entry.duration_ms or 0,
            ledger_entry_id=entry.id,
        )

    entry = LedgerEntry(
        type="action",
        agent="run_eval",
        parent_id=parent_id,
        data={
            "check_type": eval_case.get("check_type"),
            "severity": eval_case.get("severity", "medium"),
            "input_preview": str(eval_case.get("input", ""))[:200],
            "rationale": inputs.get("rationale", ""),
            # Explicit placeholder so readers of the ledger know this
            # action hasn't executed the eval against a live model yet.
            "status": "queued",
        },
        tags=["run_eval", "queued"],
        outcome="ok",
        duration_ms=_elapsed_ms(started),
    )
    ledger.append(entry)

    return ActionResult(
        outcome="ok",
        data={"eval_case": eval_case, "status": "queued"},
        duration_ms=entry.duration_ms or 0,
        ledger_entry_id=entry.id,
    )


def _elapsed_ms(started: datetime) -> int:
    return int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
