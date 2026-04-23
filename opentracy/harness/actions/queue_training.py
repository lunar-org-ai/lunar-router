"""`queue_training` action — records the decision to run a distillation
training job as a queued ledger row.

For MVP this does not kick off real training. The action writes an
`action` ledger row with the proposed config and outcome=ok so the
dashboard can show "training queued on <date>" in the chain drawer. A
follow-up PR will wire the real trainer, keeping the same ledger
contract so the dashboard's drill-down stays stable.

Expected inputs (from the critic step via input_from):
  - decision: "approve" | "reject" — if anything other than approve,
    the action refuses to queue and records outcome=skipped.
  - recommendation: str (from proposer, typically "train_now")
  - suggested_config: dict (from proposer)
  - estimated_cost_usd: float (from critic)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from opentracy.harness.ledger import LedgerEntry, LedgerStore

from .base import ActionResult, register_action

logger = logging.getLogger(__name__)


@register_action("queue_training")
async def execute(
    inputs: dict[str, Any],
    ledger: LedgerStore,
    parent_id: str,
) -> ActionResult:
    started = datetime.now(timezone.utc)

    decision = inputs.get("decision")
    if decision != "approve":
        # Record the refusal so the ledger tells the full story, but
        # outcome=skipped reflects that no training was queued.
        entry = LedgerEntry(
            type="action",
            agent="queue_training",
            parent_id=parent_id,
            data={
                "status": "not_queued",
                "reason": f"critic decision is {decision!r}, not 'approve'",
            },
            tags=["queue_training", "not_queued"],
            outcome="skipped",
            duration_ms=_elapsed_ms(started),
        )
        ledger.append(entry)
        return ActionResult(
            outcome="skipped",
            data={"status": "not_queued"},
            duration_ms=entry.duration_ms or 0,
            ledger_entry_id=entry.id,
        )

    suggested_config = inputs.get("suggested_config") or {}
    recommendation = inputs.get("recommendation", "train_now")
    estimated_cost = _coerce_cost(inputs.get("estimated_cost_usd"))

    payload = {
        "status": "queued",
        "recommendation": recommendation,
        "suggested_config": suggested_config,
        "estimated_cost_usd": estimated_cost,
        # Explicit placeholder: MVP records but does not launch the
        # trainer. A follow-up wires this to auto_trainer.
        "execution": "deferred",
    }

    entry = LedgerEntry(
        type="action",
        agent="queue_training",
        parent_id=parent_id,
        data=payload,
        tags=["queue_training", "queued", "training"],
        outcome="ok",
        cost_usd=estimated_cost or 0.0,
        duration_ms=_elapsed_ms(started),
    )
    ledger.append(entry)

    return ActionResult(
        outcome="ok",
        data=payload,
        cost_usd=entry.cost_usd or 0.0,
        duration_ms=entry.duration_ms or 0,
        ledger_entry_id=entry.id,
    )


def _coerce_cost(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _elapsed_ms(started: datetime) -> int:
    return int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
