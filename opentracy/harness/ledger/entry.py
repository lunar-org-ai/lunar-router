"""Pydantic model for a single ledger entry.

Every primitive execution in the harness (sensor, agent run, decision,
action) writes exactly one entry. Chain-of-causation is encoded via
`parent_id`: a root signal has parent_id=None; everything downstream
points back to its cause.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


EntryType = Literal[
    "signal",       # a sensor detected that something changed
    "run",          # an agent executed
    "observation",  # an inspector produced a structured observation
    "proposal",     # a proposer suggested a mutation
    "decision",     # a critic/executor decided to act or not
    "action",       # a side-effect was performed (eval, training, promote)
    "lesson",       # narrator distilled a closed chain into reusable knowledge
]

Outcome = Literal["ok", "failed", "rolled_back", "skipped"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LedgerEntry(BaseModel):
    """One append-only row in the harness ledger."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = Field(default_factory=_now_iso)
    type: EntryType

    objective_id: Optional[str] = None
    subject: Optional[str] = None
    agent: Optional[str] = None
    parameters_in: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    duration_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    outcome: Optional[Outcome] = None
