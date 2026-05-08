"""Ledger data types.

The ledger is append-only. Once written, an entry is immutable; rollbacks
write a *new* entry that references the prior version, they don't rewrite
history. Lessons are the user-visible cards in the UI; they're a derived
view over a curated subset of LedgerEntry rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ENTRY_KINDS = (
    "proposal",       # a Proposal was created (heuristic, claude_code, human)
    "critic_verdict", # a critic scored a proposal
    "candidate_run",  # candidate vs baseline scored
    "promote",        # candidate promoted → live agent/
    "rollback",       # live agent/ reverted to a prior version
    "queued_review",  # promotion queued for human (mode=review)
    "rejected",       # final rejection
)


@dataclass
class LedgerEntry:
    """One immutable row in the ledger."""

    entry_id: str           # ULID-like or uuid
    kind: str               # one of ENTRY_KINDS
    timestamp: str          # ISO 8601 with Z
    parent_entry_id: Optional[str] = None  # forms causal chains
    candidate_id: Optional[str] = None
    agent_version_before: Optional[str] = None
    agent_version_after: Optional[str] = None
    summary: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Lesson:
    """A curated, user-visible record of one approved or queued change.

    Maps to the lesson cards in ui/. Generated when a promotion lands or when
    the policy queues a candidate for human review. Carries the change kind,
    the proof (eval delta), and the trace lineage.

    Provisional lessons (status=awaiting_review) leave version and promoted_at
    as None until a human approves and the candidate is actually promoted.
    """

    id: str
    kind: str               # "prompt" | "router" | "rag" | "rerank" | …
    status: str             # "auto_promoted" | "approved" | "awaiting_review"
                            #   | "human_rejected" | "rolled_back"
    title: str
    summary: str
    proposal_source: str    # "heuristic" | "claude_code" | "human"
    delta: dict[str, Any]   # overall_score, pass_rate, per-rubric
    mutations: list[str]    # describe() output
    parent_version: str
    candidate_id: str
    version: Optional[str] = None        # set on promotion
    promoted_at: Optional[str] = None    # set on promotion
    ledger_entry_id: Optional[str] = None
    voice: Optional[str] = None          # one-line "in my own words" rendering
