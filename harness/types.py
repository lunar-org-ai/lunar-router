"""Harness data types: Proposal, CriticVerdict, CriticContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from experiments.types import Mutation


@dataclass
class Proposal:
    """A candidate mutation the proposer wants to test.

    A Proposal does not yet have a candidate_id — it's pre-branching. The loop
    orchestrator turns approved Proposals into branched candidates.
    """

    mutations: list[Mutation]
    description: Optional[str] = None
    source: str = "heuristic"     # "heuristic" | "claude_code" | "human" | …
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticVerdict:
    """One critic's verdict on a Proposal (or candidate result)."""

    critic: str
    approved: bool
    reason: str
    severity: str = "info"  # "info" | "warn" | "block"


@dataclass
class CriticContext:
    """What a critic sees.

    Pre-flight critics (scope, budget) only see `proposal`.
    Post-eval critics (eval_lift, regression) also see `candidate_result`.
    """

    proposal: Proposal
    candidate_result: Optional[Any] = None  # experiments.runner.CandidateResult


@dataclass
class LoopOutcome:
    """Final state of one proposer→critics→eval round for one Proposal."""

    proposal: Proposal
    candidate_id: Optional[str]            # set if the Proposal got branched
    verdicts: list[CriticVerdict] = field(default_factory=list)
    candidate_result: Optional[Any] = None # experiments.runner.CandidateResult
    final: str = "pending"                 # "approved" | "rejected" | "pending"

    @property
    def approved(self) -> bool:
        return self.final == "approved"
