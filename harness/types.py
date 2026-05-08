"""Harness data types: Proposal, Prediction, Critic*, LoopOutcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from experiments.types import Mutation


def kind_from_mutations(mutations: list[str]) -> str:
    """Map a candidate's mutations to one of the change kinds the policy
    overrides table can target. Used by both the executor (lesson labeling)
    and the approver (per-kind policy overrides)."""
    files = {m.split(":")[0] for m in mutations}
    if any("retrieve" in f for f in files):
        return "rag"
    if any("rerank" in f for f in files):
        return "rerank"
    if any("route" in f for f in files):
        return "router"
    if any("generate" in f or "prompts/" in f for f in files):
        return "prompt"
    if any("memory" in f for f in files):
        return "memory"
    return "other"


@dataclass
class Prediction:
    """A falsifiable claim about what a Proposal will do (AHE pillar 3).

    `rubric` may be a specific eval rubric name (e.g. "keywords_match") or
    the special string "overall" to predict the aggregate score. Direction is
    encoded in the sign of expected_delta.
    """

    rubric: str                  # rubric name or "overall"
    expected_delta: float        # signed; positive = improvement
    rationale: str
    confidence: float = 0.5      # optional 0-1; future routing use


@dataclass
class VerificationOutcome:
    """Did reality match the Prediction? Computed post-eval."""

    rubric: str
    expected_delta: float
    actual_delta: float
    direction_correct: bool      # sign(actual) == sign(expected)
    magnitude_met: bool           # abs(actual) >= abs(expected)
    verdict: str                  # "verified" | "partial" | "wrong" | "no_change"

    @classmethod
    def evaluate(
        cls, prediction: Prediction, actual_delta: float
    ) -> "VerificationOutcome":
        expected = prediction.expected_delta
        if expected == 0:
            verdict = "verified" if actual_delta == 0 else "no_change"
            return cls(
                rubric=prediction.rubric,
                expected_delta=expected,
                actual_delta=actual_delta,
                direction_correct=True,
                magnitude_met=actual_delta == 0,
                verdict=verdict,
            )

        same_sign = (actual_delta > 0 and expected > 0) or (
            actual_delta < 0 and expected < 0
        )
        magnitude_met = abs(actual_delta) >= abs(expected)

        if not same_sign and actual_delta != 0:
            verdict = "wrong"
        elif same_sign and magnitude_met:
            verdict = "verified"
        elif same_sign:
            verdict = "partial"
        else:
            verdict = "no_change"

        return cls(
            rubric=prediction.rubric,
            expected_delta=expected,
            actual_delta=actual_delta,
            direction_correct=same_sign,
            magnitude_met=magnitude_met,
            verdict=verdict,
        )


@dataclass
class Proposal:
    """A candidate mutation the proposer wants to test.

    A Proposal does not yet have a candidate_id — it's pre-branching. The loop
    orchestrator turns approved Proposals into branched candidates. If a
    Prediction is attached, the loop verifies it post-eval and records the
    outcome in the ledger (AHE: decision observability).
    """

    mutations: list[Mutation]
    description: Optional[str] = None
    source: str = "heuristic"     # "heuristic" | "claude_code" | "human" | …
    metadata: dict[str, Any] = field(default_factory=dict)
    prediction: Optional[Prediction] = None


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
    verification: Optional[VerificationOutcome] = None   # if proposal had prediction
    final: str = "pending"                 # "approved" | "rejected" | "pending"

    @property
    def approved(self) -> bool:
        return self.final == "approved"
