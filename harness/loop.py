"""Harness loop orchestration: Proposal → critics → branch → score → critics.

This is the seam where the harness becomes self-driving. Given a list of
Proposals and a critic configuration, the loop:

  1. Runs all PRE-stage critics on each Proposal (cheap; skips bad mutations).
  2. For survivors: branches a candidate, runs the suite, scores.
  3. Runs all POST-stage critics on each candidate result (eval lift,
     regression checks).
  4. Returns one LoopOutcome per Proposal.

Approved outcomes are what the next phase (`harness/approver/`) consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dataclasses import dataclass

from experiments.branching import create_candidate
from experiments.runner import CandidateResult, run_candidate
from harness.approver import ApprovalDecision, Policy, decide
from harness.critics import Critic, CriticStage, make_critic
from harness.executor import promote
from harness.types import CriticContext, CriticVerdict, LoopOutcome, Proposal

DEFAULT_PRE_CRITICS = ["scope"]
DEFAULT_POST_CRITICS = ["eval_lift"]


@dataclass
class LoopRound:
    """One full pass through the loop for a single proposal."""

    outcome: LoopOutcome
    decision: ApprovalDecision
    promoted_version: Optional[str] = None
    promoted_lesson_id: Optional[str] = None


def _split_critics(
    pre_names: list[str], post_names: list[str]
) -> tuple[list[Critic], list[Critic]]:
    """Build critic instances and verify each is in its expected stage."""
    pre = [make_critic(n) for n in pre_names]
    post = [make_critic(n) for n in post_names]
    for c in pre:
        if c.stage != CriticStage.PRE:
            raise ValueError(f"critic {c.name!r} is {c.stage.value}, expected pre")
    for c in post:
        if c.stage != CriticStage.POST:
            raise ValueError(f"critic {c.name!r} is {c.stage.value}, expected post")
    return pre, post


def _run_critics(
    critics: list[Critic],
    ctx: CriticContext,
) -> tuple[list[CriticVerdict], bool]:
    """Run every critic; return verdicts and whether any blocked."""
    verdicts: list[CriticVerdict] = []
    blocked = False
    for c in critics:
        v = c.verdict(ctx)
        verdicts.append(v)
        if not v.approved and v.severity == "block":
            blocked = True
    return verdicts, blocked


def propose_and_score(
    proposals: list[Proposal],
    suite_path: Path | str,
    pre_critics: Optional[list[str]] = None,
    post_critics: Optional[list[str]] = None,
) -> list[LoopOutcome]:
    """Run the full pipeline for a batch of proposals.

    Returns one LoopOutcome per input proposal (same order). The caller
    decides what to do with `approved` outcomes — promote to live, queue
    for human review, etc.
    """
    pre, post = _split_critics(
        pre_critics if pre_critics is not None else DEFAULT_PRE_CRITICS,
        post_critics if post_critics is not None else DEFAULT_POST_CRITICS,
    )

    outcomes: list[LoopOutcome] = []
    for proposal in proposals:
        outcome = LoopOutcome(proposal=proposal, candidate_id=None)

        # Pre-flight critics
        ctx_pre = CriticContext(proposal=proposal)
        pre_verdicts, pre_blocked = _run_critics(pre, ctx_pre)
        outcome.verdicts.extend(pre_verdicts)

        if pre_blocked:
            outcome.final = "rejected"
            outcomes.append(outcome)
            continue

        # Branch + score
        manifest = create_candidate(proposal.mutations, description=proposal.description)
        outcome.candidate_id = manifest.id
        result: CandidateResult = run_candidate(manifest.id, suite_path)
        outcome.candidate_result = result

        # Post-eval critics
        ctx_post = CriticContext(proposal=proposal, candidate_result=result)
        post_verdicts, post_blocked = _run_critics(post, ctx_post)
        outcome.verdicts.extend(post_verdicts)

        outcome.final = "rejected" if post_blocked else "approved"
        outcomes.append(outcome)

    return outcomes


def run_loop(
    proposals: list[Proposal],
    suite_path: Path | str,
    pre_critics: Optional[list[str]] = None,
    post_critics: Optional[list[str]] = None,
    policy: Optional[Policy] = None,
    auto_promote: bool = False,
    promote_strategy: str = "best",   # "best" | "all" | "none"
) -> list[LoopRound]:
    """Full pipeline including approver + executor.

    Steps:
      1. propose_and_score (Proposal → critics → branch → score → critics)
      2. For each outcome, ask the approver (mode in policies/auto_approve.yaml)
      3. If decision is AUTO_APPROVE *and* `auto_promote=True`, promote according
         to `promote_strategy`:
           - "best": only the single highest-Δoverall outcome promotes
           - "all":  every AUTO_APPROVE outcome promotes (sequential)
           - "none": don't promote even on AUTO_APPROVE (records-only run)

    Returns one LoopRound per input proposal.
    """
    if promote_strategy not in {"best", "all", "none"}:
        raise ValueError(f"promote_strategy must be best|all|none, got {promote_strategy!r}")

    outcomes = propose_and_score(proposals, suite_path, pre_critics, post_critics)
    pol = policy or Policy.from_yaml()

    decisions = [decide(o, pol) for o in outcomes]
    rounds = [LoopRound(outcome=o, decision=d) for o, d in zip(outcomes, decisions)]

    if not auto_promote or promote_strategy == "none":
        return rounds

    eligible_idxs = [
        i for i, r in enumerate(rounds) if r.decision == ApprovalDecision.AUTO_APPROVE
    ]
    if not eligible_idxs:
        return rounds

    if promote_strategy == "best":
        best_idx = max(
            eligible_idxs,
            key=lambda i: (
                rounds[i].outcome.candidate_result.delta["overall_score"]
                if rounds[i].outcome.candidate_result is not None
                else float("-inf")
            ),
        )
        version, lesson_id = promote(rounds[best_idx].outcome)
        rounds[best_idx].promoted_version = version
        rounds[best_idx].promoted_lesson_id = lesson_id
    else:  # "all" — sequential promotions
        for i in eligible_idxs:
            version, lesson_id = promote(rounds[i].outcome)
            rounds[i].promoted_version = version
            rounds[i].promoted_lesson_id = lesson_id

    return rounds
