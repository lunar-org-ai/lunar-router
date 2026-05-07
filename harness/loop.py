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

from experiments.branching import create_candidate
from experiments.runner import CandidateResult, run_candidate
from harness.critics import Critic, CriticStage, make_critic
from harness.types import CriticContext, CriticVerdict, LoopOutcome, Proposal

DEFAULT_PRE_CRITICS = ["scope"]
DEFAULT_POST_CRITICS = ["eval_lift"]


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
