"""Approver: policy-driven gate between critic-approved and live promotion.

Critics already filtered out scope violations and regressions. The approver
decides whether an outcome that *passed* the critics should auto-promote, get
queued for human review, or be hard-rejected by policy.

policies/auto_approve.yaml controls behavior. Conservative default: review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from harness.types import LoopOutcome

DEFAULT_POLICY_PATH = Path("policies/auto_approve.yaml")


class ApprovalDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    QUEUE_HUMAN = "queue_human"
    REJECT = "reject"


@dataclass
class Policy:
    mode: str = "review"          # auto | review | off
    auto_min_lift: float = 0.01

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_POLICY_PATH) -> "Policy":
        p = Path(path)
        if not p.exists():
            return cls()  # safe defaults
        with p.open() as f:
            d = yaml.safe_load(f) or {}
        return cls(
            mode=d.get("mode", "review"),
            auto_min_lift=float(d.get("thresholds", {}).get("auto_min_lift", 0.01)),
        )


def decide(outcome: LoopOutcome, policy: Policy) -> ApprovalDecision:
    """Apply policy to one critic-approved outcome.

    Outcomes that didn't pass critics are rejected outright.
    """
    if not outcome.approved:
        return ApprovalDecision.REJECT

    if policy.mode == "off":
        return ApprovalDecision.REJECT

    if policy.mode == "review":
        return ApprovalDecision.QUEUE_HUMAN

    if policy.mode == "auto":
        if outcome.candidate_result is None:
            return ApprovalDecision.QUEUE_HUMAN  # no proof → ask human
        delta = float(outcome.candidate_result.delta["overall_score"])
        if delta >= policy.auto_min_lift:
            return ApprovalDecision.AUTO_APPROVE
        return ApprovalDecision.QUEUE_HUMAN  # passed critics but not enough lift

    raise ValueError(f"unknown policy mode {policy.mode!r}; expected auto|review|off")
