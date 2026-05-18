"""Approver: policy-driven gate between critic-approved and live promotion.

Critics already filtered out scope violations and regressions. The approver
decides whether an outcome that *passed* the critics should auto-promote, get
queued for human review, or be hard-rejected by policy.

policies/auto_approve.yaml controls behavior. Conservative default: review.

Per-kind overrides
------------------
The YAML may include an `overrides:` map (kind → mode) so different change
kinds can have different ship policies (e.g. prompt edits go auto but tool
wrappers always require review). The approver picks the kind from the
candidate's mutations via `kind_from_mutations()` and consults overrides
before falling back to the global mode.

Auto-rollback
-------------
The YAML carries thresholds for production-metric-triggered rollback
(csat_drop, resolution_drop, window_hours, notify_channels). The values are
persisted and exposed to the UI, but the actual rollback runner that watches
production telemetry doesn't exist yet — that lands when real production
metrics flow through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from harness.types import LoopOutcome, kind_from_mutations

DEFAULT_POLICY_PATH = Path("policies/auto_approve.yaml")
VALID_MODES = ("auto", "review", "off")


class ApprovalDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    QUEUE_HUMAN = "queue_human"
    REJECT = "reject"


@dataclass
class AutoRollback:
    """Thresholds for metric-triggered rollback. Watched in production once
    telemetry is wired; persisted in YAML today, not yet acted on."""

    csat_drop: float = 0.3                  # absolute drop on a 0..5 scale
    resolution_drop: float = 0.05           # 5pp drop in resolution rate
    window_hours: int = 24
    notify_channels: list[str] = field(default_factory=lambda: ["email"])


@dataclass
class Policy:
    mode: str = "review"                    # auto | review | off (global default)
    auto_min_lift: float = 0.01
    overrides: dict[str, str] = field(default_factory=dict)  # kind → mode
    auto_rollback: AutoRollback = field(default_factory=AutoRollback)

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_POLICY_PATH) -> "Policy":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open() as f:
            d = yaml.safe_load(f) or {}

        overrides_raw = d.get("overrides") or {}
        overrides: dict[str, str] = {}
        if isinstance(overrides_raw, dict):
            for k, v in overrides_raw.items():
                if isinstance(v, str) and v in VALID_MODES:
                    overrides[str(k)] = v

        ar_raw = d.get("auto_rollback") or {}
        notify = ar_raw.get("notify_channels") or ["email"]
        if not isinstance(notify, list):
            notify = ["email"]
        auto_rollback = AutoRollback(
            csat_drop=float(ar_raw.get("csat_drop", 0.3)),
            resolution_drop=float(ar_raw.get("resolution_drop", 0.05)),
            window_hours=int(ar_raw.get("window_hours", 24)),
            notify_channels=[str(c) for c in notify],
        )

        return cls(
            mode=d.get("mode", "review"),
            auto_min_lift=float(d.get("thresholds", {}).get("auto_min_lift", 0.01)),
            overrides=overrides,
            auto_rollback=auto_rollback,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the YAML structure the loader expects. Round-trips."""
        return {
            "mode": self.mode,
            "thresholds": {"auto_min_lift": self.auto_min_lift},
            "overrides": dict(self.overrides),
            "auto_rollback": {
                "csat_drop": self.auto_rollback.csat_drop,
                "resolution_drop": self.auto_rollback.resolution_drop,
                "window_hours": self.auto_rollback.window_hours,
                "notify_channels": list(self.auto_rollback.notify_channels),
            },
        }

    def write_yaml(self, path: Path | str = DEFAULT_POLICY_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def mode_for(self, kind: str) -> str:
        """Effective mode for a given change kind: override → global."""
        return self.overrides.get(kind, self.mode)


def decide(outcome: LoopOutcome, policy: Policy) -> ApprovalDecision:
    """Apply policy to one critic-approved outcome.

    Outcomes that didn't pass critics are rejected outright. Otherwise the
    per-kind override wins; only if absent does the global mode apply.
    """
    if not outcome.approved:
        return ApprovalDecision.REJECT

    mutations = [m.describe() for m in outcome.proposal.mutations]
    kind = kind_from_mutations(mutations) if mutations else "other"
    effective_mode = policy.mode_for(kind)

    if effective_mode == "off":
        return ApprovalDecision.REJECT

    if effective_mode == "review":
        return ApprovalDecision.QUEUE_HUMAN

    if effective_mode == "auto":
        if outcome.candidate_result is None:
            return ApprovalDecision.QUEUE_HUMAN
        delta = float(outcome.candidate_result.delta["overall_score"])
        if delta >= policy.auto_min_lift:
            return ApprovalDecision.AUTO_APPROVE
        return ApprovalDecision.QUEUE_HUMAN

    raise ValueError(
        f"unknown policy mode {effective_mode!r}; expected auto|review|off"
    )
