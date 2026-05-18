"""Built-in critics for v0.

- ScopeCritic (pre)        : every mutation.file must match config/claude_code.yaml
                             `mutable` allowlist. The hard boundary that stops
                             the loop from touching the framework.
- EvalLiftCritic (post)    : candidate's Δoverall_score must be ≥ a threshold.
                             The quality gate.
- NoCriticalRegression (post): no rubric may regress below an absolute floor.
                               Avoids "won overall, broke something critical".
"""

from __future__ import annotations

import fnmatch
import posixpath
from pathlib import Path
from typing import Any, Optional

import yaml

from harness.critics.base import Critic, CriticStage, register_critic
from harness.types import CriticContext, CriticVerdict


@register_critic
class ScopeCritic(Critic):
    """Each mutation's file must match the `mutable` allowlist.

    Mutations carry agent/-relative file paths (e.g. "pipeline/retrieve.yaml").
    The allowlist patterns are repo-relative (e.g. "agent/**"). We prepend
    "agent/" before checking.

    Params:
      config_path: str = "config/claude_code.yaml"
    """

    name = "scope"
    stage = CriticStage.PRE

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(params)
        self._config_path = Path(self.params.get("config_path", "config/claude_code.yaml"))
        self._cache: Optional[dict[str, Any]] = None

    def _load(self) -> dict[str, Any]:
        if self._cache is None:
            with self._config_path.open() as f:
                self._cache = yaml.safe_load(f) or {}
        return self._cache

    def verdict(self, ctx: CriticContext) -> CriticVerdict:
        cfg = self._load()
        mutable: list[str] = cfg.get("mutable", [])
        if not mutable:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=f"{self._config_path}: no `mutable` patterns; refusing all mutations",
                severity="block",
            )

        violations: list[str] = []
        for m in ctx.proposal.mutations:
            # Mutations are agent/-relative; canonicalize so `..` traversal
            # doesn't sneak past fnmatch (which is pure string matching).
            effective = posixpath.normpath(f"agent/{m.file}")
            if effective.startswith(".."):
                violations.append(f"{m.file} (escapes repo root via ..)")
                continue
            if not any(fnmatch.fnmatch(effective, pat) for pat in mutable):
                violations.append(effective)

        if violations:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=f"scope violations: {violations}",
                severity="block",
            )
        return CriticVerdict(
            critic=self.name,
            approved=True,
            reason=f"{len(ctx.proposal.mutations)} mutation(s) within allowlist",
        )


@register_critic
class EvalLiftCritic(Critic):
    """Candidate must score better than baseline by at least `min_delta`.

    Default min_delta=0.0 means "non-regressing" passes; >0 requires real lift.

    Params:
      min_delta: float = 0.0
    """

    name = "eval_lift"
    stage = CriticStage.POST

    def verdict(self, ctx: CriticContext) -> CriticVerdict:
        if ctx.candidate_result is None:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason="no candidate_result available (post-stage critic needs scored candidate)",
                severity="block",
            )

        min_delta = float(self.params.get("min_delta", 0.0))
        delta = float(ctx.candidate_result.delta["overall_score"])

        if delta >= min_delta:
            return CriticVerdict(
                critic=self.name,
                approved=True,
                reason=f"Δoverall={delta:+.4f} >= {min_delta:+.4f}",
            )
        return CriticVerdict(
            critic=self.name,
            approved=False,
            reason=f"Δoverall={delta:+.4f} < {min_delta:+.4f}",
            severity="block",
        )


@register_critic
class NoCriticalRegression(Critic):
    """No per-rubric score may drop below `floor`.

    Catches "won on average but tanked a key rubric" — a Goodhart guard.

    Params:
      floor: float = 0.0     # candidate per-rubric score must be >= floor
      rubrics: list[str] = []  # if non-empty, only check these rubric names
    """

    name = "no_critical_regression"
    stage = CriticStage.POST

    def verdict(self, ctx: CriticContext) -> CriticVerdict:
        if ctx.candidate_result is None:
            return CriticVerdict(
                critic=self.name, approved=False,
                reason="no candidate_result available", severity="block",
            )

        floor = float(self.params.get("floor", 0.0))
        watched: list[str] = list(self.params.get("rubrics", []))
        per_rubric: dict[str, float] = ctx.candidate_result.candidate["per_rubric"]

        offenders: list[str] = []
        for name, score in per_rubric.items():
            if watched and name not in watched:
                continue
            if score < floor:
                offenders.append(f"{name}={score:.3f}")

        if offenders:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=f"rubrics below floor {floor}: {offenders}",
                severity="block",
            )
        return CriticVerdict(
            critic=self.name,
            approved=True,
            reason=f"all rubrics ≥ floor {floor}",
        )
