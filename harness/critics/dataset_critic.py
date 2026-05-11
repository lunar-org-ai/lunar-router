"""Dataset quality-gate critic (P15.4.4).

Reads the candidate payload off the Proposal's first Mutation and gates
``kind="dataset"`` proposals on three independent checks:

  1. Schema: every new sample has the required keys
     (id, prompt, embedding, tag).
  2. Dedup: no duplicate IDs in the merged sample list.
  3. Coverage: ``gap_score_after`` is no higher than
     ``gap_score_before + epsilon`` — the candidate may not widen
     cluster gaps.

Optional (off by default): suite re-run. When ``params['run_suites']``
is True, the critic reverse-looks-up suites that name this dataset and
runs each against an agent (``params['agent_path']``). It checks no
rubric regresses below ``min_rubric_score`` (default 0.0). This is
expensive — wake-up callers typically leave it disabled and rely on
the cheap checks above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from harness.critics.base import Critic, CriticStage, register_critic
from harness.types import CriticContext, CriticVerdict


logger = logging.getLogger("harness.critics.dataset_critic")


@dataclass
class DatasetQualityGate:
    """Defaults — override via critic params."""

    coverage_epsilon: float = 0.01    # gap_score_after may rise by this much
    max_total_samples: int = 10_000   # hard cap on dataset size
    min_rubric_score: float = 0.0     # for optional suite re-run


_REQUIRED_SAMPLE_KEYS = {"id", "prompt", "embedding", "tag"}


@register_critic
class DatasetCritic(Critic):
    """Quality-gate critic for ``kind="dataset"`` proposals.

    Params (all optional):
      coverage_epsilon: float = 0.01
      max_total_samples: int = 10000
      min_rubric_score: float = 0.0
      run_suites: bool = False
      agent_path: str — required when run_suites=True
      suites_dir: str — defaults to evals/suites
    """

    name = "dataset_quality_gate"
    stage = CriticStage.POST

    def verdict(self, ctx: CriticContext) -> CriticVerdict:
        gate = self._gate_from_params()

        try:
            payload = self._extract_payload(ctx)
        except ValueError as e:
            return CriticVerdict(
                critic=self.name, approved=False,
                reason=str(e), severity="block",
            )

        # 1. Schema check on every sample
        samples = payload.get("samples") or []
        schema_err = self._validate_samples(samples)
        if schema_err is not None:
            return CriticVerdict(
                critic=self.name, approved=False,
                reason=f"schema_invalid: {schema_err}", severity="block",
            )

        # 2. Dedup check
        ids = [s["id"] for s in samples]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            return CriticVerdict(
                critic=self.name, approved=False,
                reason=f"duplicate_ids: {duplicates[:5]}", severity="block",
            )

        # 3. Cap check
        if len(samples) > gate.max_total_samples:
            return CriticVerdict(
                critic=self.name, approved=False,
                reason=(
                    f"size_cap_exceeded: {len(samples)} > "
                    f"{gate.max_total_samples}"
                ),
                severity="block",
            )

        # 4. Coverage gap check (proposer attached scores in metadata)
        meta = payload.get("metadata") or {}
        gap_before = meta.get("gap_score_before")
        gap_after = meta.get("gap_score_after")
        if gap_before is not None and gap_after is not None:
            if gap_after > gap_before + gate.coverage_epsilon:
                return CriticVerdict(
                    critic=self.name, approved=False,
                    reason=(
                        f"coverage_widened: gap_score "
                        f"{gap_before:.3f} → {gap_after:.3f} "
                        f"(epsilon={gate.coverage_epsilon})"
                    ),
                    severity="block",
                )

        # 5. Optional: re-run affected suites against an agent
        if bool(self.params.get("run_suites", False)):
            regression = self._suite_regression(payload, gate)
            if regression is not None:
                return CriticVerdict(
                    critic=self.name, approved=False,
                    reason=f"suite_regression: {regression}",
                    severity="block",
                )

        added = int(meta.get("added", 0))
        gap_text = ""
        if gap_before is not None and gap_after is not None:
            gap_text = f", gap_score {gap_before:.3f} → {gap_after:.3f}"
        return CriticVerdict(
            critic=self.name, approved=True,
            reason=f"added={added}, size={len(samples)}{gap_text}",
            severity="info",
        )

    # ------------------------------------------------------------------

    def _gate_from_params(self) -> DatasetQualityGate:
        return DatasetQualityGate(
            coverage_epsilon=float(self.params.get(
                "coverage_epsilon", DatasetQualityGate.coverage_epsilon
            )),
            max_total_samples=int(self.params.get(
                "max_total_samples", DatasetQualityGate.max_total_samples
            )),
            min_rubric_score=float(self.params.get(
                "min_rubric_score", DatasetQualityGate.min_rubric_score
            )),
        )

    def _extract_payload(self, ctx: CriticContext) -> dict:
        if not ctx.proposal.mutations:
            raise ValueError("proposal has no mutations")
        value = ctx.proposal.mutations[0].value
        if not isinstance(value, dict):
            raise ValueError(
                f"expected dict payload on Mutation.value, got {type(value).__name__}"
            )
        for k in ("version", "name", "samples"):
            if k not in value:
                raise ValueError(f"payload missing required key {k!r}")
        return value

    def _validate_samples(self, samples: list[dict]) -> Optional[str]:
        for i, s in enumerate(samples):
            if not isinstance(s, dict):
                return f"sample[{i}] is not a dict"
            missing = _REQUIRED_SAMPLE_KEYS - set(s.keys())
            if missing:
                return f"sample[{i}] missing keys: {sorted(missing)}"
            if not isinstance(s.get("prompt"), str) or not s["prompt"].strip():
                return f"sample[{i}] has empty prompt"
            emb = s.get("embedding")
            if not isinstance(emb, list) or len(emb) == 0:
                return f"sample[{i}] has invalid embedding"
        return None

    def _suite_regression(
        self,
        payload: dict,
        gate: DatasetQualityGate,
    ) -> Optional[str]:
        """Return a human reason when any suite regresses; None on pass.

        This is the expensive path. Operators turn it on for full validation;
        the wakeup loop typically leaves it off.
        """
        from pathlib import Path

        name = payload["name"]
        try:
            from evals.loader import find_suites_for_dataset
        except ImportError:
            return None  # evals package not available; skip
        suites_dir = self.params.get("suites_dir")
        suite_names = find_suites_for_dataset(
            name,
            suites_dir=Path(suites_dir) if suites_dir else Path("evals/suites"),
        )
        if not suite_names:
            return None  # no suites point at this dataset

        agent_path = self.params.get("agent_path")
        if not agent_path:
            return "run_suites=True but no agent_path provided"

        try:
            from evals.runners.runner import run_suite
        except ImportError:
            return None

        suites_root = Path(suites_dir) if suites_dir else Path("evals/suites")
        floor = gate.min_rubric_score
        for sn in suite_names:
            try:
                report = run_suite(
                    suites_root / f"{sn}.yaml",
                    agent_path=agent_path,
                    write_report=False,
                )
            except Exception as e:  # pragma: no cover — best effort
                return f"suite {sn!r} failed to run: {type(e).__name__}: {e}"
            for case in report.cases:
                for rr in case.rubric_results:
                    if rr.score < floor:
                        return (
                            f"suite={sn} golden={case.golden_id} "
                            f"rubric={rr.rubric} score={rr.score:.3f} < floor={floor}"
                        )
        return None
