"""Run a candidate against a suite and record the delta vs baseline.

This is what makes the loop start to move: every candidate gets compared to
the live agent under the same suite, with the same goldens and rubrics, and
the result is appended (immutable) to experiments/results/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.runners.runner import run_suite
from experiments.branching import candidate_agent_path, list_candidates
from experiments.types import CandidateManifest

BASELINE_AGENT = Path("agent/agent.yaml")
RESULTS_DIR = Path("experiments/results")
EVAL_REPORTS_DIR = Path("evals/reports")


@dataclass
class CandidateResult:
    candidate_id: str
    suite: str
    parent_version: str
    mutations: list[str]
    recorded_at: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    delta: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _summary_view(summary: dict[str, Any]) -> dict[str, Any]:
    """Trim Report.summary to what we want pinned in the JSONL."""
    return {
        "overall_score": summary.get("overall_score"),
        "pass_rate": summary.get("pass_rate"),
        "per_rubric": dict(summary.get("per_rubric", {})),
        "n_passed": summary.get("n_passed"),
        "n_total": summary.get("n_total"),
    }


def _compute_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_score": round(candidate["overall_score"] - baseline["overall_score"], 4),
        "pass_rate": round(candidate["pass_rate"] - baseline["pass_rate"], 4),
        "per_rubric": {
            k: round(candidate["per_rubric"].get(k, 0.0) - baseline["per_rubric"].get(k, 0.0), 4)
            for k in baseline["per_rubric"].keys() | candidate["per_rubric"].keys()
        },
    }


def _load_manifest(candidate_id: str) -> CandidateManifest:
    for m in list_candidates():
        if m.id == candidate_id:
            return m
    raise FileNotFoundError(f"candidate not found: {candidate_id}")


def run_candidate(
    candidate_id: str,
    suite_path: Path | str,
    baseline_agent: Path | str = BASELINE_AGENT,
    results_dir: Path | str = RESULTS_DIR,
    write_result: bool = True,
) -> CandidateResult:
    """Run baseline + candidate against the same suite; record the delta."""
    manifest = _load_manifest(candidate_id)
    cand_yaml = candidate_agent_path(candidate_id)
    if not cand_yaml.exists():
        raise FileNotFoundError(f"candidate agent.yaml missing: {cand_yaml}")

    # Run baseline first so the candidate sees identical golden state.
    baseline_report = run_suite(suite_path, agent_path=baseline_agent, write_report=False)
    candidate_report = run_suite(suite_path, agent_path=cand_yaml, write_report=False)

    # Persist the candidate's full report so the UI Lesson view can recover the
    # trace lineage later (lesson.candidate_id → cand_<id>.json → cases[].trace_id).
    _persist_candidate_report(candidate_id, candidate_report)

    baseline_view = _summary_view(baseline_report.summary)
    candidate_view = _summary_view(candidate_report.summary)
    delta = _compute_delta(baseline_view, candidate_view)

    result = CandidateResult(
        candidate_id=candidate_id,
        suite=baseline_report.suite,
        parent_version=manifest.parent_version,
        mutations=[m.describe() for m in manifest.mutations],
        recorded_at=_now_iso(),
        baseline=baseline_view,
        candidate=candidate_view,
        delta=delta,
    )

    if write_result:
        _append_result(result, results_dir)

    return result


def _persist_candidate_report(
    candidate_id: str, report: Any, reports_dir: Path | str = EVAL_REPORTS_DIR
) -> Path:
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cand_{candidate_id}.json"
    with path.open("w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    return path


def _append_result(result: CandidateResult, results_dir: Path | str) -> Path:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"{today}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return path
