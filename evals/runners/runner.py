"""Eval runner — runs a suite end-to-end and produces a Report.

Per golden: compiles the agent, runs the request, scores every rubric, and
also persists the trace (so the eval result is causally linked to the trace
in traces/raw/). Aggregates across rubrics into summary stats.

The runner is the load-bearing seam for the harness loop: any candidate
agent config can be scored by pointing run_suite at it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from evals.loader import load_suite, resolve_goldens
from evals.rubrics import EvalContext, make_rubric
from evals.types import EvalCase, Report, RubricResult
from runtime.compiler.builder import compile_agent
from runtime.compiler.loader import load_agent
from runtime.executor.pipeline import PipelineExecutor
from runtime.executor.tracing import write_trace
from runtime.protocols import Message

DEFAULT_AGENT = "agent/agent.yaml"
DEFAULT_REPORTS_DIR = Path("evals/reports")


def run_suite(
    suite_path: Path | str,
    agent_path: Path | str = DEFAULT_AGENT,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    write_report: bool = True,
) -> Report:
    """Load suite + agent, run all goldens, score, optionally write the report."""
    suite = load_suite(suite_path)
    goldens = resolve_goldens(suite)
    rubrics = [make_rubric(spec) for spec in suite.rubrics]

    cfg = load_agent(agent_path)
    pipeline = compile_agent(cfg)
    executor = PipelineExecutor(pipeline)

    started_at = Report.now_iso()
    cases: list[EvalCase] = []

    for golden in goldens:
        history = [Message(role=m.role, content=m.content) for m in golden.input.history]
        _, exec_record = executor.run(golden.input.request, history=history)
        trace_id = write_trace(exec_record)

        ctx = EvalContext(
            golden=golden,
            response=exec_record.response,
            duration_ms=exec_record.duration_ms,
            success=exec_record.success,
            error=exec_record.error,
        )
        rubric_results: list[RubricResult] = [r.score(ctx) for r in rubrics]
        cases.append(
            EvalCase(
                golden_id=golden.id,
                request=golden.input.request,
                response=exec_record.response,
                duration_ms=exec_record.duration_ms,
                success=exec_record.success,
                error=exec_record.error,
                trace_id=trace_id,
                rubric_results=rubric_results,
            )
        )

    finished_at = Report.now_iso()
    summary = _aggregate(cases, suite.aggregation)

    report = Report(
        suite=suite.suite,
        agent_version=cfg.version,
        started_at=started_at,
        finished_at=finished_at,
        cases=cases,
        summary=summary,
    )

    if write_report:
        _write_report(report, reports_dir)

    return report


def _aggregate(cases: list[EvalCase], method: str) -> dict[str, Any]:
    by_rubric: dict[str, list[float]] = {}
    pass_count = 0
    total_checks = 0

    for case in cases:
        for r in case.rubric_results:
            by_rubric.setdefault(r.rubric, []).append(r.score)
            total_checks += 1
            if r.passed:
                pass_count += 1

    if method != "mean":
        raise ValueError(f"unsupported aggregation: {method!r} (v0 supports 'mean')")

    per_rubric = {name: sum(scores) / len(scores) for name, scores in by_rubric.items()}
    overall_score = sum(per_rubric.values()) / len(per_rubric) if per_rubric else 0.0
    pass_rate = pass_count / total_checks if total_checks else 0.0

    return {
        "overall_score": round(overall_score, 4),
        "pass_rate": round(pass_rate, 4),
        "per_rubric": {k: round(v, 4) for k, v in per_rubric.items()},
        "n_goldens": len(cases),
        "n_rubrics": len({r.rubric for case in cases for r in case.rubric_results}),
        "n_passed": pass_count,
        "n_total": total_checks,
    }


def _write_report(report: Report, reports_dir: Path | str) -> Path:
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = (
        report.started_at.replace(":", "").replace(".", "").replace("-", "")
    )
    path = out_dir / f"{report.suite}_{ts_safe}.json"
    with path.open("w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    return path
