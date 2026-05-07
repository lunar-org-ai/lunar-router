"""CLI: `uv run python -m evals.runners <suite>`.

Runs a suite against the current agent.yaml, prints a summary, and writes
the full report to evals/reports/.
"""

from __future__ import annotations

import argparse
import sys

from evals.runners.runner import run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.runners")
    parser.add_argument("suite", help="Path to a suite YAML (e.g. evals/suites/smoke_v0.yaml)")
    parser.add_argument(
        "--agent", default="agent/agent.yaml", help="Path to agent.yaml (default: agent/agent.yaml)"
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Skip writing the report to disk"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Exit with 1 if pass_rate is below this value (default: 0.0 = never fail)",
    )
    args = parser.parse_args(argv)

    report = run_suite(args.suite, agent_path=args.agent, write_report=not args.no_write)
    s = report.summary

    print(f"\n=== {report.suite}  (agent: {report.agent_version}) ===", file=sys.stderr)
    print(
        f"goldens={s['n_goldens']}  rubrics={s['n_rubrics']}  "
        f"checks_passed={s['n_passed']}/{s['n_total']}",
        file=sys.stderr,
    )
    print(
        f"overall_score={s['overall_score']:.3f}  pass_rate={s['pass_rate']:.3f}",
        file=sys.stderr,
    )
    print("\nper-rubric:", file=sys.stderr)
    for name, score in s["per_rubric"].items():
        bar = "█" * int(score * 20)
        print(f"  {name:<22}  {score:.3f}  {bar}", file=sys.stderr)
    print("\nper-case:", file=sys.stderr)
    for c in report.cases:
        passed = sum(1 for r in c.rubric_results if r.passed)
        total = len(c.rubric_results)
        marker = "✓" if passed == total else "✗"
        print(
            f"  {marker} {c.golden_id:<12} {passed}/{total}  {c.duration_ms:6.2f}ms  {c.request[:60]!r}",
            file=sys.stderr,
        )
    print(file=sys.stderr)

    if s["pass_rate"] < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
