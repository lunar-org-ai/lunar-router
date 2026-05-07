"""CLI: `uv run python -m experiments <subcommand>`.

Subcommands:
  create  — create a candidate from mutations, optionally score it
  run     — run an existing candidate against a suite
  list    — list all candidates
  show    — show a candidate's full manifest
"""

from __future__ import annotations

import argparse
import json
import sys

from experiments.branching import candidate_agent_path, create_candidate, list_candidates
from experiments.runner import CandidateResult, run_candidate
from experiments.types import Mutation


def _print_result(result: CandidateResult) -> None:
    print(f"\n=== {result.candidate_id} vs baseline ({result.parent_version}) ===", file=sys.stderr)
    print(f"  suite:        {result.suite}", file=sys.stderr)
    print(f"  mutations:    {result.mutations}", file=sys.stderr)
    print(
        f"  baseline:     overall={result.baseline['overall_score']:.3f}  "
        f"pass={result.baseline['pass_rate']:.3f}",
        file=sys.stderr,
    )
    print(
        f"  candidate:    overall={result.candidate['overall_score']:.3f}  "
        f"pass={result.candidate['pass_rate']:.3f}",
        file=sys.stderr,
    )
    print(
        f"  Δ overall:    {result.delta['overall_score']:+.4f}  "
        f"Δ pass: {result.delta['pass_rate']:+.4f}",
        file=sys.stderr,
    )
    print("  Δ per-rubric:", file=sys.stderr)
    for k, v in result.delta["per_rubric"].items():
        print(f"    {k:<22}  {v:+.4f}", file=sys.stderr)
    print(file=sys.stderr)


def cmd_create(args: argparse.Namespace) -> int:
    mutations = [Mutation.parse(s) for s in args.mutate]
    manifest = create_candidate(mutations, description=args.description)
    print(f"created candidate: {manifest.id}", file=sys.stderr)
    for m in manifest.mutations:
        print(f"  + {m.describe()}", file=sys.stderr)
    if args.suite:
        result = run_candidate(manifest.id, args.suite)
        _print_result(result)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    result = run_candidate(args.candidate_id, args.suite)
    _print_result(result)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    candidates = list_candidates()
    if not candidates:
        print("(no candidates yet)", file=sys.stderr)
        return 0
    print(f"{len(candidates)} candidate(s):", file=sys.stderr)
    for m in candidates:
        muts = ", ".join(x.describe() for x in m.mutations)
        desc = f"  — {m.description}" if m.description else ""
        print(f"  {m.id}  parent={m.parent_version}  {m.created_at}{desc}", file=sys.stderr)
        print(f"    {muts}", file=sys.stderr)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    for m in list_candidates():
        if m.id == args.candidate_id:
            print(
                json.dumps(
                    {
                        "id": m.id,
                        "parent_version": m.parent_version,
                        "created_at": m.created_at,
                        "description": m.description,
                        "mutations": [
                            {"file": x.file, "path": x.path, "value": x.value}
                            for x in m.mutations
                        ],
                        "agent_yaml_path": str(candidate_agent_path(m.id)),
                    },
                    indent=2,
                )
            )
            return 0
    print(f"candidate not found: {args.candidate_id}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="create a candidate from mutations")
    p_create.add_argument(
        "--mutate", action="append", required=True, help="file:path=value (repeatable)"
    )
    p_create.add_argument("--suite", help="if set, score the candidate against this suite")
    p_create.add_argument("--description", help="optional human description")
    p_create.set_defaults(func=cmd_create)

    p_run = sub.add_parser("run", help="run a candidate against a suite")
    p_run.add_argument("candidate_id")
    p_run.add_argument("--suite", required=True)
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list all candidates")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show a candidate's full manifest")
    p_show.add_argument("candidate_id")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
