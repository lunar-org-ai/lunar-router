"""CLI: `uv run python -m harness <subcommand>`.

Subcommands:
  sweep — heuristic sweep on a single knob; runs full Proposal→Critic→Score→Critic loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from harness.loop import DEFAULT_POST_CRITICS, DEFAULT_PRE_CRITICS, propose_and_score
from harness.proposer import sweep_knob


def _parse_values(raw: str) -> list[Any]:
    """Parse a comma-separated list of values; each parsed as JSON when possible."""
    items: list[Any] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            items.append(json.loads(tok))
        except (ValueError, TypeError):
            items.append(tok)
    return items


def cmd_sweep(args: argparse.Namespace) -> int:
    if ":" not in args.knob:
        print(f"--knob must be 'file:dotted.path' (got {args.knob!r})", file=sys.stderr)
        return 2
    file_part, path_part = args.knob.split(":", 1)
    values = _parse_values(args.values)
    if not values:
        print("--values produced no parseable items", file=sys.stderr)
        return 2

    proposals = sweep_knob(file_part.strip(), path_part.strip(), values)
    pre = args.pre_critics.split(",") if args.pre_critics else DEFAULT_PRE_CRITICS
    post = args.post_critics.split(",") if args.post_critics else DEFAULT_POST_CRITICS

    print(f"\nharness sweep: {args.knob} over {values}", file=sys.stderr)
    print(f"  pre-critics:  {pre}", file=sys.stderr)
    print(f"  post-critics: {post}", file=sys.stderr)
    print(f"  suite:        {args.suite}", file=sys.stderr)
    print(file=sys.stderr)

    outcomes = propose_and_score(proposals, args.suite, pre, post)

    print(f"{'value':<10} {'final':<10} {'cand_id':<32} {'Δoverall':>9}  {'verdicts':<60}", file=sys.stderr)
    print("-" * 130, file=sys.stderr)
    for o, p in zip(outcomes, proposals):
        v = p.mutations[0].value
        delta = (
            f"{o.candidate_result.delta['overall_score']:+.4f}"
            if o.candidate_result is not None
            else "    —    "
        )
        cand = o.candidate_id or "—"
        verdicts_short = "; ".join(
            f"{vd.critic}={'ok' if vd.approved else 'BLOCK'}" for vd in o.verdicts
        )
        print(f"{str(v):<10} {o.final:<10} {cand:<32} {delta:>9}  {verdicts_short:<60}", file=sys.stderr)
    print(file=sys.stderr)

    n_approved = sum(1 for o in outcomes if o.approved)
    print(f"approved {n_approved}/{len(outcomes)}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sweep = sub.add_parser("sweep", help="heuristic sweep on a single knob")
    p_sweep.add_argument("--knob", required=True, help="file:dotted.path (e.g. pipeline/retrieve.yaml:knobs.k)")
    p_sweep.add_argument("--values", required=True, help="comma-separated values to sweep (JSON-parsed)")
    p_sweep.add_argument("--suite", required=True, help="path to suite YAML")
    p_sweep.add_argument("--pre-critics", help="comma-separated PRE critics (default: scope)")
    p_sweep.add_argument("--post-critics", help="comma-separated POST critics (default: eval_lift)")
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
