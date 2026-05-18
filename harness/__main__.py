"""CLI: `uv run python -m harness <subcommand>`.

Subcommands:
  sweep — heuristic sweep on a single knob; runs full Proposal→Critic→Score→Critic loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from harness.approver import ApprovalDecision, Policy
from harness.loop import (
    DEFAULT_POST_CRITICS,
    DEFAULT_PRE_CRITICS,
    propose_and_score,
    run_loop,
)
from harness.proposer import sweep_knob
from harness.rollback import rollback_to
from ledger.versioning import list_snapshots, read_version


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

    policy = (
        Policy(mode=args.policy_mode, auto_min_lift=args.policy_min_lift)
        if args.policy_mode is not None
        else Policy.from_yaml()
    )

    print(f"\nharness sweep: {args.knob} over {values}", file=sys.stderr)
    print(f"  pre-critics:  {pre}", file=sys.stderr)
    print(f"  post-critics: {post}", file=sys.stderr)
    print(f"  policy:       mode={policy.mode}  min_lift={policy.auto_min_lift}", file=sys.stderr)
    print(f"  auto_promote: {args.auto_promote}  strategy={args.promote_strategy}", file=sys.stderr)
    print(f"  suite:        {args.suite}", file=sys.stderr)
    print(file=sys.stderr)

    rounds = run_loop(
        proposals,
        args.suite,
        pre_critics=pre,
        post_critics=post,
        policy=policy,
        auto_promote=args.auto_promote,
        promote_strategy=args.promote_strategy,
    )

    print(
        f"{'value':<10} {'final':<10} {'decision':<14} {'Δoverall':>9}  {'promoted':<10}  verdicts",
        file=sys.stderr,
    )
    print("-" * 110, file=sys.stderr)
    for r, p in zip(rounds, proposals):
        o = r.outcome
        v = p.mutations[0].value
        delta = (
            f"{o.candidate_result.delta['overall_score']:+.4f}"
            if o.candidate_result is not None
            else "    —    "
        )
        promoted = r.promoted_version or "—"
        verdicts_short = "; ".join(
            f"{vd.critic}={'ok' if vd.approved else 'BLOCK'}" for vd in o.verdicts
        )
        print(
            f"{str(v):<10} {o.final:<10} {r.decision.value:<14} {delta:>9}  {promoted:<10}  {verdicts_short}",
            file=sys.stderr,
        )
    print(file=sys.stderr)

    n_promoted = sum(1 for r in rounds if r.promoted_version)
    n_approved = sum(1 for r in rounds if r.outcome.approved)
    print(f"approved by critics: {n_approved}/{len(rounds)}  promoted: {n_promoted}", file=sys.stderr)
    if n_promoted:
        print(f"live agent now: {read_version()}", file=sys.stderr)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    snapshots = list_snapshots()
    if args.version not in snapshots:
        print(f"unknown version {args.version!r}", file=sys.stderr)
        print(f"available: {snapshots}", file=sys.stderr)
        return 1
    cur = read_version()
    if cur == args.version:
        print(f"already at {args.version}; no-op", file=sys.stderr)
        return 0
    rollback_to(args.version, reason=args.reason)
    print(f"rolled back: {cur} → {args.version}", file=sys.stderr)
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
    p_sweep.add_argument(
        "--auto-promote", action="store_true",
        help="auto-promote AUTO_APPROVE outcomes (default: off)",
    )
    p_sweep.add_argument(
        "--promote-strategy", choices=["best", "all", "none"], default="best",
        help="if --auto-promote, which outcomes get promoted (default: best)",
    )
    p_sweep.add_argument(
        "--policy-mode", choices=["auto", "review", "off"],
        help="override policies/auto_approve.yaml mode for this run",
    )
    p_sweep.add_argument(
        "--policy-min-lift", type=float, default=0.0,
        help="min Δoverall_score required for auto promotion (only if mode=auto)",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_rollback = sub.add_parser("rollback", help="restore live agent/ to a prior version")
    p_rollback.add_argument("version", help="version id (e.g. v0.0.1)")
    p_rollback.add_argument("--reason", default="manual rollback", help="reason recorded in the ledger")
    p_rollback.set_defaults(func=cmd_rollback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
