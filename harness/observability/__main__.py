"""CLI: `uv run python -m harness.observability <subcommand>`.

Subcommands:
  sessions          — distill every candidate into traces/distilled/sessions/
  day <YYYY-MM-DD>  — distill a single day into traces/distilled/epochs/
  version <vX.Y.Z>  — distill a single version
  all               — sessions + today + every snapshotted version
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from harness.observability.distillation import (
    distill_all_sessions,
    distill_day,
    distill_version,
)
from ledger.versioning import list_snapshots


def cmd_sessions(_: argparse.Namespace) -> int:
    sessions = distill_all_sessions()
    print(f"distilled {len(sessions)} sessions to traces/distilled/sessions/")
    return 0


def cmd_day(args: argparse.Namespace) -> int:
    epoch = distill_day(args.date)
    print(f"epoch_id={epoch.epoch_id}")
    print(f"  {epoch.summary}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    epoch = distill_version(args.version)
    print(f"epoch_id={epoch.epoch_id}")
    print(f"  {epoch.summary}")
    return 0


def cmd_all(_: argparse.Namespace) -> int:
    sessions = distill_all_sessions()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_epoch = distill_day(today)
    version_epochs = []
    for v in list_snapshots():
        try:
            version_epochs.append(distill_version(v))
        except Exception as e:
            print(f"  warn: version {v}: {e}", file=sys.stderr)
    print(
        f"distilled: {len(sessions)} sessions, 1 day epoch ({today}), "
        f"{len(version_epochs)} version epoch(s)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.observability")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sess = sub.add_parser("sessions", help="distill every candidate")
    p_sess.set_defaults(func=cmd_sessions)

    p_day = sub.add_parser("day", help="distill a single day")
    p_day.add_argument("date", help="YYYY-MM-DD")
    p_day.set_defaults(func=cmd_day)

    p_ver = sub.add_parser("version", help="distill a single version")
    p_ver.add_argument("version", help="e.g. v0.0.2")
    p_ver.set_defaults(func=cmd_version)

    p_all = sub.add_parser("all", help="distill everything")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
