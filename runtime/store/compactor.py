"""Compact a day's JSONL traces into Parquet partitioned by agent_version.

  traces/raw/YYYY-MM-DD.jsonl
       │
       └──► traces/parquet/dt=YYYY-MM-DD/agent_version=<v>/part-0.parquet

Snappy-compressed. Idempotent: writes to a tmp directory, then atomic
rename. Raw JSONL is **kept** in place — it's the audit trail and the
fallback if Parquet ever needs to be rebuilt (rm -rf parquet/ && replay).

Usage:
    python -m runtime.store.compactor                  # compacts yesterday
    python -m runtime.store.compactor 2026-05-07       # compacts a given day
    python -m runtime.store.compactor --all            # compacts every JSONL day
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "traces" / "raw"
PARQUET_DIR = ROOT / "traces" / "parquet"


def _is_iso_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def compact_day(day: str, *, force: bool = False) -> Path | None:
    """Compact one day. Returns the partition root if it wrote anything,
    None if there was nothing to compact."""
    if not _is_iso_date(day):
        raise ValueError(f"day must be YYYY-MM-DD, got {day!r}")

    src = RAW_DIR / f"{day}.jsonl"
    if not src.exists() or src.stat().st_size == 0:
        return None

    dst_root = PARQUET_DIR / f"dt={day}"
    if dst_root.exists() and not force:
        # Already compacted — skip. Caller can pass force=True to rebuild.
        return dst_root

    tmp_root = PARQUET_DIR / f"dt={day}.tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # DuckDB's COPY ... TO with PARTITION_BY writes one subdir per partition
    # value. We read the JSONL with union_by_name so older lines missing
    # session_id/history don't blow up.
    sql = f"""
    COPY (
      SELECT *,
             COALESCE(agent_version, 'unknown') AS _av
      FROM read_json_auto('{src.as_posix()}', union_by_name=true)
    ) TO '{tmp_root.as_posix()}'
    (FORMAT PARQUET, COMPRESSION SNAPPY,
     PARTITION_BY (_av), OVERWRITE_OR_IGNORE TRUE,
     FILENAME_PATTERN 'part-{{i}}')
    """
    con = duckdb.connect(":memory:")
    try:
        con.execute(sql)
    finally:
        con.close()

    # DuckDB writes partition_by subdirs as `_av=<value>/`. Rename to
    # `agent_version=<value>/` to keep the path human-friendly.
    for sub in tmp_root.glob("_av=*"):
        new_name = "agent_version=" + sub.name[len("_av=") :]
        sub.rename(sub.parent / new_name)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    tmp_root.rename(dst_root)
    return dst_root


def all_jsonl_days() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_DIR.glob("*.jsonl") if _is_iso_date(p.stem))


def yesterday_utc() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("day", nargs="?", help="YYYY-MM-DD (default: yesterday UTC)")
    p.add_argument("--all", action="store_true", help="compact every JSONL day")
    p.add_argument("--force", action="store_true", help="rebuild even if dt= exists")
    args = p.parse_args(argv)

    if args.all and args.day:
        p.error("--all is mutually exclusive with a specific day")

    if args.all:
        days = all_jsonl_days()
    else:
        days = [args.day or yesterday_utc()]

    if not days:
        print("nothing to compact (no JSONL files found)")
        return 0

    rc = 0
    for d in days:
        try:
            out = compact_day(d, force=args.force)
        except Exception as e:
            print(f"  {d} FAILED: {e}", file=sys.stderr)
            rc = 1
            continue
        if out is None:
            print(f"  {d} skipped (no source or empty)")
        else:
            n_parts = sum(1 for _ in out.rglob("*.parquet"))
            print(f"  {d} -> {out.relative_to(ROOT)}  ({n_parts} part file(s))")
    return rc


if __name__ == "__main__":
    sys.exit(main())
