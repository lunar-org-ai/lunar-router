"""Trace writer — persists ExecutionRecord as JSONL.

Format: one JSON object per line in traces/raw/<YYYY-MM-DD>.jsonl. Each line is
an envelope (trace_id, timestamp) wrapping the ExecutionRecord. This schema is
the input to evals (read these to score the agent), to the ledger (causal
chains start here), and to the harness proposer (failed traces become
candidates for proposals).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime.executor.pipeline import ExecutionRecord

TRACES_DIR = Path("traces/raw")


def _new_trace_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def envelope(
    record: ExecutionRecord,
    trace_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the on-disk dict from a record + optional ids."""
    return {
        "trace_id": trace_id or _new_trace_id(),
        "timestamp": timestamp or _now_iso(),
        **record.to_dict(),
    }


def write_trace(
    record: ExecutionRecord,
    traces_dir: Path | str = TRACES_DIR,
    trace_id: str | None = None,
) -> str:
    """Append the trace as JSONL. Returns the trace_id."""
    traces_dir = Path(traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    env = envelope(record, trace_id=trace_id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = traces_dir / f"{today}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(env, ensure_ascii=False) + "\n")
    return env["trace_id"]
