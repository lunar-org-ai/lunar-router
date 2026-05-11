"""Adapter from this repo's filesystem trace store to TraceRecord.

The reference impl reads ClickHouse rows. This repo's traces live as
JSONL partitions in ``traces/raw/<YYYY-MM-DD>.jsonl`` (also surfaced
through DuckDB in ``runtime/store/traces.py``). The adapter knows the
JSONL schema and **only that** — if the trace schema evolves, only this
file changes.

JSONL row shape (verified live during P15.3.4 design):
    {
      "trace_id": "...",
      "timestamp": "2026-05-09T22:20:45.911Z",
      "request": "<user prompt>",
      "response": "<agent response>",
      "duration_ms": 0.088,
      "stages": [
        { "stage": "retrieve", "routing_model": null, "error": null, ... },
        { "stage": "route", "routing_model": "claude-haiku-4-5", ... },
        { "stage": "generate", "routing_model": "claude-haiku-4-5", ... }
      ]
    }

Cold-start mode: when no fitted ``ClusterAssigner`` exists yet, callers
pass ``embedder=None, assigner=None``. The adapter then yields records
with ``cluster_id=-1`` (sentinel "unassigned"). Those records are useful
for *fitting* the first cluster set — but are filtered out of any Psi
update math by ``TraceToTraining.add_trace``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from router.core.clustering import ClusterAssigner
from router.core.embeddings import PromptEmbedder
from router.feedback.trace_to_training import TraceRecord


logger = logging.getLogger("router.feedback.store_adapter")


_TRACES_RAW = Path("traces") / "raw"


def iter_traces_since(
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    *,
    embedder: Optional[PromptEmbedder] = None,
    assigner: Optional[ClusterAssigner] = None,
    traces_root: Optional[Path] = None,
) -> Iterator[TraceRecord]:
    """Stream traces from JSONL partitions, yielding TraceRecord values.

    cluster_id is filled when both ``embedder`` and ``assigner`` are
    provided. Cold-start callers (no fitted assigner yet) pass
    ``embedder=None, assigner=None`` and get records with
    ``cluster_id=-1``. Such records are still useful for fitting clusters
    but NOT for Psi math (TraceToTraining filters cluster_id < 0).

    Args:
        since_iso: ISO-8601 lower bound (inclusive). Skips partition files
                   for dates strictly before this.
        until_iso: ISO-8601 upper bound (exclusive). Skips partition files
                   for dates on or after this.
        embedder: Optional PromptEmbedder. When supplied alongside
                  ``assigner``, each trace's prompt is embedded + assigned.
        assigner: Optional ClusterAssigner.
        traces_root: Override the default ``traces/raw`` location (used in
                     tests).
    """
    root = traces_root if traces_root is not None else _TRACES_RAW
    files = _select_partition_files(root, since_iso, until_iso)

    have_assignment = embedder is not None and assigner is not None

    for path in files:
        try:
            with path.open() as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.warning("skip malformed JSON line in %s", path.name)
                        continue
                    rec = _row_to_trace_record(row, embedder if have_assignment else None,
                                               assigner if have_assignment else None)
                    if rec is not None:
                        yield rec
        except OSError as e:
            logger.warning("could not read %s: %s", path, e)
            continue


def _select_partition_files(
    root: Path,
    since_iso: Optional[str],
    until_iso: Optional[str],
) -> list[Path]:
    """Pick JSONL partition files that overlap the [since, until) window.

    Partition naming: ``<YYYY-MM-DD>.jsonl``. We compare lexically because
    that's exactly date-ordered for ISO-8601.
    """
    if not root.exists():
        return []

    since_date = (since_iso or "")[:10] if since_iso else ""
    until_date = (until_iso or "")[:10] if until_iso else ""

    out: list[Path] = []
    for path in sorted(root.glob("*.jsonl")):
        date = path.stem  # "2026-05-09"
        if since_date and date < since_date:
            continue
        if until_date and date >= until_date:
            continue
        out.append(path)
    return out


def _row_to_trace_record(
    row: dict,
    embedder: Optional[PromptEmbedder],
    assigner: Optional[ClusterAssigner],
) -> Optional[TraceRecord]:
    """Convert one JSONL row → TraceRecord, or None when essential fields
    are missing (no prompt, or no model attribution from any stage).
    """
    request = row.get("request")
    if not request:
        return None  # nothing to embed.

    stages = row.get("stages") or []

    # selected_model: pull the last non-null routing_model from stages
    # (typically set by the route stage and propagated through generate).
    selected = None
    for s in reversed(stages):
        rm = s.get("routing_model") if isinstance(s, dict) else None
        if rm:
            selected = rm
            break
    if not selected:
        return None  # no model attribution → can't update Psi for any model.

    is_error = any(
        bool(s.get("error")) for s in stages if isinstance(s, dict)
    )
    error_category = _first_error_category(stages)

    cluster_id = -1
    if embedder is not None and assigner is not None:
        try:
            emb = embedder.embed(request)
            cluster_id = int(assigner.assign(emb).cluster_id)
        except Exception as e:  # pragma: no cover — defensive; embed should be cheap
            logger.warning("embed/assign failed for trace %s: %s", row.get("trace_id"), e)
            cluster_id = -1

    return TraceRecord(
        request_id=str(row.get("trace_id") or ""),
        selected_model=str(selected),
        cluster_id=cluster_id,
        is_error=is_error,
        latency_ms=float(row.get("duration_ms") or 0.0),
        # Token-level cost accounting isn't wired across the project yet —
        # leave 0.0 and let token-cost arithmetic enter when the runtime
        # records token usage per stage.
        total_cost_usd=0.0,
        input_text=request,
        output_text=row.get("response"),
        error_category=error_category,
        metadata={"timestamp": row.get("timestamp")},
    )


def _first_error_category(stages: list) -> Optional[str]:
    """Return the first non-null stage error string, or None."""
    for s in stages:
        if not isinstance(s, dict):
            continue
        err = s.get("error")
        if err:
            return str(err)
    return None
