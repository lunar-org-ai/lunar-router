"""Adapter — surfaces prompts from low-CSAT traces (P16.3).

P15.4.3 shipped this as a stub raising ``NotImplementedError`` because
no production feedback channel existed. P16.2 added the channel
(``POST /traces/{id}/feedback`` writing to
``traces/feedback/<date>.jsonl``). P16.3 wires the adapter to it.

A trace whose latest feedback score is ``≤ threshold`` (default 2) is
treated as a "frustrated customer" signal — exactly the kind of
candidate a curator wants to add to a dataset so the agent can be
evaluated against similar cases going forward.

Tag carries the score (e.g. ``"csat_2"``) so downstream filtering by
severity is easy.

Source label: ``"feedback signals"`` — matches the UI's source chip
exactly as the P15.4.2 modal exposes it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from router.data.dataset import DatasetSample
from runtime.store import feedback as feedback_store

from .base import build_sample, embed_list, prompt_hash


logger = logging.getLogger("harness.proposer.dataset.mining.feedback_signals")


SOURCE_LABEL = "feedback signals"
_DEFAULT_THRESHOLD = 2  # scores at-or-below this are mining candidates


def iter_candidates(
    *,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    embedder,
    existing: Optional[set[str]] = None,
    feedback_root: Optional[Path] = None,
    threshold: int = _DEFAULT_THRESHOLD,
    get_trace=None,
    limit: Optional[int] = None,
) -> Iterator[DatasetSample]:
    """Yield DatasetSample candidates from low-CSAT traces.

    Args:
        since_iso / until_iso: date window for feedback rows.
        embedder: warm PromptEmbedder.
        existing: target dataset's current sample IDs (for dedup).
        feedback_root: override traces/feedback (tests use this).
        threshold: max score that counts as "low" (default 2).
        get_trace: callable trace_id → trace dict, default lazy import
                   from runtime.store.traces. Tests inject a stub.
        limit: max candidates to emit.
    """
    seen = set(existing or ())
    if get_trace is None:
        from runtime.store.traces import get_trace as _real
        get_trace = _real
    yielded = 0

    # Track latest feedback row per trace_id; the adapter scores on the
    # latest rating, not the first.
    latest_by_trace: dict[str, dict] = {}
    for row in feedback_store.iter_feedback(since_iso=since_iso, root=feedback_root):
        if until_iso and (row.get("at") or "") >= until_iso:
            continue
        tid = row.get("trace_id")
        if not tid:
            continue
        prev = latest_by_trace.get(tid)
        if prev is None or (row.get("at") or "") > (prev.get("at") or ""):
            latest_by_trace[tid] = row

    for tid, row in latest_by_trace.items():
        try:
            score = int(row.get("score") or 0)
        except (TypeError, ValueError):
            continue
        if score > threshold:
            continue

        # Look up the trace to get the prompt + agent's response.
        try:
            trace = get_trace(tid)
        except Exception:
            trace = None
        if not trace:
            continue
        prompt = (trace.get("request") or "").strip()
        if not prompt:
            continue

        tag = f"csat_{score}"
        sid = prompt_hash(prompt, tag)
        if sid in seen:
            continue
        seen.add(sid)

        # ground_truth: the agent's actual response is NOT the correct
        # answer (user said it sucks). Leave empty so a curator can
        # provide the right one later — same shape as the failed_lookups
        # adapter.
        yield build_sample(
            prompt=prompt,
            tag=tag,
            trace_id=tid,
            source=SOURCE_LABEL,
            embedding=embed_list(embedder, prompt),
            ground_truth="",
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return
