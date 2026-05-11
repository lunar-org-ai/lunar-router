"""Adapter — surfaces prompts where the retrieve stage came up empty.

A trace where the ``retrieve`` stage produced ``docs_out == 0`` is a
gap in the RAG corpus: the prompt was something the agent doesn't know
how to look up. Those are exactly the candidates a curator wants to
review (either add docs that should match, or add the prompt as a
golden so the eval suite catches the regression).

Source label: ``"failed lookups"`` — matches the UI's source chip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from router.data.dataset import DatasetSample

from .base import build_sample, embed_list, prompt_hash


SOURCE_LABEL = "failed lookups"
_DEFAULT_TRACES_RAW = Path("traces") / "raw"


def iter_candidates(
    *,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    embedder,
    existing: Optional[set[str]] = None,
    traces_root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Iterator[DatasetSample]:
    """Yield DatasetSample candidates from traces where retrieve.docs_out == 0.

    Dedups by ``prompt_hash(prompt, tag)`` against ``existing`` (the
    target dataset's current sample IDs). Also dedups within the current
    call so repeated prompts in the trace stream only yield once.
    """
    seen = set(existing or ())
    root = traces_root or _DEFAULT_TRACES_RAW
    yielded = 0

    for row in _iter_raw_rows(root, since_iso, until_iso):
        prompt = (row.get("request") or "").strip()
        if not prompt:
            continue
        if not _retrieve_failed(row):
            continue
        tag = "failed_lookup"
        sid = prompt_hash(prompt, tag)
        if sid in seen:
            continue
        seen.add(sid)

        yield build_sample(
            prompt=prompt,
            tag=tag,
            trace_id=row.get("trace_id"),
            source=SOURCE_LABEL,
            embedding=embed_list(embedder, prompt),
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _retrieve_failed(row: dict) -> bool:
    """True iff any stage with technique='rag' (or stage name 'retrieve')
    produced docs_out == 0."""
    for stage in row.get("stages") or []:
        is_retrieve = (
            stage.get("stage") == "retrieve" or stage.get("technique") == "rag"
        )
        if not is_retrieve:
            continue
        docs_out = stage.get("docs_out")
        if docs_out is None:
            continue
        if int(docs_out) == 0:
            return True
    return False


def _iter_raw_rows(
    traces_root: Path,
    since_iso: Optional[str],
    until_iso: Optional[str],
) -> Iterator[dict]:
    """Stream raw trace rows from JSONL partitions, date-filtered."""
    if not traces_root.exists():
        return
    since_date = (since_iso or "")[:10] if since_iso else ""
    until_date = (until_iso or "")[:10] if until_iso else ""
    for path in sorted(traces_root.glob("*.jsonl")):
        date = path.stem
        if since_date and date < since_date:
            continue
        if until_date and date >= until_date:
            continue
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
