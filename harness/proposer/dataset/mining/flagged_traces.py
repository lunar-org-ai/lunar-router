"""Adapter — surfaces traces an operator flagged as worth keeping.

Two sources, in order:
  1. ``evals/golden/*.yaml`` where ``metadata.source`` starts with
     ``trace:`` (P16.1's "Promote to golden" flow drops these here).
  2. ``traces/pinned/*.jsonl`` — reserved for a future flag-trace endpoint;
     when populated, each line is read as a raw trace row.

Both surfaces are scanned with dedup against the target dataset's
existing IDs. Output samples carry ``tag = expected.category`` when
available, else ``"flagged"``.

Source label: ``"flagged traces"`` — matches the UI's source chip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import yaml

from router.data.dataset import DatasetSample

from .base import build_sample, embed_list, prompt_hash


SOURCE_LABEL = "flagged traces"
_DEFAULT_GOLDENS = Path("evals") / "golden"
_DEFAULT_PINNED = Path("traces") / "pinned"


def iter_candidates(
    *,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    embedder,
    existing: Optional[set[str]] = None,
    goldens_dir: Optional[Path] = None,
    pinned_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Iterator[DatasetSample]:
    """Yield samples from goldens-with-trace-source then pinned traces."""
    seen = set(existing or ())
    yielded = 0

    # 1. Goldens promoted from traces
    for sample in _iter_promoted_goldens(
        goldens_dir or _DEFAULT_GOLDENS,
        embedder=embedder,
        seen=seen,
    ):
        yield sample
        yielded += 1
        if limit is not None and yielded >= limit:
            return

    # 2. Pinned trace rows (when populated)
    for sample in _iter_pinned(
        pinned_dir or _DEFAULT_PINNED,
        since_iso=since_iso,
        until_iso=until_iso,
        embedder=embedder,
        seen=seen,
    ):
        yield sample
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _iter_promoted_goldens(
    goldens_dir: Path,
    *,
    embedder,
    seen: set[str],
) -> Iterator[DatasetSample]:
    if not goldens_dir.exists():
        return
    for path in sorted(goldens_dir.glob("*.yaml")):
        try:
            with path.open() as f:
                doc = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue

        source = ((doc.get("metadata") or {}).get("source") or "")
        if not source.startswith("trace:"):
            continue

        prompt = ((doc.get("input") or {}).get("request") or "").strip()
        if not prompt:
            continue

        expected = doc.get("expected") or {}
        tag = expected.get("category") or "flagged"
        ground_truth = expected.get("exact") or (
            (expected.get("contains") or [""])[0]
        )

        sid = prompt_hash(prompt, tag)
        if sid in seen:
            continue
        seen.add(sid)

        trace_id = source[len("trace:"):] or None
        yield build_sample(
            prompt=prompt,
            tag=tag,
            trace_id=trace_id,
            source=SOURCE_LABEL,
            embedding=embed_list(embedder, prompt),
            ground_truth=str(ground_truth or ""),
        )


def _iter_pinned(
    pinned_dir: Path,
    *,
    since_iso: Optional[str],
    until_iso: Optional[str],
    embedder,
    seen: set[str],
) -> Iterator[DatasetSample]:
    if not pinned_dir.exists():
        return
    since_date = (since_iso or "")[:10] if since_iso else ""
    until_date = (until_iso or "")[:10] if until_iso else ""
    for path in sorted(pinned_dir.glob("*.jsonl")):
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
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = (row.get("request") or row.get("prompt") or "").strip()
                    if not prompt:
                        continue
                    tag = (row.get("flag_reason") or "flagged")
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
        except OSError:
            continue
