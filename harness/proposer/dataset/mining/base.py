"""Common interface for dataset mining adapters (P15.4.3).

Every adapter implements ``iter_candidates(...)`` yielding ``DatasetSample``
records ready to land in a dataset. The shape mirrors the storage layer
from P15.4.1 (`router.data.dataset.DatasetSample`).

Adapters embed prompts on-the-fly via the warm ``PromptEmbedder`` singleton
from ``runtime.embedder_pool`` — same model the migration script uses, so
sample IDs collide reliably across sources (manual / auto / migrated).

The common ``_dedup`` helper rejects samples whose prompt+tag hash already
exists in the target dataset's current version. This is what lets the
proposer (P15.4.4) call adapters incrementally without re-adding samples
that were already curated.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterator, Optional, Protocol

from router.data.dataset import Dataset, DatasetSample


def prompt_hash(prompt: str, tag: Optional[str]) -> str:
    """Stable short hash. Mirrors tools/migrate_goldens_to_dataset._sample_id.

    Same prompt + same tag → same hash → same DatasetSample.id, regardless
    of which adapter produced it. That's how dedup works across sources.
    """
    key = f"{prompt}\0{tag or ''}".encode("utf-8")
    return f"smp_{hashlib.sha256(key).hexdigest()[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_sample(
    *,
    prompt: str,
    tag: Optional[str],
    trace_id: Optional[str],
    source: str,
    embedding: list[float],
    ground_truth: str = "",
    added_at: Optional[str] = None,
) -> DatasetSample:
    """Construct a DatasetSample with the canonical hashed ID."""
    return DatasetSample(
        id=prompt_hash(prompt, tag),
        prompt=prompt,
        ground_truth=ground_truth,
        tag=tag,
        trace_id=trace_id,
        added_at=added_at or now_iso(),
        source=source,
        embedding=list(embedding),
    )


def existing_ids(dataset: Optional[Dataset]) -> set[str]:
    """Sample IDs already in a dataset — used by adapters to short-circuit dedup."""
    if dataset is None:
        return set()
    return {s.id for s in dataset.samples}


class MiningAdapter(Protocol):
    """Adapter interface (structural — adapters are plain modules).

    Implementations expose a single ``iter_candidates`` function with the
    same shape so the proposer can pick one by ``source`` string at runtime.
    """

    source_label: str

    def iter_candidates(
        self,
        *,
        since_iso: Optional[str] = None,
        until_iso: Optional[str] = None,
        embedder,
        existing: Optional[set[str]] = None,
    ) -> Iterator[DatasetSample]: ...


def embed_list(embedder, prompt: str) -> list[float]:
    """Embed a prompt and return a plain list[float] suitable for JSON storage."""
    vec = embedder.embed(prompt)
    try:
        return vec.tolist()  # numpy
    except AttributeError:
        return list(vec)
