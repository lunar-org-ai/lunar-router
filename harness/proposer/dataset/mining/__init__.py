"""Mining adapters for the P15.4 dataset backend.

The proposer (P15.4.4) looks up an adapter by the dataset's ``source``
string (the same string the UI surfaces as the source chip). Each
adapter exposes ``iter_candidates(...)`` returning a stream of
``DatasetSample`` candidates.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

from router.data.dataset import DatasetSample

from . import failed_lookups, feedback_signals, flagged_traces, language_router


# (label, callable) — callable matches the iter_candidates signature.
_ADAPTERS: dict[str, Callable[..., Iterator[DatasetSample]]] = {
    flagged_traces.SOURCE_LABEL: flagged_traces.iter_candidates,
    language_router.SOURCE_LABEL: language_router.iter_candidates,
    failed_lookups.SOURCE_LABEL: failed_lookups.iter_candidates,
    feedback_signals.SOURCE_LABEL: feedback_signals.iter_candidates,
}


def get_adapter(source: str) -> Optional[Callable[..., Iterator[DatasetSample]]]:
    """Look up an adapter by its source label. Returns None when unknown.

    Callers should treat None as "no auto-mining for this source" — manual
    datasets (source='manual') always return None.
    """
    return _ADAPTERS.get(source)


def available_sources() -> list[str]:
    """All source labels the runtime knows how to mine for."""
    return list(_ADAPTERS.keys())


__all__ = [
    "get_adapter",
    "available_sources",
    "flagged_traces",
    "language_router",
    "failed_lookups",
    "feedback_signals",
]
