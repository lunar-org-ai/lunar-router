"""Adapter — STUB. No production feedback channel exists yet.

The UI surfaces ``"feedback signals"`` as a dataset source, but the
runtime currently has no channel collecting user thumbs-up / thumbs-down /
inline-correction signals. Until one lands, this adapter raises
``NotImplementedError`` so callers (the proposer) can downgrade to
another source rather than silently producing zero candidates.

When the feedback channel ships, replace this body with a real iterator
that filters traces where ``metadata.feedback`` is ``"negative"`` or
``feedback_score < 0`` — same interface as the other adapters.

Source label: ``"feedback signals"`` — matches the UI's source chip.
"""

from __future__ import annotations

from typing import Iterator, Optional

from router.data.dataset import DatasetSample


SOURCE_LABEL = "feedback signals"


def iter_candidates(
    *,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
    embedder=None,
    existing: Optional[set[str]] = None,
    **kwargs,
) -> Iterator[DatasetSample]:
    raise NotImplementedError(
        "feedback_signals mining requires a production feedback channel; "
        "no signal source is wired yet. Use 'flagged traces', "
        "'language router', or 'failed lookups' instead."
    )
    # Unreachable — yield to make this a generator at the type-checker level.
    yield  # pragma: no cover
