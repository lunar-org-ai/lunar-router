"""Online incremental Psi update — DEFERRED.

This module is scaffolded so harness/proposer/router_proposer.py
(P15.3.7) can reference the import path without breaking. Real
implementation lands only if P15.3.7 hits eval-cost issues that
incremental updates would solve. See ROADMAP_P15.3.md.

The intended shape: each new ProductionPsiUpdate slides into the
existing LLMProfile via an exponential moving average so we can refresh
Psi without re-running the full eval suite. For v1 we always run the
full pipeline.
"""

from router.feedback.trace_to_training import ProductionPsiUpdate
from router.models.llm_profile import LLMProfile


_DEFERRED_MSG = (
    "IncrementalPsiUpdater is deferred — see ROADMAP_P15.3.md. "
    "Land this when P15.3.7 hits eval-cost issues that incremental "
    "updates would solve."
)


class IncrementalPsiUpdater:
    """Stub. Construction itself raises so callers fail loudly."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEFERRED_MSG)

    def update(
        self, profile: LLMProfile, update: ProductionPsiUpdate
    ) -> LLMProfile:  # pragma: no cover — guarded by __init__
        raise NotImplementedError(_DEFERRED_MSG)
