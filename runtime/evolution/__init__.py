"""AHE evolution loop — Algorithm 1 of Lin et al. 2604.25850v3.

One iteration: optionally verify the previous round's Change Manifest
predictions, run a rollout on the current harness, distill evidence,
spawn a separate Evolve Agent sandbox that edits NexAU components +
writes a fresh pending manifest, snapshot the workspace back, and
invalidate the per-agent executor cache so the next chat request
picks up the new harness.

See :func:`runtime.evolution.loop.run_one_iteration` for the entry
point. The :class:`runtime.evolution.types.IterationResult` dataclass
captures everything that happened so the caller (or UI) can show the
delta.

v0 limitations (documented in :mod:`.types`):
  - rollout runs each task once (k=1); no statistical signal yet
  - Clean/dedup step skipped
  - Attribution is best-effort: verdict recorded, no automatic
    file-level rollback when predictions miss
  - Distill is the raw rollout corpus, not the layered evidence
    report the paper's Agent Debugger emits
"""

from runtime.evolution.loop import run_one_iteration
from runtime.evolution.types import (
    Evidence,
    EvidenceCluster,
    IterationResult,
    RolloutResult,
    TaskOutcome,
)

__all__ = [
    "Evidence",
    "EvidenceCluster",
    "IterationResult",
    "RolloutResult",
    "TaskOutcome",
    "run_one_iteration",
]
