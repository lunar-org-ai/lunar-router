"""Dataset curation proposer (P15.4.4).

Builds a ``Proposal(kind="dataset")`` carrying an inline candidate payload
for ``datasets/<name>/v<n+1>.json``. The candidate adds samples sourced
from whatever mining adapter matches the dataset's ``source`` string.

Adapter selection (by `dataset.source`):
  - "flagged traces"   → mining.flagged_traces
  - "language router"  → mining.language_router
  - "failed lookups"   → mining.failed_lookups
  - "feedback signals" → NotImplementedError (stub raises)
  - "manual" / other   → no auto-curation; raises NoAdapterError

The proposer:
  1. Loads the current dataset version.
  2. Picks an adapter via `mining.get_adapter`.
  3. Computes coverage gaps via `coverage.cluster_gaps` against the
     current router config (best-effort — cold-start router is fine).
  4. Streams candidate samples from the adapter (capped at
     `max_additions`). Already-existing IDs are filtered.
  5. Builds the v(n+1) payload: existing samples + new samples,
     bumped version, appended history line.
  6. Wraps it in a Proposal with kind="dataset" inline metadata + an
     optional Prediction tying coverage-gap shrinkage to the candidate.

Cold-start (no dataset exists yet for `name`) raises
`DatasetNotFoundError` — the caller (MCP / wakeup) can create one
manually first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from experiments.types import Mutation
from harness.proposer.dataset.coverage import CoverageReport, cluster_gaps
from harness.proposer.dataset.mining import get_adapter
from harness.types import Prediction, Proposal
from router.core.clustering import ClusterAssigner
from router.core.embeddings import PromptEmbedder
from router.data.dataset import Dataset, DatasetSample
from router.data.dataset_io import load_current
from router.errors import RouterError


logger = logging.getLogger("harness.proposer.dataset_proposer")


class NoAdapterError(RouterError):
    """Raised when no mining adapter matches the dataset's source string."""


class NothingToAddError(RouterError):
    """Raised when the adapter yielded zero new candidate samples."""


@dataclass(frozen=True)
class DatasetProposerConfig:
    """Static knobs. Defaults are intentionally conservative."""

    max_additions: int = 50              # cap per curation cycle
    proposal_source: str = "claude_code"  # surface in Lesson
    since_iso: Optional[str] = None       # date-window filter for adapters


@dataclass
class _Materials:
    """Intermediate state passed between proposer steps."""

    dataset: Dataset
    new_samples: list[DatasetSample]
    coverage_before: Optional[CoverageReport]
    coverage_after: Optional[CoverageReport]
    adapter_source: str


class DatasetProposer:
    """Build one dataset curation candidate per ``propose(name)`` call."""

    def __init__(
        self,
        *,
        embedder: PromptEmbedder,
        assigner: Optional[ClusterAssigner] = None,
        cfg: Optional[DatasetProposerConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.assigner = assigner
        self.cfg = cfg or DatasetProposerConfig()

    # ------------------------------------------------------------------

    def propose(
        self,
        name: str,
        *,
        source_override: Optional[str] = None,
    ) -> Proposal:
        """Build one curation candidate for dataset ``name``."""
        materials = self._gather(name, source_override=source_override)
        payload = self._build_payload(materials)

        next_version = payload["version"]
        target_path = f"datasets/{name}/v{next_version}.json"

        gap_before = (
            materials.coverage_before.gap_score
            if materials.coverage_before is not None
            else None
        )
        gap_after = (
            materials.coverage_after.gap_score
            if materials.coverage_after is not None
            else None
        )

        added = len(materials.new_samples)
        rationale = (
            f"Mined {added} new sample(s) for dataset {name!r} via "
            f"{materials.adapter_source!r}."
        )
        if gap_before is not None and gap_after is not None:
            rationale += (
                f" Expecting gap_score {gap_before:.2f} → {gap_after:.2f}."
            )

        prediction = Prediction(
            rubric="coverage_gap_score",
            expected_delta=(gap_after - gap_before) if (gap_before is not None and gap_after is not None) else -0.05,
            rationale=rationale,
            confidence=0.45,
        )

        proposal = Proposal(
            mutations=[
                Mutation(
                    file=target_path,
                    path="<inline_payload>",
                    value=payload,
                )
            ],
            description=(
                f"dataset curation: {name!r} v{next_version} "
                f"(+{added} from {materials.adapter_source})"
            ),
            source=self.cfg.proposal_source,
            metadata={
                "name": name,
                "adapter_source": materials.adapter_source,
                "added": added,
                "total_size": len(payload["samples"]),
                "gap_score_before": gap_before,
                "gap_score_after": gap_after,
            },
            prediction=prediction,
        )
        logger.info(
            "proposed dataset %s v%d: added=%d source=%s",
            name,
            next_version,
            added,
            materials.adapter_source,
        )
        return proposal

    # ------------------------------------------------------------------

    def _gather(
        self,
        name: str,
        *,
        source_override: Optional[str],
    ) -> _Materials:
        dataset = load_current(name)
        source = source_override or dataset.metadata.source

        adapter = get_adapter(source)
        if adapter is None:
            raise NoAdapterError(
                f"no mining adapter for dataset {name!r} source={source!r}. "
                f"Manual datasets must be edited via PUT /v1/datasets."
            )

        existing_ids = {s.id for s in dataset.samples}

        coverage_before = cluster_gaps(
            dataset,
            embedder=self.embedder,
            assigner=self.assigner,
        )

        # Stream new candidates from the adapter.
        new_samples: list[DatasetSample] = []
        try:
            for sample in adapter(
                since_iso=self.cfg.since_iso,
                embedder=self.embedder,
                existing=existing_ids,
                limit=self.cfg.max_additions,
            ):
                new_samples.append(sample)
        except NotImplementedError as e:
            raise NoAdapterError(str(e)) from e

        if not new_samples:
            raise NothingToAddError(
                f"adapter {source!r} yielded zero new samples for {name!r}"
            )

        # Compute coverage AFTER by simulating the merged dataset.
        merged_samples = list(dataset.samples) + new_samples
        merged_dataset = Dataset(
            metadata=dataset.metadata,
            version=dataset.version,
            samples=merged_samples,
            history=list(dataset.history),
            created_at=dataset.created_at,
            extra=dict(dataset.extra or {}),
        )
        coverage_after = cluster_gaps(
            merged_dataset,
            embedder=self.embedder,
            assigner=self.assigner,
        )

        return _Materials(
            dataset=dataset,
            new_samples=new_samples,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            adapter_source=source,
        )

    def _build_payload(self, m: _Materials) -> dict:
        now = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        new_version = m.dataset.version + 1
        merged_samples = list(m.dataset.samples) + m.new_samples

        history_line = {
            "when": now,
            "what": (
                f"Agent added {len(m.new_samples)} sample(s) "
                f"from {m.adapter_source}."
            ),
        }

        meta = dict(m.dataset.extra or {})
        meta.update({
            "stage": "auto_curate",
            "previous_version": m.dataset.version,
            "added": len(m.new_samples),
            "adapter_source": m.adapter_source,
            "gap_score_before": (
                m.coverage_before.gap_score if m.coverage_before else None
            ),
            "gap_score_after": (
                m.coverage_after.gap_score if m.coverage_after else None
            ),
        })

        return {
            "version": new_version,
            "name": m.dataset.metadata.name,
            "desc": m.dataset.metadata.desc,
            "source": m.dataset.metadata.source,
            "sourceType": m.dataset.metadata.sourceType,
            "use": list(m.dataset.metadata.use),
            "owner": m.dataset.metadata.owner,
            "growing": True,
            "created_at": m.dataset.created_at or now,
            "embedder_model": m.dataset.metadata.embedder_model,
            "embedding_dim": m.dataset.metadata.embedding_dim,
            "samples": [
                {
                    "id": s.id,
                    "prompt": s.prompt,
                    "ground_truth": s.ground_truth,
                    "tag": s.tag,
                    "trace_id": s.trace_id,
                    "added_at": s.added_at,
                    "source": s.source,
                    "embedding": s.embedding,
                }
                for s in merged_samples
            ],
            "history": list(m.dataset.history) + [history_line],
            "metadata": meta,
        }
