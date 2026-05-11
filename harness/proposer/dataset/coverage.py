"""Cluster gap analysis for datasets (P15.4.4).

Given a fitted router (`KMeansClusterAssigner`) and a dataset, this
module reports per-cluster coverage: how many samples the dataset
actually has per cluster vs. an "expected" floor (uniform distribution
of `dataset.size / k` rounded up, by default).

The output drives:
  1. The proposer: pick adapters whose new samples land in
     under-covered clusters first.
  2. The critic: assert the gap distribution shrunk (or stayed flat)
     after a curation cycle — never widened.
  3. The UI / MCP health endpoint: surface a single coverage_gap_score
     in [0, 1] so the operator can see at a glance whether a dataset
     spans the prompt space.

Cold-start (no fitted assigner): returns an empty gaps dict and a
gap_score of None — callers degrade gracefully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from router.core.clustering import ClusterAssigner
from router.core.embeddings import PromptEmbedder
from router.data.dataset import Dataset


@dataclass(frozen=True)
class ClusterGap:
    cluster_id: int
    expected: int
    actual: int
    gap: int  # max(expected - actual, 0)

    def as_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "expected": self.expected,
            "actual": self.actual,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class CoverageReport:
    k: int
    total_samples: int
    expected_per_cluster: int
    cluster_distribution: dict[int, int]
    gaps: list[ClusterGap]
    gap_score: float  # in [0, 1]; 0 = uniform, 1 = all in one cluster

    def under_covered_clusters(self) -> list[int]:
        """Cluster IDs with gap > 0, biggest gap first."""
        return [
            g.cluster_id
            for g in sorted(self.gaps, key=lambda x: -x.gap)
            if g.gap > 0
        ]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "total_samples": self.total_samples,
            "expected_per_cluster": self.expected_per_cluster,
            "cluster_distribution": {str(k): v for k, v in self.cluster_distribution.items()},
            "gap_score": self.gap_score,
            "gaps": [g.as_dict() for g in self.gaps],
        }


def cluster_gaps(
    dataset: Dataset,
    *,
    embedder: Optional[PromptEmbedder] = None,
    assigner: Optional[ClusterAssigner] = None,
) -> Optional[CoverageReport]:
    """Compute per-cluster coverage for a dataset.

    Args:
        dataset: the dataset to analyze.
        embedder: not used when samples already have embeddings; required
                  only as a fallback for legacy samples missing
                  ``embedding``.
        assigner: fitted cluster assigner. None → cold-start → returns None.

    Returns:
        CoverageReport, or None when no assigner exists (cold-start).
    """
    if assigner is None or assigner.centroids is None or assigner.centroids.size == 0:
        return None

    k = int(assigner.centroids.shape[0])
    if k <= 0 or not dataset.samples:
        return CoverageReport(
            k=max(k, 0),
            total_samples=len(dataset.samples),
            expected_per_cluster=0,
            cluster_distribution={},
            gaps=[],
            gap_score=0.0,
        )

    # Project each sample's embedding into its cluster.
    counts: dict[int, int] = {i: 0 for i in range(k)}
    for s in dataset.samples:
        if s.embedding:
            vec = np.asarray(s.embedding, dtype=float)
        elif embedder is not None:
            vec = np.asarray(embedder.embed(s.prompt), dtype=float)
        else:
            # No embedding + no embedder → can't assign; skip.
            continue
        cid = int(assigner.assign(vec).cluster_id)
        if 0 <= cid < k:
            counts[cid] += 1

    n = sum(counts.values())
    expected = math.ceil(n / k) if n > 0 else 0

    gaps = [
        ClusterGap(
            cluster_id=cid,
            expected=expected,
            actual=counts[cid],
            gap=max(expected - counts[cid], 0),
        )
        for cid in range(k)
    ]

    # gap_score = total deficit relative to perfect coverage.
    # 0 = each cluster has at least `expected` samples; 1 = all in one cluster.
    if n == 0:
        gap_score = 0.0
    else:
        total_deficit = sum(g.gap for g in gaps)
        max_possible_deficit = expected * (k - 1)
        gap_score = (
            total_deficit / max_possible_deficit
            if max_possible_deficit > 0
            else 0.0
        )
        gap_score = max(0.0, min(1.0, gap_score))

    return CoverageReport(
        k=k,
        total_samples=n,
        expected_per_cluster=expected,
        cluster_distribution=counts,
        gaps=gaps,
        gap_score=gap_score,
    )
