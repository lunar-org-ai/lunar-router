"""KMeansTrainResult — what KMeansTrainer.train() returns.

Carries the fitted assigner plus quality metrics + provenance so the
proposer in P15.3.7 can log to the ledger and the snapshotter can write
a partial router_config_<n>.json without re-deriving anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from router.core.clustering import KMeansClusterAssigner


@dataclass(frozen=True)
class KMeansTrainResult:
    """Output of KMeansTrainer.train().

    Attributes:
        assigner: Fitted KMeansClusterAssigner.
        k: Number of clusters.
        n_samples: Number of prompts the fit was trained on.
        silhouette: Silhouette score (NaN when N < 2K — sklearn rejects it).
                    Computed on a random subsample capped at silhouette_sample
                    to keep the O(N²) cost bounded.
        inertia: KMeans inertia (sum of squared distances to nearest centroid).
        cluster_sizes: dict[cluster_id -> count].
        embedder_model_id: model_name of the SentenceTransformerProvider used
                           (or 'mock' for tests). Stored alongside so the
                           on-disk router_config records which embedder
                           produced these centroids.
        fitted_at: ISO-8601 UTC timestamp.
        fitted_from: Provenance of the corpus (e.g.
                     {"source": "production_traces", "n_traces": 412,
                      "earliest": "...", "latest": "..."}).
                     Required so ledger entries can never lose where data
                     came from.
    """

    assigner: KMeansClusterAssigner
    k: int
    n_samples: int
    silhouette: float
    inertia: float
    cluster_sizes: dict[int, int]
    embedder_model_id: str
    fitted_at: str
    fitted_from: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human summary used in logs + ledger entries."""
        sizes = list(self.cluster_sizes.values()) or [0]
        return (
            f"K={self.k} N={self.n_samples} "
            f"silhouette={self.silhouette:.4f} inertia={self.inertia:.2f} "
            f"sizes_min={min(sizes)} sizes_max={max(sizes)}"
        )
