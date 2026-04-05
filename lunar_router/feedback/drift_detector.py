"""
Distribution drift detection for routing clusters.

Monitors whether production prompt embeddings are drifting away from
the trained cluster centroids. If drift exceeds a threshold, triggers
re-clustering to adapt to changing traffic patterns.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

from ..core.clustering import ClusterAssigner, KMeansClusterAssigner

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Result of a drift detection check."""

    avg_distance: float  # mean distance of recent embeddings to nearest centroid
    baseline_distance: float  # historical baseline average distance
    drift_ratio: float  # avg_distance / baseline_distance (>1 = drift)
    outlier_fraction: float  # fraction of embeddings far from all centroids
    needs_reclustering: bool  # whether drift exceeds threshold
    num_embeddings: int
    cluster_usage: dict[int, int]  # how many embeddings per cluster

    def summary(self) -> str:
        lines = [
            f"Drift Report ({self.num_embeddings} embeddings):",
            f"  Avg distance:    {self.avg_distance:.4f} (baseline: {self.baseline_distance:.4f})",
            f"  Drift ratio:     {self.drift_ratio:.2f}x",
            f"  Outlier fraction: {self.outlier_fraction:.1%}",
            f"  Needs reclustering: {self.needs_reclustering}",
            f"  Active clusters: {sum(1 for v in self.cluster_usage.values() if v > 0)}/{len(self.cluster_usage)}",
        ]
        return "\n".join(lines)


class DriftDetector:
    """
    Detects distribution drift in production prompt embeddings.

    Compares recent embeddings to trained cluster centroids. If the
    average distance increases significantly, the clusters no longer
    represent the traffic well and re-clustering is needed.

    Usage:
        detector = DriftDetector(assigner, baseline_distance=0.5)
        report = detector.check(recent_embeddings)
        if report.needs_reclustering:
            # trigger re-clustering pipeline
    """

    def __init__(
        self,
        cluster_assigner: ClusterAssigner,
        baseline_distance: Optional[float] = None,
        drift_threshold: float = 1.5,
        outlier_threshold: float = 0.2,
        outlier_distance_multiplier: float = 3.0,
    ):
        """
        Args:
            cluster_assigner: Current cluster assigner with centroids.
            baseline_distance: Historical avg distance. If None, computed from
                first check() call.
            drift_threshold: Ratio above which drift is flagged (1.5 = 50% increase).
            outlier_threshold: Fraction of outliers that triggers reclustering.
            outlier_distance_multiplier: A point is an outlier if its distance
                to nearest centroid exceeds this * baseline_distance.
        """
        self.assigner = cluster_assigner
        self.baseline_distance = baseline_distance
        self.drift_threshold = drift_threshold
        self.outlier_threshold = outlier_threshold
        self.outlier_distance_multiplier = outlier_distance_multiplier

    def check(self, embeddings: np.ndarray) -> DriftReport:
        """
        Check embeddings for distribution drift.

        Args:
            embeddings: Array of shape (N, d) — recent prompt embeddings.

        Returns:
            DriftReport with drift metrics and reclustering recommendation.
        """
        if len(embeddings) == 0:
            return DriftReport(
                avg_distance=0.0, baseline_distance=0.0, drift_ratio=1.0,
                outlier_fraction=0.0, needs_reclustering=False,
                num_embeddings=0, cluster_usage={},
            )

        # Compute distances to nearest centroids
        distances = []
        cluster_usage: dict[int, int] = {i: 0 for i in range(self.assigner.num_clusters)}

        for emb in embeddings:
            result = self.assigner.assign(emb)
            cluster_usage[result.cluster_id] = cluster_usage.get(result.cluster_id, 0) + 1

            # Compute actual distance to nearest centroid
            if isinstance(self.assigner, KMeansClusterAssigner):
                centroid = self.assigner._centroids[result.cluster_id]
                dist = float(np.linalg.norm(emb - centroid))
            else:
                # For learned map, use max probability as proxy (1 - max_prob)
                dist = float(1.0 - result.probabilities.max())
            distances.append(dist)

        distances = np.array(distances)
        avg_distance = float(distances.mean())

        # Set baseline on first call if not provided
        if self.baseline_distance is None:
            self.baseline_distance = avg_distance
            logger.info(f"Drift baseline set to {avg_distance:.4f}")

        # Compute drift metrics
        drift_ratio = avg_distance / self.baseline_distance if self.baseline_distance > 0 else 1.0

        outlier_cutoff = self.baseline_distance * self.outlier_distance_multiplier
        outlier_fraction = float((distances > outlier_cutoff).mean())

        needs_reclustering = (
            drift_ratio > self.drift_threshold
            or outlier_fraction > self.outlier_threshold
        )

        report = DriftReport(
            avg_distance=avg_distance,
            baseline_distance=self.baseline_distance,
            drift_ratio=drift_ratio,
            outlier_fraction=outlier_fraction,
            needs_reclustering=needs_reclustering,
            num_embeddings=len(embeddings),
            cluster_usage=cluster_usage,
        )

        if needs_reclustering:
            logger.warning(f"Drift detected! {report.summary()}")

        return report

    def update_baseline(self, new_baseline: float) -> None:
        """Update the baseline distance after re-clustering."""
        self.baseline_distance = new_baseline
        logger.info(f"Drift baseline updated to {new_baseline:.4f}")
