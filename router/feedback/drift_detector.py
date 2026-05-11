"""Distribution drift detection for routing clusters.

Monitors whether production prompt embeddings are drifting away from the
trained cluster centroids. P15.3.9's MCP ``router_health_check`` tool
calls ``DriftDetector.check()`` to expose the drift_score Claude Code
reads when deciding whether to propose a retrain.

The detector reports — it never acts. The brain interprets.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

from router.core.clustering import ClusterAssigner, KMeansClusterAssigner

logger = logging.getLogger("router.feedback.drift_detector")


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
        active = sum(1 for v in self.cluster_usage.values() if v > 0)
        total = len(self.cluster_usage)
        lines = [
            f"Drift Report ({self.num_embeddings} embeddings):",
            f"  Avg distance:    {self.avg_distance:.4f} (baseline: {self.baseline_distance:.4f})",
            f"  Drift ratio:     {self.drift_ratio:.2f}x",
            f"  Outlier fraction: {self.outlier_fraction:.1%}",
            f"  Needs reclustering: {self.needs_reclustering}",
            f"  Active clusters: {active}/{total}",
        ]
        return "\n".join(lines)


class DriftDetector:
    """Detects distribution drift in production prompt embeddings.

    Compares recent embeddings to trained cluster centroids. If the
    average distance increases significantly, the clusters no longer
    represent the traffic well and re-clustering is needed.

    Usage:
        detector = DriftDetector(assigner, baseline_distance=0.5)
        report = detector.check(recent_embeddings)
        if report.needs_reclustering:
            # P15.3.9: Claude Code sees this and may propose a retrain.
            ...

    Note for P15.3.7 callers: pass ``baseline_distance`` explicitly using
    the silhouette-fit's intra-cluster mean distance from the persisted
    config (``router_config_<n>.json:drift_baseline``). Do NOT let
    ``check()`` self-baseline on the first arbitrary embedding batch.
    """

    def __init__(
        self,
        cluster_assigner: ClusterAssigner,
        baseline_distance: Optional[float] = None,
        drift_threshold: float = 1.5,
        outlier_threshold: float = 0.2,
        outlier_distance_multiplier: float = 3.0,
    ):
        self.assigner = cluster_assigner
        self.baseline_distance = baseline_distance
        self.drift_threshold = drift_threshold
        self.outlier_threshold = outlier_threshold
        self.outlier_distance_multiplier = outlier_distance_multiplier

    def check(self, embeddings: np.ndarray) -> DriftReport:
        """Check embeddings for distribution drift.

        Args:
            embeddings: Array of shape (N, d) — recent prompt embeddings.

        Returns:
            DriftReport with drift metrics and a reclustering recommendation.
        """
        if len(embeddings) == 0:
            return DriftReport(
                avg_distance=0.0, baseline_distance=0.0, drift_ratio=1.0,
                outlier_fraction=0.0, needs_reclustering=False,
                num_embeddings=0, cluster_usage={},
            )

        distances = []
        cluster_usage: dict[int, int] = {i: 0 for i in range(self.assigner.num_clusters)}

        for emb in embeddings:
            result = self.assigner.assign(emb)
            cluster_usage[result.cluster_id] = cluster_usage.get(result.cluster_id, 0) + 1

            if isinstance(self.assigner, KMeansClusterAssigner):
                centroid = self.assigner.centroids[result.cluster_id]
                dist = float(np.linalg.norm(emb - centroid))
            else:
                # Learned-map / future assigners — use max probability as a proxy.
                dist = float(1.0 - result.probabilities.max())
            distances.append(dist)

        distances = np.array(distances)
        avg_distance = float(distances.mean())

        if self.baseline_distance is None:
            self.baseline_distance = avg_distance
            logger.info("drift baseline auto-set to %.4f (first check)", avg_distance)

        drift_ratio = (
            avg_distance / self.baseline_distance if self.baseline_distance > 0 else 1.0
        )

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
            logger.warning("drift detected: %s", report.summary())

        return report

    def update_baseline(self, new_baseline: float) -> None:
        """Update the baseline distance after a successful re-clustering."""
        self.baseline_distance = new_baseline
        logger.info("drift baseline updated to %.4f", new_baseline)
