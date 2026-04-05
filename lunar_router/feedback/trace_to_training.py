"""
Convert production traces into training signal.

Queries ClickHouse for recent request traces, computes empirical
error rates per (cluster, model), and produces production Psi vectors
that can be blended with benchmark Psi vectors.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import numpy as np

from ..models.llm_profile import LLMProfile

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """A single production trace (subset of ClickHouse llm_traces row)."""

    request_id: str
    selected_model: str
    cluster_id: int
    is_error: bool
    latency_ms: float
    total_cost_usd: float
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    error_category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionPsiUpdate:
    """Production-derived Psi vector for a single model."""

    model_id: str
    psi_vector: np.ndarray  # per-cluster error rates from production
    sample_counts: np.ndarray  # traces per cluster
    total_traces: int
    error_rate: float  # overall error rate
    avg_latency_ms: float
    avg_cost_usd: float


class TraceToTraining:
    """
    Converts production traces into training signal for Psi vector updates.

    Computes empirical error rates per (cluster, model) pair from real
    traffic data. Supports multiple error signals:
    - Direct errors (is_error=True)
    - Latency violations (exceeds threshold)
    - Quality issues (from external quality flags)

    Usage:
        converter = TraceToTraining(num_clusters=100)
        converter.add_traces(traces)
        updates = converter.compute_psi_updates()
        for update in updates:
            print(f"{update.model_id}: error_rate={update.error_rate:.2%}")
    """

    def __init__(
        self,
        num_clusters: int,
        latency_threshold_ms: float = 30000.0,
        quality_flags: Optional[dict[str, bool]] = None,
    ):
        """
        Args:
            num_clusters: Number of clusters K in the routing system.
            latency_threshold_ms: Latency above this is counted as an error.
            quality_flags: Dict of request_id -> is_bad_quality from TraceScanner.
        """
        self.num_clusters = num_clusters
        self.latency_threshold_ms = latency_threshold_ms
        self.quality_flags = quality_flags or {}

        # Accumulators: [model_id][(cluster, errors, counts, latency_sum, cost_sum)]
        self._errors: dict[str, np.ndarray] = {}
        self._counts: dict[str, np.ndarray] = {}
        self._latency_sums: dict[str, np.ndarray] = {}
        self._cost_sums: dict[str, np.ndarray] = {}

    def add_trace(self, trace: TraceRecord) -> None:
        """Add a single trace to the accumulators."""
        mid = trace.selected_model
        k = trace.cluster_id

        if k < 0 or k >= self.num_clusters:
            return

        # Initialize if first time seeing this model
        if mid not in self._errors:
            self._errors[mid] = np.zeros(self.num_clusters)
            self._counts[mid] = np.zeros(self.num_clusters)
            self._latency_sums[mid] = np.zeros(self.num_clusters)
            self._cost_sums[mid] = np.zeros(self.num_clusters)

        self._counts[mid][k] += 1
        self._latency_sums[mid][k] += trace.latency_ms
        self._cost_sums[mid][k] += trace.total_cost_usd

        # Count as error if any signal indicates failure
        is_error = (
            trace.is_error
            or trace.latency_ms > self.latency_threshold_ms
            or self.quality_flags.get(trace.request_id, False)
        )
        if is_error:
            self._errors[mid][k] += 1

    def add_traces(self, traces: list[TraceRecord]) -> None:
        """Add multiple traces."""
        for t in traces:
            self.add_trace(t)

    def compute_psi_updates(self) -> list[ProductionPsiUpdate]:
        """
        Compute production Psi vectors from accumulated traces.

        Returns:
            List of ProductionPsiUpdate, one per model.
        """
        updates = []

        for mid in self._errors:
            errors = self._errors[mid]
            counts = self._counts[mid]
            latency_sums = self._latency_sums[mid]
            cost_sums = self._cost_sums[mid]

            total = counts.sum()
            if total == 0:
                continue

            # Psi = error_rate per cluster, with global fallback for empty clusters
            global_error_rate = errors.sum() / total if total > 0 else 0.0
            psi = np.where(
                counts > 0,
                errors / np.maximum(counts, 1),
                global_error_rate,  # fallback for empty clusters
            )

            avg_latency = latency_sums.sum() / total if total > 0 else 0.0
            avg_cost = cost_sums.sum() / total if total > 0 else 0.0

            updates.append(ProductionPsiUpdate(
                model_id=mid,
                psi_vector=psi,
                sample_counts=counts,
                total_traces=int(total),
                error_rate=float(global_error_rate),
                avg_latency_ms=float(avg_latency),
                avg_cost_usd=float(avg_cost),
            ))

        return updates

    def blend_with_profiles(
        self,
        existing_profiles: list[LLMProfile],
        alpha: float = 0.3,
    ) -> list[LLMProfile]:
        """
        Blend production Psi vectors with existing benchmark profiles.

        Psi_new = alpha * Psi_production + (1 - alpha) * Psi_benchmark

        Args:
            existing_profiles: Current benchmark-trained profiles.
            alpha: Weight for production data (0.0 = all benchmark, 1.0 = all production).

        Returns:
            New LLMProfile list with blended Psi vectors.
        """
        updates = {u.model_id: u for u in self.compute_psi_updates()}
        blended = []

        for profile in existing_profiles:
            update = updates.get(profile.model_id)

            if update is None:
                # No production data for this model — keep as-is
                blended.append(profile)
                continue

            # Blend Psi vectors
            new_psi = alpha * update.psi_vector + (1 - alpha) * profile.psi_vector

            # Blend sample counts
            new_counts = alpha * update.sample_counts + (1 - alpha) * profile.cluster_sample_counts

            blended.append(LLMProfile(
                model_id=profile.model_id,
                psi_vector=new_psi,
                cost_per_1k_tokens=profile.cost_per_1k_tokens,
                num_validation_samples=profile.num_validation_samples + update.total_traces,
                cluster_sample_counts=new_counts,
                metadata={
                    **profile.metadata,
                    "production_blended": True,
                    "blend_alpha": alpha,
                    "production_traces": update.total_traces,
                    "production_error_rate": update.error_rate,
                },
            ))

            logger.info(
                f"Blended {profile.model_id}: "
                f"benchmark_error={profile.overall_error_rate:.3f} -> "
                f"blended_error={blended[-1].overall_error_rate:.3f} "
                f"(alpha={alpha}, traces={update.total_traces})"
            )

        return blended

    def reset(self) -> None:
        """Clear all accumulated traces."""
        self._errors.clear()
        self._counts.clear()
        self._latency_sums.clear()
        self._cost_sums.clear()
