"""Convert production traces into training signal.

Computes empirical error rates per (cluster, model) from production
traces, producing per-model ProductionPsiUpdate objects that the
proposer in P15.3.7 blends with benchmark Psi.

Ports the reference's TraceToTraining sans the ClickHouse query path —
this repo's traces live as filesystem JSONL + DuckDB. router/feedback/
store_adapter.py handles the read; this file just consumes TraceRecord.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import numpy as np

from router.models.llm_profile import LLMProfile

logger = logging.getLogger("router.feedback.trace_to_training")


@dataclass
class TraceRecord:
    """A single production trace.

    cluster_id can be -1 to indicate "unassigned" (cold-start mode of the
    store_adapter, when no fitted ClusterAssigner exists yet). Such records
    are skipped by add_trace before any Psi math touches them.
    """

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
    """Converts production traces into training signal for Psi updates.

    Error signals it considers:
    - Direct errors (TraceRecord.is_error)
    - Latency violations (latency_ms > latency_threshold_ms)
    - External quality flags keyed by request_id

    Usage:
        converter = TraceToTraining(num_clusters=K)
        converter.add_traces(traces)         # records with cluster_id == -1 are skipped
        updates = converter.compute_psi_updates()
        blended = converter.blend_with_profiles(existing_profiles, alpha=0.3)
    """

    def __init__(
        self,
        num_clusters: int,
        latency_threshold_ms: float = 30000.0,
        quality_flags: Optional[dict[str, bool]] = None,
    ):
        """Args:
            num_clusters: Number of clusters K in the routing system.
            latency_threshold_ms: Latency above this is counted as an error.
            quality_flags: Dict of request_id -> is_bad_quality (e.g. from a
                trace scanner / judge). Optional.
        """
        self.num_clusters = num_clusters
        self.latency_threshold_ms = latency_threshold_ms
        self.quality_flags = quality_flags or {}

        # Accumulators per model: error counts, sample counts, latency sums, cost sums.
        self._errors: dict[str, np.ndarray] = {}
        self._counts: dict[str, np.ndarray] = {}
        self._latency_sums: dict[str, np.ndarray] = {}
        self._cost_sums: dict[str, np.ndarray] = {}

    def add_trace(self, trace: TraceRecord) -> None:
        """Add a single trace.

        Records with ``cluster_id < 0`` (sentinel "unassigned") are skipped —
        the cold-start adapter emits these and we don't want them silently
        landing in psi_vector[-1] (which would be the last cluster).
        """
        mid = trace.selected_model
        k = trace.cluster_id

        if k < 0 or k >= self.num_clusters:
            return

        if mid not in self._errors:
            self._errors[mid] = np.zeros(self.num_clusters)
            self._counts[mid] = np.zeros(self.num_clusters)
            self._latency_sums[mid] = np.zeros(self.num_clusters)
            self._cost_sums[mid] = np.zeros(self.num_clusters)

        self._counts[mid][k] += 1
        self._latency_sums[mid][k] += trace.latency_ms
        self._cost_sums[mid][k] += trace.total_cost_usd

        is_error = (
            trace.is_error
            or trace.latency_ms > self.latency_threshold_ms
            or self.quality_flags.get(trace.request_id, False)
        )
        if is_error:
            self._errors[mid][k] += 1

    def add_traces(self, traces: list[TraceRecord]) -> None:
        for t in traces:
            self.add_trace(t)

    def compute_psi_updates(self) -> list[ProductionPsiUpdate]:
        """Compute production Psi vectors from accumulated traces.

        For empty clusters we fall back to the global error rate (avoids
        zero-bias on under-represented clusters; refinement comes from
        future traces or judge-driven preference data in P15.3.5/6).
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

            global_error_rate = errors.sum() / total
            psi = np.where(
                counts > 0,
                errors / np.maximum(counts, 1),
                global_error_rate,  # fallback for empty clusters
            )

            avg_latency = latency_sums.sum() / total
            avg_cost = cost_sums.sum() / total

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
        """Blend production Psi vectors with existing benchmark profiles.

        Psi_new = alpha * Psi_production + (1 - alpha) * Psi_benchmark

        Args:
            existing_profiles: Current benchmark-trained profiles.
            alpha: Weight for production data (0.0 = all benchmark,
                1.0 = all production). P15.3 default is 0.3 (locked).

        Returns:
            New LLMProfile list with blended Psi vectors. Profiles without
            production data pass through unchanged.
        """
        updates = {u.model_id: u for u in self.compute_psi_updates()}
        blended = []

        for profile in existing_profiles:
            update = updates.get(profile.model_id)

            if update is None:
                # No production data for this model — keep as-is.
                blended.append(profile)
                continue

            new_psi = alpha * update.psi_vector + (1 - alpha) * profile.psi_vector
            new_counts = (
                alpha * update.sample_counts
                + (1 - alpha) * profile.cluster_sample_counts
            )

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
                "blended model=%s benchmark_err=%.3f blended_err=%.3f alpha=%.2f traces=%d",
                profile.model_id,
                profile.overall_error_rate,
                blended[-1].overall_error_rate,
                alpha,
                update.total_traces,
            )

        return blended

    def reset(self) -> None:
        """Clear all accumulated traces."""
        self._errors.clear()
        self._counts.clear()
        self._latency_sums.clear()
        self._cost_sums.clear()
