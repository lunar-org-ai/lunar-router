"""
Incremental updater: orchestrates the production feedback loop.

1. Load current weights
2. Ingest production traces → compute production Psi deltas
3. Blend production + benchmark Psi vectors
4. Check for distribution drift
5. Export updated weights
6. Optionally trigger Go engine hot-reload
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import json
import logging
import time

from ..models.llm_profile import LLMProfile
from ..models.llm_registry import LLMRegistry
from ..core.clustering import ClusterAssigner, load_cluster_assigner
from .trace_to_training import TraceToTraining, TraceRecord, ProductionPsiUpdate
from .drift_detector import DriftDetector, DriftReport

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of an incremental update cycle."""

    success: bool
    traces_processed: int
    models_updated: int
    drift_report: Optional[DriftReport]
    old_error_rates: dict[str, float]  # model -> old error rate
    new_error_rates: dict[str, float]  # model -> new error rate
    output_path: Optional[Path] = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Update Result ({'SUCCESS' if self.success else 'FAILED'}):",
            f"  Traces processed: {self.traces_processed}",
            f"  Models updated:   {self.models_updated}",
            f"  Duration:         {self.duration_ms:.0f}ms",
        ]
        if self.old_error_rates and self.new_error_rates:
            lines.append("  Error rate changes:")
            for mid in self.new_error_rates:
                old = self.old_error_rates.get(mid, 0)
                new = self.new_error_rates[mid]
                delta = new - old
                arrow = "+" if delta > 0 else ""
                lines.append(f"    {mid}: {old:.3f} -> {new:.3f} ({arrow}{delta:.3f})")
        if self.drift_report:
            lines.append(f"  Drift ratio: {self.drift_report.drift_ratio:.2f}x")
        if self.output_path:
            lines.append(f"  Weights saved to: {self.output_path}")
        return "\n".join(lines)


class IncrementalUpdater:
    """
    End-to-end incremental update pipeline.

    Usage:
        updater = IncrementalUpdater(
            weights_path="./weights",
            alpha=0.3,
        )

        # Feed production traces
        result = updater.update(traces)
        print(result.summary())

        # Result includes new weights path for Go engine reload
    """

    def __init__(
        self,
        weights_path: str | Path,
        alpha: float = 0.3,
        latency_threshold_ms: float = 30000.0,
        drift_threshold: float = 1.5,
        output_path: Optional[str | Path] = None,
    ):
        """
        Args:
            weights_path: Path to current weights directory.
            alpha: Blend weight for production data (0=all benchmark, 1=all production).
            latency_threshold_ms: Latency above this counts as error.
            drift_threshold: Drift ratio triggering re-clustering.
            output_path: Where to save updated weights. Defaults to weights_path.
        """
        self.weights_path = Path(weights_path)
        self.alpha = alpha
        self.latency_threshold_ms = latency_threshold_ms
        self.drift_threshold = drift_threshold
        self.output_path = Path(output_path) if output_path else self.weights_path

        # Load current state
        self._profiles: list[LLMProfile] = []
        self._assigner: Optional[ClusterAssigner] = None
        self._num_clusters: int = 0

    def load_weights(self) -> None:
        """Load current weights from disk."""
        # Load profiles
        profiles_dir = self.weights_path / "profiles"
        if profiles_dir.exists():
            self._profiles = []
            for f in sorted(profiles_dir.glob("*.json")):
                self._profiles.append(LLMProfile.load(f))
            logger.info(f"Loaded {len(self._profiles)} profiles from {profiles_dir}")

        # Load cluster assigner
        clusters_dir = self.weights_path / "clusters"
        if clusters_dir.exists():
            for name in ["mmlu_full.npz", "default.npz"]:
                path = clusters_dir / name
                if path.exists():
                    self._assigner = load_cluster_assigner(path)
                    self._num_clusters = self._assigner.num_clusters
                    break

        if not self._profiles:
            raise ValueError(f"No profiles found in {self.weights_path}")

    def update(
        self,
        traces: list[TraceRecord],
        quality_flags: Optional[dict[str, bool]] = None,
        embeddings: Optional[Any] = None,
    ) -> UpdateResult:
        """
        Run one incremental update cycle.

        Args:
            traces: Production traces from ClickHouse.
            quality_flags: Optional {request_id: is_bad} from TraceScanner.
            embeddings: Optional (N, d) array of trace embeddings for drift detection.

        Returns:
            UpdateResult with details of what changed.
        """
        start = time.time()

        if not self._profiles:
            self.load_weights()

        num_clusters = self._num_clusters or (
            self._profiles[0].num_clusters if self._profiles else 100
        )

        # Step 1: Convert traces to training signal
        converter = TraceToTraining(
            num_clusters=num_clusters,
            latency_threshold_ms=self.latency_threshold_ms,
            quality_flags=quality_flags or {},
        )
        converter.add_traces(traces)

        # Record old error rates
        old_rates = {p.model_id: p.overall_error_rate for p in self._profiles}

        # Step 2: Blend production + benchmark Psi
        new_profiles = converter.blend_with_profiles(self._profiles, alpha=self.alpha)

        # Record new error rates
        new_rates = {p.model_id: p.overall_error_rate for p in new_profiles}

        # Step 3: Drift detection
        drift_report = None
        if embeddings is not None and self._assigner is not None:
            detector = DriftDetector(
                self._assigner,
                drift_threshold=self.drift_threshold,
            )
            drift_report = detector.check(embeddings)

        # Step 4: Export updated weights
        output = self.output_path
        self._save_profiles(new_profiles, output)

        # Update internal state
        self._profiles = new_profiles

        duration = (time.time() - start) * 1000

        result = UpdateResult(
            success=True,
            traces_processed=len(traces),
            models_updated=len(new_profiles),
            drift_report=drift_report,
            old_error_rates=old_rates,
            new_error_rates=new_rates,
            output_path=output,
            duration_ms=duration,
        )

        logger.info(result.summary())
        return result

    def update_from_profiles(
        self,
        production_profiles: list[LLMProfile],
        alpha: Optional[float] = None,
    ) -> UpdateResult:
        """
        Update by blending with pre-computed production profiles.

        Useful when production Psi vectors are computed externally.
        """
        start = time.time()

        if not self._profiles:
            self.load_weights()

        blend_alpha = alpha if alpha is not None else self.alpha
        old_rates = {p.model_id: p.overall_error_rate for p in self._profiles}

        prod_map = {p.model_id: p for p in production_profiles}
        new_profiles = []

        for profile in self._profiles:
            prod = prod_map.get(profile.model_id)
            if prod:
                new_psi = blend_alpha * prod.psi_vector + (1 - blend_alpha) * profile.psi_vector
                new_counts = blend_alpha * prod.cluster_sample_counts + (1 - blend_alpha) * profile.cluster_sample_counts
                new_profiles.append(LLMProfile(
                    model_id=profile.model_id,
                    psi_vector=new_psi,
                    cost_per_1k_tokens=profile.cost_per_1k_tokens,
                    num_validation_samples=profile.num_validation_samples,
                    cluster_sample_counts=new_counts,
                    metadata={**profile.metadata, "production_blended": True},
                ))
            else:
                new_profiles.append(profile)

        new_rates = {p.model_id: p.overall_error_rate for p in new_profiles}
        self._save_profiles(new_profiles, self.output_path)
        self._profiles = new_profiles

        return UpdateResult(
            success=True,
            traces_processed=0,
            models_updated=len(new_profiles),
            drift_report=None,
            old_error_rates=old_rates,
            new_error_rates=new_rates,
            output_path=self.output_path,
            duration_ms=(time.time() - start) * 1000,
        )

    def _save_profiles(self, profiles: list[LLMProfile], output_dir: Path) -> None:
        """Save updated profiles to disk."""
        profiles_dir = output_dir / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        for profile in profiles:
            safe_name = profile.model_id.replace("/", "_")
            profile.save(profiles_dir / f"{safe_name}.json")

        logger.info(f"Saved {len(profiles)} updated profiles to {profiles_dir}")
