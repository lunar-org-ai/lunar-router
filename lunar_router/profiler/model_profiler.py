"""
Model Profiler: Generates Psi vectors for LLMs.

Implements Eq. 12 from the paper: computes the error rate of an LLM
in each cluster by running it on the validation set S_val.
"""

from typing import Optional, Callable
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from ..core.embeddings import PromptEmbedder
from ..core.clustering import ClusterAssigner
from ..core.metrics import MetricType, get_metric
from ..models.llm_client import LLMClient
from ..models.llm_profile import LLMProfile
from ..data.dataset import PromptDataset


class ProfileCheckpoint:
    """Handles saving/loading profiling progress for resumption."""

    def __init__(self, checkpoint_dir: str = "./uniroute_state/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, model_id: str) -> Path:
        safe_name = model_id.replace("/", "_").replace(":", "_")
        return self.checkpoint_dir / f"{safe_name}_checkpoint.json"

    def save(
        self,
        model_id: str,
        current_idx: int,
        errors_per_cluster: np.ndarray,
        counts_per_cluster: np.ndarray,
        total_samples: int,
        metadata: dict,
    ) -> None:
        """Save profiling checkpoint."""
        data = {
            "model_id": model_id,
            "current_idx": current_idx,
            "total_samples": total_samples,
            "errors_per_cluster": errors_per_cluster.tolist(),
            "counts_per_cluster": counts_per_cluster.tolist(),
            "metadata": metadata,
        }
        with open(self._get_path(model_id), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, model_id: str) -> Optional[dict]:
        """Load checkpoint if exists."""
        path = self._get_path(model_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        data["errors_per_cluster"] = np.array(data["errors_per_cluster"])
        data["counts_per_cluster"] = np.array(data["counts_per_cluster"])
        return data

    def delete(self, model_id: str) -> None:
        """Delete checkpoint after successful completion."""
        path = self._get_path(model_id)
        if path.exists():
            path.unlink()

    def exists(self, model_id: str) -> bool:
        """Check if checkpoint exists."""
        return self._get_path(model_id).exists()


class ModelProfiler:
    """
    Profiles LLMs by computing their Ψ(h) representation.

    The Ψ vector contains the average error rate of the model
    in each cluster, computed from the validation set S_val.

    Implements Eq. 12:
        Ψ_k(h) = (1/|S_val,k|) Σ_{(x,y)∈S_val,k} 1{y ≠ h(x)}

    Attributes:
        embedder: Prompt embedder for computing φ(x).
        cluster_assigner: Cluster assigner for computing Φ(x).
        validation_set: S_val dataset.
        loss_fn: Loss function to measure errors.
    """

    def __init__(
        self,
        embedder: PromptEmbedder,
        cluster_assigner: ClusterAssigner,
        validation_set: PromptDataset,
        metric: MetricType = MetricType.EXACT_MATCH,
    ):
        """
        Initialize the profiler.

        Args:
            embedder: PromptEmbedder for generating embeddings.
            cluster_assigner: ClusterAssigner for mapping to clusters.
            validation_set: S_val dataset with (prompt, ground_truth) pairs.
            metric: Metric type for computing errors.
        """
        self.embedder = embedder
        self.cluster_assigner = cluster_assigner
        self.validation_set = validation_set
        self.metric = metric
        self.loss_fn = get_metric(metric)

        # Precompute cluster assignments for S_val
        self._precompute_clusters()

    def _precompute_clusters(self) -> None:
        """Precompute cluster assignments for all validation prompts."""
        self._val_clusters: list[int] = []
        self._val_embeddings: list[np.ndarray] = []

        for prompt, _ in self.validation_set:
            embedding = self.embedder.embed(prompt)
            result = self.cluster_assigner.assign(embedding)
            self._val_clusters.append(result.cluster_id)
            self._val_embeddings.append(embedding)

    def profile(
        self,
        llm_client: LLMClient,
        show_progress: bool = True,
        max_tokens: int = 256,
        temperature: float = 0.0,
        checkpoint_every: int = 10,
        resume: bool = True,
    ) -> LLMProfile:
        """
        Profile an LLM by evaluating it on S_val.

        Args:
            llm_client: The LLM client to profile.
            show_progress: Whether to show a progress bar.
            max_tokens: Max tokens for generation.
            temperature: Temperature for generation.
            checkpoint_every: Save checkpoint every N samples.
            resume: Whether to resume from checkpoint if available.

        Returns:
            LLMProfile with the computed Ψ vector.
        """
        K = self.cluster_assigner.num_clusters
        checkpoint = ProfileCheckpoint()

        # Try to resume from checkpoint
        start_idx = 0
        errors_per_cluster = np.zeros(K)
        counts_per_cluster = np.zeros(K)

        if resume:
            saved = checkpoint.load(llm_client.model_id)
            if saved is not None:
                start_idx = saved["current_idx"]
                errors_per_cluster = saved["errors_per_cluster"]
                counts_per_cluster = saved["counts_per_cluster"]
                print(f"   Resuming from checkpoint at sample {start_idx}/{saved['total_samples']}")

        # Prepare iterator
        samples = list(self.validation_set)

        if show_progress:
            pbar = tqdm(
                total=len(samples),
                initial=start_idx,
                desc=f"Profiling {llm_client.model_id}",
            )

        metadata = {
            "metric": self.metric.value,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Evaluate on each sample
        for idx in range(start_idx, len(samples)):
            prompt, ground_truth = samples[idx]

            # Generate response
            response = llm_client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Compute error (loss)
            error = self.loss_fn(response.text, ground_truth)

            # Accumulate in the appropriate cluster
            cluster_id = self._val_clusters[idx]
            errors_per_cluster[cluster_id] += error
            counts_per_cluster[cluster_id] += 1

            if show_progress:
                pbar.update(1)

            # Save checkpoint periodically
            if (idx + 1) % checkpoint_every == 0:
                checkpoint.save(
                    llm_client.model_id,
                    idx + 1,
                    errors_per_cluster,
                    counts_per_cluster,
                    len(samples),
                    metadata,
                )

        if show_progress:
            pbar.close()

        # Compute Ψ: average error per cluster
        psi_vector = self._compute_psi(errors_per_cluster, counts_per_cluster)

        # Delete checkpoint after successful completion
        checkpoint.delete(llm_client.model_id)

        return LLMProfile(
            model_id=llm_client.model_id,
            psi_vector=psi_vector,
            cost_per_1k_tokens=llm_client.cost_per_1k_tokens,
            num_validation_samples=len(samples),
            cluster_sample_counts=counts_per_cluster,
            metadata=metadata,
        )

    def _compute_psi(
        self,
        errors: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Ψ vector from accumulated errors and counts.

        Handles empty clusters by using the global average error rate.

        Args:
            errors: Total errors per cluster.
            counts: Sample counts per cluster.

        Returns:
            Ψ vector of shape (K,).
        """
        K = len(errors)
        psi = np.zeros(K)

        # Compute global fallback
        total_errors = errors.sum()
        total_count = counts.sum()
        global_error = total_errors / total_count if total_count > 0 else 0.5

        for k in range(K):
            if counts[k] > 0:
                psi[k] = errors[k] / counts[k]
            else:
                # Empty cluster: use global average
                psi[k] = global_error

        return psi

    def profile_batch(
        self,
        llm_clients: list[LLMClient],
        show_progress: bool = True,
        **kwargs,
    ) -> list[LLMProfile]:
        """
        Profile multiple LLMs.

        Args:
            llm_clients: List of LLM clients to profile.
            show_progress: Whether to show progress.
            **kwargs: Additional arguments for profile().

        Returns:
            List of LLMProfile objects.
        """
        profiles = []

        for client in llm_clients:
            profile = self.profile(client, show_progress=show_progress, **kwargs)
            profiles.append(profile)

        return profiles

    def incremental_profile(
        self,
        llm_client: LLMClient,
        existing_profile: Optional[LLMProfile] = None,
        new_samples: Optional[PromptDataset] = None,
        show_progress: bool = True,
    ) -> LLMProfile:
        """
        Incrementally update a profile with new samples.

        Useful when adding new validation samples without
        re-running the entire evaluation.

        Args:
            llm_client: The LLM client.
            existing_profile: Previous profile to update.
            new_samples: New samples to evaluate.
            show_progress: Whether to show progress.

        Returns:
            Updated LLMProfile.
        """
        if existing_profile is None or new_samples is None:
            # Fall back to full profiling
            return self.profile(llm_client, show_progress=show_progress)

        K = self.cluster_assigner.num_clusters

        # Start from existing stats
        errors = existing_profile.psi_vector * existing_profile.cluster_sample_counts
        counts = existing_profile.cluster_sample_counts.copy()

        # Evaluate new samples
        iterator = new_samples
        if show_progress:
            iterator = tqdm(iterator, desc="Incremental profiling")

        for prompt, ground_truth in iterator:
            embedding = self.embedder.embed(prompt)
            result = self.cluster_assigner.assign(embedding)

            response = llm_client.generate(prompt)
            error = self.loss_fn(response.text, ground_truth)

            cluster_id = result.cluster_id
            errors[cluster_id] += error
            counts[cluster_id] += 1

        # Recompute Ψ
        psi_vector = self._compute_psi(errors, counts)

        return LLMProfile(
            model_id=llm_client.model_id,
            psi_vector=psi_vector,
            cost_per_1k_tokens=llm_client.cost_per_1k_tokens,
            num_validation_samples=int(counts.sum()),
            cluster_sample_counts=counts,
            metadata={**existing_profile.metadata, "incremental": True},
        )


class ProfileAnalyzer:
    """
    Utilities for analyzing and comparing LLM profiles.
    """

    @staticmethod
    def compare_profiles(
        profile_a: LLMProfile,
        profile_b: LLMProfile,
    ) -> dict:
        """
        Compare two LLM profiles.

        Args:
            profile_a: First profile.
            profile_b: Second profile.

        Returns:
            Dictionary with comparison statistics.
        """
        diff = profile_a.psi_vector - profile_b.psi_vector

        return {
            "model_a": profile_a.model_id,
            "model_b": profile_b.model_id,
            "accuracy_a": profile_a.overall_accuracy,
            "accuracy_b": profile_b.overall_accuracy,
            "accuracy_diff": profile_a.overall_accuracy - profile_b.overall_accuracy,
            "clusters_a_better": int((diff < 0).sum()),
            "clusters_b_better": int((diff > 0).sum()),
            "clusters_equal": int((diff == 0).sum()),
            "max_diff": float(np.max(np.abs(diff))),
            "mean_diff": float(np.mean(diff)),
        }

    @staticmethod
    def find_complementary_models(
        profiles: list[LLMProfile],
    ) -> list[tuple[str, str, float]]:
        """
        Find pairs of models that complement each other.

        Two models are complementary if they excel in different clusters.

        Args:
            profiles: List of profiles to analyze.

        Returns:
            List of (model_a, model_b, complementarity_score) tuples.
        """
        results = []

        for i, p1 in enumerate(profiles):
            for p2 in profiles[i + 1:]:
                # Count clusters where each model is better
                p1_better = (p1.psi_vector < p2.psi_vector).sum()
                p2_better = (p2.psi_vector < p1.psi_vector).sum()

                # Complementarity: how evenly split are the wins?
                total = p1_better + p2_better
                if total > 0:
                    balance = min(p1_better, p2_better) / total
                else:
                    balance = 0

                # Weight by how different the errors are
                diff_magnitude = np.mean(np.abs(p1.psi_vector - p2.psi_vector))

                score = balance * diff_magnitude

                results.append((p1.model_id, p2.model_id, float(score)))

        return sorted(results, key=lambda x: -x[2])
