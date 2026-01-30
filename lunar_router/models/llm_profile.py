"""
LLM Profile: Stores the Psi vector representation of a model.

The profile contains the error rates per cluster computed during profiling,
along with metadata about the model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import json
import numpy as np


@dataclass
class LLMProfile:
    """
    Profile of an LLM containing its Ψ(h) representation.

    The Ψ vector contains the average error rate of the model
    in each cluster, computed from the validation set S_val.

    Attributes:
        model_id: Unique identifier for the model (e.g., "gpt-4o-mini").
        psi_vector: Error rates per cluster, shape (K,).
        cost_per_1k_tokens: Cost in dollars per 1000 tokens.
        num_validation_samples: Number of samples used to compute Ψ.
        cluster_sample_counts: Number of samples per cluster used in computation.
        metadata: Additional metadata (provider, version, etc.).
    """
    model_id: str
    psi_vector: np.ndarray
    cost_per_1k_tokens: float
    num_validation_samples: int
    cluster_sample_counts: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.psi_vector = np.asarray(self.psi_vector)
        self.cluster_sample_counts = np.asarray(self.cluster_sample_counts)

        if len(self.psi_vector) != len(self.cluster_sample_counts):
            raise ValueError(
                f"psi_vector length {len(self.psi_vector)} != "
                f"cluster_sample_counts length {len(self.cluster_sample_counts)}"
            )

    @property
    def num_clusters(self) -> int:
        """Return the number of clusters K."""
        return len(self.psi_vector)

    @property
    def overall_error_rate(self) -> float:
        """
        Compute the weighted average error rate across all clusters.

        Weighted by the number of samples in each cluster.
        """
        total_samples = self.cluster_sample_counts.sum()
        if total_samples == 0:
            return 0.0
        weighted_sum = (self.psi_vector * self.cluster_sample_counts).sum()
        return float(weighted_sum / total_samples)

    @property
    def overall_accuracy(self) -> float:
        """Compute the overall accuracy (1 - error_rate)."""
        return 1.0 - self.overall_error_rate

    def get_expected_error(self, phi: np.ndarray) -> float:
        """
        Compute expected error for a prompt given its cluster representation.

        Implements: γ(x, h) = Φ(x)ᵀ · Ψ(h)

        Args:
            phi: Cluster probability vector Φ(x) of shape (K,).

        Returns:
            Expected error rate for this model on the prompt.
        """
        phi = np.asarray(phi)
        if len(phi) != self.num_clusters:
            raise ValueError(
                f"phi length {len(phi)} != num_clusters {self.num_clusters}"
            )
        return float(np.dot(phi, self.psi_vector))

    def get_cluster_error(self, cluster_id: int) -> float:
        """
        Get the error rate for a specific cluster.

        Args:
            cluster_id: The cluster index.

        Returns:
            Error rate for the cluster.
        """
        if cluster_id < 0 or cluster_id >= self.num_clusters:
            raise ValueError(f"cluster_id {cluster_id} out of range [0, {self.num_clusters})")
        return float(self.psi_vector[cluster_id])

    def get_cluster_accuracy(self, cluster_id: int) -> float:
        """Get the accuracy for a specific cluster."""
        return 1.0 - self.get_cluster_error(cluster_id)

    def strongest_clusters(self, n: int = 5) -> list[tuple[int, float]]:
        """
        Get the clusters where this model performs best (lowest error).

        Args:
            n: Number of clusters to return.

        Returns:
            List of (cluster_id, error_rate) tuples, sorted by error ascending.
        """
        indices = np.argsort(self.psi_vector)[:n]
        return [(int(i), float(self.psi_vector[i])) for i in indices]

    def weakest_clusters(self, n: int = 5) -> list[tuple[int, float]]:
        """
        Get the clusters where this model performs worst (highest error).

        Args:
            n: Number of clusters to return.

        Returns:
            List of (cluster_id, error_rate) tuples, sorted by error descending.
        """
        indices = np.argsort(self.psi_vector)[-n:][::-1]
        return [(int(i), float(self.psi_vector[i])) for i in indices]

    def save(self, path: str | Path) -> None:
        """
        Save profile to a JSON file.

        Args:
            path: Path to save the profile.
        """
        path = Path(path)

        data = {
            "model_id": self.model_id,
            "psi_vector": self.psi_vector.tolist(),
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "num_validation_samples": self.num_validation_samples,
            "cluster_sample_counts": self.cluster_sample_counts.tolist(),
            "metadata": self.metadata,
            "_stats": {
                "num_clusters": self.num_clusters,
                "overall_error_rate": self.overall_error_rate,
                "overall_accuracy": self.overall_accuracy,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LLMProfile":
        """
        Load profile from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            LLMProfile instance.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            model_id=data["model_id"],
            psi_vector=np.array(data["psi_vector"]),
            cost_per_1k_tokens=data["cost_per_1k_tokens"],
            num_validation_samples=data["num_validation_samples"],
            cluster_sample_counts=np.array(data["cluster_sample_counts"]),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to a dictionary."""
        return {
            "model_id": self.model_id,
            "psi_vector": self.psi_vector.tolist(),
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "num_validation_samples": self.num_validation_samples,
            "cluster_sample_counts": self.cluster_sample_counts.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMProfile":
        """Create profile from a dictionary."""
        return cls(
            model_id=data["model_id"],
            psi_vector=np.array(data["psi_vector"]),
            cost_per_1k_tokens=data["cost_per_1k_tokens"],
            num_validation_samples=data["num_validation_samples"],
            cluster_sample_counts=np.array(data["cluster_sample_counts"]),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"LLMProfile(model_id='{self.model_id}', "
            f"num_clusters={self.num_clusters}, "
            f"accuracy={self.overall_accuracy:.2%}, "
            f"cost=${self.cost_per_1k_tokens}/1k)"
        )
