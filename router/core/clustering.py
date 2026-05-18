"""
Clustering components for UniRoute.

Implements the prompt representation function Φ(x) that maps
embeddings to cluster assignments.

P15.3.1 ports KMeans only (Section 5.1 of the paper).
LearnedMapClusterAssigner is deferred — see ROADMAP_P15.3.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ClusterResult:
    """
    Result of cluster assignment for a prompt.

    Attributes:
        cluster_id: The dominant cluster index (argmax of probabilities).
        probabilities: Full probability distribution over K clusters.
    """
    cluster_id: int
    probabilities: np.ndarray

    def __post_init__(self):
        """Validate the cluster result."""
        if self.cluster_id < 0 or self.cluster_id >= len(self.probabilities):
            raise ValueError(f"cluster_id {self.cluster_id} out of range")

    @property
    def num_clusters(self) -> int:
        """Return the number of clusters."""
        return len(self.probabilities)

    def to_one_hot(self) -> np.ndarray:
        """Return one-hot encoding of the dominant cluster."""
        one_hot = np.zeros(self.num_clusters)
        one_hot[self.cluster_id] = 1.0
        return one_hot


class ClusterAssigner(ABC):
    """
    Abstract base class for cluster assignment.

    Subclasses implement different strategies for mapping
    embeddings φ(x) to cluster representations Φ(x).
    """

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Return the number of clusters K."""
        ...

    @abstractmethod
    def assign(self, embedding: np.ndarray) -> ClusterResult:
        """
        Assign an embedding to clusters.

        Args:
            embedding: The prompt embedding φ(x) ∈ R^d.

        Returns:
            ClusterResult with cluster_id and probabilities.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the assigner state to a file."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "ClusterAssigner":
        """Load the assigner state from a file."""
        ...

    def assign_batch(self, embeddings: np.ndarray) -> list[ClusterResult]:
        """
        Assign multiple embeddings to clusters.

        Args:
            embeddings: Array of shape (N, d) with N embeddings.

        Returns:
            List of N ClusterResult objects.
        """
        return [self.assign(emb) for emb in embeddings]


class KMeansClusterAssigner(ClusterAssigner):
    """
    K-Means based cluster assignment (Section 5.1 of paper).

    Uses pre-computed centroids from K-Means clustering on training data.
    Assigns each prompt to the nearest centroid (one-hot encoding).

    Attributes:
        _centroids: Array of shape (K, d) with cluster centroids.
    """

    def __init__(self, centroids: np.ndarray):
        """
        Initialize with pre-computed centroids.

        Args:
            centroids: Array of shape (K, d) with K centroid vectors.
        """
        self._centroids = np.asarray(centroids)
        if self._centroids.ndim != 2:
            raise ValueError("centroids must be 2D array of shape (K, d)")

    @property
    def num_clusters(self) -> int:
        return self._centroids.shape[0]

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension d."""
        return self._centroids.shape[1]

    @property
    def centroids(self) -> np.ndarray:
        """Return the centroids array."""
        return self._centroids

    def assign(self, embedding: np.ndarray) -> ClusterResult:
        """
        Assign embedding to nearest centroid.

        Uses Euclidean distance to find the closest centroid.

        Args:
            embedding: The prompt embedding φ(x) ∈ R^d.

        Returns:
            ClusterResult with one-hot probabilities.
        """
        embedding = np.asarray(embedding)

        # Compute distances to all centroids
        # Using broadcasting: (K, d) - (d,) -> (K, d) -> sum -> (K,)
        distances = np.linalg.norm(self._centroids - embedding, axis=1)

        # Find nearest centroid
        cluster_id = int(np.argmin(distances))

        # One-hot probability distribution
        probs = np.zeros(self.num_clusters)
        probs[cluster_id] = 1.0

        return ClusterResult(cluster_id=cluster_id, probabilities=probs)

    def assign_batch(self, embeddings: np.ndarray) -> list[ClusterResult]:
        """
        Optimized batch assignment using vectorized operations.

        Args:
            embeddings: Array of shape (N, d).

        Returns:
            List of N ClusterResult objects.
        """
        embeddings = np.asarray(embeddings)

        # Compute all pairwise distances: (N, K)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        emb_sq = np.sum(embeddings ** 2, axis=1, keepdims=True)  # (N, 1)
        cent_sq = np.sum(self._centroids ** 2, axis=1)  # (K,)
        cross = embeddings @ self._centroids.T  # (N, K)

        distances_sq = emb_sq + cent_sq - 2 * cross  # (N, K)
        cluster_ids = np.argmin(distances_sq, axis=1)  # (N,)

        results = []
        for i, cid in enumerate(cluster_ids):
            probs = np.zeros(self.num_clusters)
            probs[cid] = 1.0
            results.append(ClusterResult(cluster_id=int(cid), probabilities=probs))

        return results

    def save(self, path: str | Path) -> None:
        """Save centroids to a .npz file."""
        path = Path(path)
        np.savez(
            path,
            type="kmeans",
            centroids=self._centroids,
        )

    @classmethod
    def load(cls, path: str | Path) -> "KMeansClusterAssigner":
        """Load centroids from a .npz file."""
        path = Path(path)
        data = np.load(path)

        if str(data.get("type", "kmeans")) != "kmeans":
            raise ValueError(f"Expected type 'kmeans', got '{data['type']}'")

        return cls(centroids=data["centroids"])


# LearnedMapClusterAssigner deferred — see ROADMAP_P15.3.md


def load_cluster_assigner(path: str | Path) -> ClusterAssigner:
    """
    Load a cluster assigner from file, auto-detecting the type.

    Args:
        path: Path to the .npz file.

    Returns:
        Currently always KMeansClusterAssigner (LearnedMap deferred).
    """
    path = Path(path)
    data = np.load(path)

    assigner_type = str(data.get("type", "kmeans"))

    if assigner_type == "kmeans":
        return KMeansClusterAssigner.load(path)
    raise ValueError(
        f"Unknown assigner type '{assigner_type}'. "
        "LearnedMap is deferred — see ROADMAP_P15.3.md."
    )
