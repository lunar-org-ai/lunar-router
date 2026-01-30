"""
K-Means Trainer: Trains cluster centroids from prompt embeddings.

Implements Section 5.1 of the paper: unsupervised K-Means clustering
on the training set S_tr to define K representative prompt clusters.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from ..core.embeddings import PromptEmbedder
from ..core.clustering import KMeansClusterAssigner
from ..data.dataset import PromptDataset


class KMeansTrainer:
    """
    Trains K-Means clusters on prompt embeddings.

    This is the unsupervised approach from Section 5.1 of the paper.
    Prompts are clustered purely based on their embedding similarity,
    without using any label information.

    Attributes:
        embedder: The prompt embedder to use.
        num_clusters: Number of clusters K.
    """

    def __init__(
        self,
        embedder: PromptEmbedder,
        num_clusters: int = 100,
    ):
        """
        Initialize the trainer.

        Args:
            embedder: PromptEmbedder instance for generating embeddings.
            num_clusters: Number of clusters K to create.
        """
        self.embedder = embedder
        self.num_clusters = num_clusters

    def train(
        self,
        training_set: PromptDataset,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        verbose: bool = True,
    ) -> KMeansClusterAssigner:
        """
        Train K-Means on the training set.

        Args:
            training_set: S_tr dataset with prompts.
            random_state: Random seed for reproducibility.
            n_init: Number of K-Means initializations.
            max_iter: Maximum iterations per initialization.
            verbose: Whether to print progress.

        Returns:
            KMeansClusterAssigner with trained centroids.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn required for K-Means training. "
                "Install with: pip install scikit-learn"
            )

        # Step 1: Extract all prompts
        prompts = training_set.get_prompts()

        if verbose:
            print(f"Training K-Means with K={self.num_clusters} on {len(prompts)} prompts...")

        # Step 2: Generate embeddings
        if verbose:
            print("Generating embeddings...")

        embeddings = self.embedder.embed_batch(prompts)

        if verbose:
            print(f"Embeddings shape: {embeddings.shape}")

        # Step 3: Run K-Means
        if verbose:
            print(f"Running K-Means (n_init={n_init}, max_iter={max_iter})...")

        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            verbose=0,
        )
        kmeans.fit(embeddings)

        if verbose:
            print(f"K-Means converged. Inertia: {kmeans.inertia_:.2f}")

        # Step 4: Create assigner
        centroids = kmeans.cluster_centers_

        if verbose:
            # Print cluster distribution
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, "
                  f"mean={counts.mean():.1f}, std={counts.std():.1f}")

        return KMeansClusterAssigner(centroids)

    def train_with_validation(
        self,
        training_set: PromptDataset,
        validation_set: PromptDataset,
        k_values: list[int],
        random_state: int = 42,
        verbose: bool = True,
    ) -> tuple[KMeansClusterAssigner, int]:
        """
        Train K-Means with automatic K selection via silhouette score.

        Tries multiple values of K and selects the best based on
        silhouette score on the validation set.

        Args:
            training_set: S_tr dataset for training.
            validation_set: S_val dataset for evaluation.
            k_values: List of K values to try.
            random_state: Random seed.
            verbose: Whether to print progress.

        Returns:
            Tuple of (best assigner, best K value).
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn required")

        # Generate embeddings
        train_prompts = training_set.get_prompts()
        val_prompts = validation_set.get_prompts()

        train_embeddings = self.embedder.embed_batch(train_prompts)
        val_embeddings = self.embedder.embed_batch(val_prompts)

        best_score = -1.0
        best_assigner = None
        best_k = k_values[0]

        for k in k_values:
            if verbose:
                print(f"Trying K={k}...")

            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
            )
            kmeans.fit(train_embeddings)

            # Evaluate on validation set
            val_labels = kmeans.predict(val_embeddings)

            # Silhouette score (higher is better)
            score = silhouette_score(val_embeddings, val_labels)

            if verbose:
                print(f"  K={k}: silhouette={score:.4f}, inertia={kmeans.inertia_:.2f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_assigner = KMeansClusterAssigner(kmeans.cluster_centers_)

        if verbose:
            print(f"Best K={best_k} with silhouette={best_score:.4f}")

        return best_assigner, best_k


class KMeansPlusPlusInitializer:
    """
    K-Means++ initialization for better centroid selection.

    Can be used to initialize centroids before training,
    or to create a custom KMeans implementation.
    """

    @staticmethod
    def initialize(
        embeddings: np.ndarray,
        k: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Initialize K centroids using K-Means++ algorithm.

        Args:
            embeddings: Array of shape (N, d).
            k: Number of centroids.
            random_state: Random seed.

        Returns:
            Array of shape (K, d) with initial centroids.
        """
        rng = np.random.default_rng(random_state)
        n_samples, n_features = embeddings.shape

        # Select first centroid randomly
        first_idx = rng.integers(0, n_samples)
        centroids = [embeddings[first_idx]]

        for _ in range(1, k):
            # Compute distances to nearest centroid
            centroids_arr = np.array(centroids)
            distances = np.min(
                np.linalg.norm(
                    embeddings[:, np.newaxis] - centroids_arr,
                    axis=2
                ),
                axis=1
            )

            # Square distances for probability
            distances_sq = distances ** 2

            # Select next centroid with probability proportional to D^2
            probs = distances_sq / distances_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(embeddings[next_idx])

        return np.array(centroids)


def analyze_clusters(
    assigner: KMeansClusterAssigner,
    dataset: PromptDataset,
    embedder: PromptEmbedder,
    top_n: int = 3,
) -> dict:
    """
    Analyze cluster composition and statistics.

    Args:
        assigner: The trained cluster assigner.
        dataset: Dataset to analyze.
        embedder: Embedder for generating embeddings.
        top_n: Number of example prompts per cluster.

    Returns:
        Dictionary with cluster statistics.
    """
    # Get embeddings and cluster assignments
    prompts = dataset.get_prompts()
    embeddings = embedder.embed_batch(prompts)
    results = assigner.assign_batch(embeddings)

    # Group prompts by cluster
    clusters: dict[int, list[str]] = {i: [] for i in range(assigner.num_clusters)}
    for prompt, result in zip(prompts, results):
        clusters[result.cluster_id].append(prompt)

    # Compute statistics
    stats = {
        "num_clusters": assigner.num_clusters,
        "num_samples": len(prompts),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "cluster_examples": {
            k: v[:top_n] for k, v in clusters.items()
        },
    }

    sizes = list(stats["cluster_sizes"].values())
    stats["size_stats"] = {
        "min": min(sizes),
        "max": max(sizes),
        "mean": np.mean(sizes),
        "std": np.std(sizes),
        "empty_clusters": sum(1 for s in sizes if s == 0),
    }

    return stats
