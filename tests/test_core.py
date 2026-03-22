"""Tests for core components: ClusterResult, KMeansClusterAssigner, PromptEmbedder."""

import numpy as np
import pytest
from pathlib import Path

from lunar_router.core.clustering import (
    ClusterResult,
    KMeansClusterAssigner,
    load_cluster_assigner,
)
from lunar_router.core.embeddings import PromptEmbedder


# ── ClusterResult ─────────────────────────────────────────────────────────────

class TestClusterResult:
    def test_basic_creation(self):
        probs = np.array([0.0, 1.0, 0.0])
        cr = ClusterResult(cluster_id=1, probabilities=probs)
        assert cr.cluster_id == 1
        assert cr.num_clusters == 3

    def test_to_one_hot(self):
        probs = np.array([0.2, 0.5, 0.3])
        cr = ClusterResult(cluster_id=1, probabilities=probs)
        one_hot = cr.to_one_hot()
        np.testing.assert_array_equal(one_hot, [0.0, 1.0, 0.0])

    def test_invalid_cluster_id(self):
        with pytest.raises(ValueError, match="out of range"):
            ClusterResult(cluster_id=5, probabilities=np.array([0.5, 0.5]))

    def test_negative_cluster_id(self):
        with pytest.raises(ValueError, match="out of range"):
            ClusterResult(cluster_id=-1, probabilities=np.array([0.5, 0.5]))


# ── KMeansClusterAssigner ────────────────────────────────────────────────────

class TestKMeansClusterAssigner:
    def test_creation(self):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assigner = KMeansClusterAssigner(centroids)
        assert assigner.num_clusters == 3
        assert assigner.embedding_dim == 2

    def test_assign_nearest(self):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        assigner = KMeansClusterAssigner(centroids)
        # Embedding close to first centroid
        result = assigner.assign(np.array([0.9, 0.1]))
        assert result.cluster_id == 0
        assert result.probabilities[0] == 1.0
        assert result.probabilities[1] == 0.0

    def test_assign_second_cluster(self):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        assigner = KMeansClusterAssigner(centroids)
        result = assigner.assign(np.array([0.1, 0.9]))
        assert result.cluster_id == 1

    def test_assign_batch(self):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        assigner = KMeansClusterAssigner(centroids)
        embeddings = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        results = assigner.assign_batch(embeddings)
        assert len(results) == 3
        assert results[0].cluster_id == 0
        assert results[1].cluster_id == 1

    def test_invalid_centroids_1d(self):
        with pytest.raises(ValueError, match="2D array"):
            KMeansClusterAssigner(np.array([1.0, 2.0]))

    def test_save_and_load(self, tmp_path):
        centroids = np.random.randn(10, 8)
        assigner = KMeansClusterAssigner(centroids)
        filepath = tmp_path / "clusters.npz"
        assigner.save(filepath)

        loaded = load_cluster_assigner(filepath)
        assert isinstance(loaded, KMeansClusterAssigner)
        assert loaded.num_clusters == 10
        assert loaded.embedding_dim == 8
        np.testing.assert_array_almost_equal(loaded.centroids, centroids)

    def test_centroids_property(self):
        centroids = np.array([[1.0, 2.0], [3.0, 4.0]])
        assigner = KMeansClusterAssigner(centroids)
        np.testing.assert_array_equal(assigner.centroids, centroids)


# ── PromptEmbedder ────────────────────────────────────────────────────────────

class FakeEmbeddingProvider:
    """Deterministic embedding provider for testing."""

    def __init__(self, dim=8):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        # Deterministic: hash-based embedding
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(self._dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class TestPromptEmbedder:
    def test_embed(self):
        provider = FakeEmbeddingProvider(dim=4)
        embedder = PromptEmbedder(provider, cache_enabled=False)
        emb = embedder.embed("hello")
        assert emb.shape == (4,)

    def test_dimension(self):
        provider = FakeEmbeddingProvider(dim=16)
        embedder = PromptEmbedder(provider)
        assert embedder.dimension == 16

    def test_caching(self):
        provider = FakeEmbeddingProvider(dim=4)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        emb1 = embedder.embed("test prompt")
        emb2 = embedder.embed("test prompt")
        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_disabled(self):
        provider = FakeEmbeddingProvider(dim=4)
        embedder = PromptEmbedder(provider, cache_enabled=False)
        emb1 = embedder.embed("test")
        emb2 = embedder.embed("test")
        # Should still be deterministic with same seed
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self):
        provider = FakeEmbeddingProvider(dim=8)
        embedder = PromptEmbedder(provider, cache_enabled=False)
        emb1 = embedder.embed("hello world")
        emb2 = embedder.embed("goodbye world")
        assert not np.array_equal(emb1, emb2)

    def test_cache_eviction(self):
        provider = FakeEmbeddingProvider(dim=4)
        embedder = PromptEmbedder(provider, cache_enabled=True, cache_max_size=10)
        for i in range(20):
            embedder.embed(f"text_{i}")
        # Cache should stay bounded around max_size
        assert len(embedder._cache) <= 10

    def test_cache_eviction_small_size(self):
        """Regression: eviction must work even with cache_max_size < 10."""
        provider = FakeEmbeddingProvider(dim=4)
        embedder = PromptEmbedder(provider, cache_enabled=True, cache_max_size=3)
        for i in range(10):
            embedder.embed(f"text_{i}")
        # With max_size=3, cache should never exceed 3
        assert len(embedder._cache) <= 3
