"""Smoke tests for router/core.

Fast tests use MockEmbeddingProvider so we don't download miniLM.
The slow test that confirms SentenceTransformerProvider's 384-dim is gated
behind OPENTRACY_RUN_SLOW=1 — run once after a fresh `uv sync --extra router`.
"""

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from router.core.embeddings import (
    PromptEmbedder,
    MockEmbeddingProvider,
    SentenceTransformerProvider,
)
from router.core.clustering import (
    ClusterResult,
    KMeansClusterAssigner,
    load_cluster_assigner,
)
from router.core.metrics import (
    MetricType,
    exact_match,
    normalized_exact_match,
    contains_match,
    f1_score_loss,
    mmlu_match,
    get_metric,
    compute_accuracy,
)


# --- PromptEmbedder + providers ---


def test_mock_embedder_dimension():
    """MockEmbeddingProvider returns the configured dimension."""
    provider = MockEmbeddingProvider(dimension=128)
    assert provider.dimension == 128
    emb = PromptEmbedder(provider)
    assert emb.dimension == 128
    vec = emb.embed("hello")
    assert vec.shape == (128,)


def test_embedder_caches_same_string():
    """Two embed() calls on the same string return the same array (cache hit)."""
    emb = PromptEmbedder(MockEmbeddingProvider(dimension=64))
    a = emb.embed("hello world")
    b = emb.embed("hello world")
    # Cache returns the same numpy array object on the second call.
    assert a is b


def test_embedder_batch_round_trip():
    """embed_batch on N strings returns shape (N, d)."""
    emb = PromptEmbedder(MockEmbeddingProvider(dimension=32))
    vecs = emb.embed_batch(["a", "b", "c", "d"])
    assert vecs.shape == (4, 32)


def test_embedder_clear_cache():
    """clear_cache() empties the cache so subsequent embed calls re-fetch."""
    emb = PromptEmbedder(MockEmbeddingProvider(dimension=8))
    a = emb.embed("x")
    emb.clear_cache()
    b = emb.embed("x")
    # Same numerical values (deterministic mock) but different array objects.
    assert np.array_equal(a, b)
    assert a is not b


@pytest.mark.skipif(
    not os.getenv("OPENTRACY_RUN_SLOW"),
    reason="needs sentence-transformers model download; set OPENTRACY_RUN_SLOW=1",
)
def test_sentence_transformer_default_dim_is_384():
    """SentenceTransformerProvider with default model is 384-dim (all-MiniLM-L6-v2)."""
    provider = SentenceTransformerProvider()
    assert provider.dimension == 384
    emb = PromptEmbedder(provider)
    vec = emb.embed("hello")
    assert vec.shape == (384,)


# --- ClusterAssigner / KMeans ---


def test_cluster_result_one_hot():
    """to_one_hot returns expected indicator vector."""
    res = ClusterResult(cluster_id=2, probabilities=np.array([0.1, 0.2, 0.7]))
    one_hot = res.to_one_hot()
    assert one_hot.shape == (3,)
    assert one_hot[2] == 1.0
    assert one_hot[0] == 0.0 and one_hot[1] == 0.0


def test_cluster_result_validates_id_range():
    """cluster_id outside probabilities length raises."""
    with pytest.raises(ValueError):
        ClusterResult(cluster_id=5, probabilities=np.array([0.1, 0.2, 0.7]))


def test_kmeans_assigner_picks_nearest_centroid():
    """assign() returns the nearest centroid in Euclidean distance."""
    centroids = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    assigner = KMeansClusterAssigner(centroids)
    assert assigner.num_clusters == 3
    assert assigner.embedding_dim == 2

    res = assigner.assign(np.array([0.9, 0.0]))
    assert res.cluster_id == 1

    res = assigner.assign(np.array([0.0, 0.95]))
    assert res.cluster_id == 2


def test_kmeans_round_trip_bit_for_bit(tmp_path: Path):
    """Save → load → assign on the same vectors gives identical assignments."""
    rng = np.random.default_rng(seed=42)
    centroids = rng.standard_normal((6, 16))
    embeddings = rng.standard_normal((50, 16))

    a = KMeansClusterAssigner(centroids)
    path = tmp_path / "ck.npz"
    a.save(path)

    b = load_cluster_assigner(path)
    assert isinstance(b, KMeansClusterAssigner)

    a_out = a.assign_batch(embeddings)
    b_out = b.assign_batch(embeddings)
    for ra, rb in zip(a_out, b_out):
        assert ra.cluster_id == rb.cluster_id
        assert np.array_equal(ra.probabilities, rb.probabilities)


def test_kmeans_assigner_rejects_non_2d():
    """1-D centroids raise."""
    with pytest.raises(ValueError):
        KMeansClusterAssigner(np.array([1.0, 2.0, 3.0]))


# --- Metrics ---


def test_metrics_smoke():
    """All metrics return finite floats in [0, 1]."""
    assert exact_match("a", "a") == 0.0
    assert exact_match("a", "b") == 1.0

    assert normalized_exact_match("Hello, world!", "hello world") == 0.0

    assert contains_match("the answer is 42", "42") == 0.0
    assert contains_match("foo", "bar") == 1.0

    assert 0.0 <= f1_score_loss("the cat", "the cat sat") <= 1.0

    # mmlu_match handles letter extraction
    assert mmlu_match("The answer is A", "A") == 0.0
    assert mmlu_match("(B)", "A") == 1.0


def test_metric_factory():
    """get_metric returns the right callable per MetricType."""
    fn = get_metric(MetricType.EXACT_MATCH)
    assert fn("hello", "hello") == 0.0
    assert fn("hello", "world") == 1.0


def test_compute_accuracy():
    """compute_accuracy = 1 - mean(losses)."""
    assert compute_accuracy([0.0, 0.0, 1.0, 1.0]) == 0.5
    assert compute_accuracy([]) == 0.0
