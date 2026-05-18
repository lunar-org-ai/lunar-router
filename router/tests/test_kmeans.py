"""Tests for router/training/{gate,kmeans,snapshot}.

All tests use a synthetic-Gaussian FakeEmbedder so we never download
MiniLM. Round-trip tests verify the bit-for-bit save/load promise from
the PLAN's DoD.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from router.core.embeddings import PromptEmbedder
from router.errors import KMeansFitError, NotEnoughDataError
from router.training.gate import check_first_fit_eligibility
from router.training.kmeans import (
    KMeansPlusPlusInitializer,
    KMeansTrainer,
    analyze_clusters,
)
from router.training.result import KMeansTrainResult
from router.training.snapshot import snapshot_clusters_only


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _GaussianBlobsProvider:
    """Embedding provider that maps each prompt to a draw from one of K
    well-separated Gaussian blobs. Deterministic by prompt text via hash.
    """

    model_name = "test-gaussian-blobs"

    def __init__(self, blob_centers: np.ndarray, sigma: float = 0.05):
        self._centers = np.asarray(blob_centers, dtype=float)
        self._sigma = sigma
        self._dim = self._centers.shape[1]

    @property
    def dimension(self) -> int:
        return self._dim

    def _blob_for(self, text: str) -> int:
        # Stable mapping: hash text → blob index. Tests can also force a
        # specific blob with a "blob:<idx>::<extra>" prefix.
        if text.startswith("blob:"):
            try:
                idx = int(text.split(":", 2)[1])
                return idx % len(self._centers)
            except ValueError:
                pass
        return abs(hash(text)) % len(self._centers)

    def embed(self, text: str) -> np.ndarray:
        idx = self._blob_for(text)
        # Deterministic noise draw per text.
        rng = np.random.default_rng(abs(hash(("g", text))) & 0xFFFFFFFF)
        noise = rng.normal(0.0, self._sigma, size=self._dim)
        return self._centers[idx] + noise

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.embed(t) for t in texts])


def _embedder_with_blobs(k: int, dim: int = 16, sigma: float = 0.05) -> PromptEmbedder:
    rng = np.random.default_rng(seed=42)
    centers = rng.standard_normal((k, dim)) * 5.0  # well-separated
    provider = _GaussianBlobsProvider(centers, sigma=sigma)
    return PromptEmbedder(provider, cache_enabled=False)


def _prompts_across_blobs(k: int, per_blob: int) -> list[str]:
    return [f"blob:{i}::p{j}" for i in range(k) for j in range(per_blob)]


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_blocks_under_min():
    eligible, reason = check_first_fit_eligibility(corpus_size=100, min_corpus_size=200)
    assert not eligible
    assert "100" in reason and "200" in reason


def test_gate_blocks_when_n_lt_2k():
    eligible, reason = check_first_fit_eligibility(
        corpus_size=200, min_corpus_size=200, requested_k=120
    )
    assert not eligible
    assert "240" in reason or "120" in reason  # 2*K=240 in message


def test_gate_passes_at_threshold():
    eligible, reason = check_first_fit_eligibility(
        corpus_size=200, min_corpus_size=200, requested_k=8
    )
    assert eligible
    assert reason == "ok"


def test_gate_rejects_k_below_2():
    eligible, reason = check_first_fit_eligibility(
        corpus_size=500, min_corpus_size=200, requested_k=1
    )
    assert not eligible
    assert "K=1" in reason or "< 2" in reason


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_basic_returns_result_with_high_silhouette():
    """3 well-separated blobs × 80 prompts each → silhouette > 0.5 (clear)."""
    embedder = _embedder_with_blobs(k=3, dim=16)
    prompts = _prompts_across_blobs(k=3, per_blob=80)  # 240 prompts ≥ min(200) and ≥ 2*K
    trainer = KMeansTrainer(embedder, num_clusters=3)

    result = trainer.train(
        prompts,
        fitted_from={"source": "synthetic", "blobs": 3},
    )
    assert isinstance(result, KMeansTrainResult)
    assert result.k == 3
    assert result.n_samples == 240
    assert result.silhouette > 0.5
    assert result.inertia > 0
    assert sum(result.cluster_sizes.values()) == 240
    assert result.embedder_model_id == "test-gaussian-blobs"
    assert result.fitted_from == {"source": "synthetic", "blobs": 3}
    assert result.fitted_at.endswith("Z")


def test_train_blocks_below_min_corpus_size():
    embedder = _embedder_with_blobs(k=3, dim=16)
    prompts = _prompts_across_blobs(k=3, per_blob=10)  # 30 < 200
    trainer = KMeansTrainer(embedder, num_clusters=3)
    with pytest.raises(NotEnoughDataError):
        trainer.train(prompts, fitted_from={"source": "synthetic"})


def test_train_blocks_when_n_lt_2k():
    embedder = _embedder_with_blobs(k=3, dim=16)
    prompts = _prompts_across_blobs(k=3, per_blob=70)  # 210 ≥ 200
    trainer = KMeansTrainer(embedder, num_clusters=120)  # 2*K=240 > 210
    with pytest.raises(NotEnoughDataError):
        trainer.train(prompts, fitted_from={"source": "synthetic"})


def test_train_logs_silhouette(caplog):
    embedder = _embedder_with_blobs(k=3, dim=16)
    prompts = _prompts_across_blobs(k=3, per_blob=80)
    trainer = KMeansTrainer(embedder, num_clusters=3)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="router.training.kmeans"):
        trainer.train(prompts, fitted_from={"source": "synthetic"})
    messages = [r.message for r in caplog.records]
    assert any("fit start" in m for m in messages)
    assert any("silhouette" in m for m in messages)


def test_train_round_trip_bit_for_bit(tmp_path: Path):
    """Train → save → load → assign N vectors gives identical assignments."""
    embedder = _embedder_with_blobs(k=4, dim=8)
    prompts = _prompts_across_blobs(k=4, per_blob=60)  # 240
    trainer = KMeansTrainer(embedder, num_clusters=4)
    result = trainer.train(prompts, fitted_from={"source": "synthetic"})

    save_path = tmp_path / "centroids.npz"
    result.assigner.save(save_path)

    from router.core.clustering import KMeansClusterAssigner

    loaded = KMeansClusterAssigner.load(save_path)

    # Use a fresh batch of test embeddings.
    rng = np.random.default_rng(seed=99)
    test_embeddings = rng.standard_normal((60, 8))

    a = result.assigner.assign_batch(test_embeddings)
    b = loaded.assign_batch(test_embeddings)
    for ra, rb in zip(a, b):
        assert ra.cluster_id == rb.cluster_id
        assert np.array_equal(ra.probabilities, rb.probabilities)


def test_train_summary_format():
    embedder = _embedder_with_blobs(k=3, dim=8)
    prompts = _prompts_across_blobs(k=3, per_blob=80)
    trainer = KMeansTrainer(embedder, num_clusters=3)
    result = trainer.train(prompts, fitted_from={"source": "synthetic"})
    summary = result.summary()
    assert "K=3" in summary and "N=240" in summary
    assert "silhouette" in summary and "inertia" in summary


# ---------------------------------------------------------------------------
# train_with_validation()
# ---------------------------------------------------------------------------


def test_train_with_validation_picks_best_k():
    """4 well-separated blobs × 80 train + 30 val each → best K = 4."""
    embedder = _embedder_with_blobs(k=4, dim=16)
    train_prompts = _prompts_across_blobs(k=4, per_blob=80)  # 320
    val_prompts = [f"blob:{i}::v{j}" for i in range(4) for j in range(30)]  # 120
    trainer = KMeansTrainer(embedder, num_clusters=2)  # ignored — sweep overrides

    result = trainer.train_with_validation(
        train_prompts,
        val_prompts,
        k_values=[2, 3, 4, 5, 6],
        fitted_from={"source": "synthetic"},
    )
    assert result.k == 4
    # Trainer's num_clusters synced to the chosen K.
    assert trainer.num_clusters == 4


def test_train_with_validation_filters_invalid_k_values():
    """K candidates < 2 or > N/2 are filtered before sweeping."""
    embedder = _embedder_with_blobs(k=3, dim=8)
    train_prompts = _prompts_across_blobs(k=3, per_blob=70)  # 210
    val_prompts = [f"blob:{i}::v{j}" for i in range(3) for j in range(20)]
    trainer = KMeansTrainer(embedder, num_clusters=3)

    # K=1 invalid (silhouette undefined); K=200 invalid (2*K > N=210).
    # K=3 valid → must succeed and pick K=3.
    result = trainer.train_with_validation(
        train_prompts, val_prompts, k_values=[1, 3, 200],
        fitted_from={"source": "synthetic"},
    )
    assert result.k == 3


def test_train_with_validation_raises_when_no_eligible_k():
    embedder = _embedder_with_blobs(k=3, dim=8)
    train_prompts = _prompts_across_blobs(k=3, per_blob=70)  # 210
    val_prompts = train_prompts[:60]
    trainer = KMeansTrainer(embedder, num_clusters=3)
    with pytest.raises(NotEnoughDataError):
        trainer.train_with_validation(
            train_prompts, val_prompts, k_values=[1, 200, 500],
            fitted_from={"source": "synthetic"},
        )


# ---------------------------------------------------------------------------
# analyze_clusters
# ---------------------------------------------------------------------------


def test_analyze_clusters_shape():
    embedder = _embedder_with_blobs(k=3, dim=8)
    prompts = _prompts_across_blobs(k=3, per_blob=80)
    trainer = KMeansTrainer(embedder, num_clusters=3)
    result = trainer.train(prompts, fitted_from={"source": "synthetic"})

    stats = analyze_clusters(prompts, result.assigner, embedder, top_n=2)
    assert stats["num_clusters"] == 3
    assert stats["num_samples"] == 240
    assert sum(stats["cluster_sizes"].values()) == 240
    for examples in stats["cluster_examples"].values():
        assert len(examples) <= 2
    assert "size_stats" in stats and "min" in stats["size_stats"]


# ---------------------------------------------------------------------------
# snapshot_clusters_only
# ---------------------------------------------------------------------------


def test_snapshot_writes_partial_config(tmp_path: Path, monkeypatch):
    """Snapshot writes router_config_v1.json with model_psi={} + sidecar
    centroids; current pointer is NOT updated."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)

    embedder = _embedder_with_blobs(k=3, dim=8)
    prompts = _prompts_across_blobs(k=3, per_blob=80)
    trainer = KMeansTrainer(embedder, num_clusters=3)
    result = trainer.train(prompts, fitted_from={"source": "synthetic"})

    path = snapshot_clusters_only(result, versions_dir=tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text())

    assert payload["version"] == 1
    assert payload["k"] == 3
    assert payload["model_psi"] == {}
    assert payload["cost_weight"] == 0.0
    assert payload["embedder_model"] == "test-gaussian-blobs"
    assert payload["embedding_dim"] == 8
    assert payload["fitted_from"] == {"source": "synthetic"}
    assert payload["metadata"]["stage"] == "clusters_only"

    # Sidecar centroids exist + match.
    npz_path = tmp_path / "router_config_v1_centroids.npz"
    assert npz_path.exists()
    data = np.load(npz_path)
    assert np.array_equal(data["centroids"], result.assigner.centroids)

    # Current pointer NOT updated — cold-start still in effect.
    assert not (tmp_path / "router_config_current").exists()
    assert not (tmp_path / "router_config_current.txt").exists()


def test_snapshot_bumps_version_relative_to_current(tmp_path: Path, monkeypatch):
    """When a current pointer exists at v3, snapshot writes v4."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    # Seed a v3 + pointer (.txt fallback to be portable).
    (tmp_path / "router_config_v3.json").write_text(
        json.dumps({"version": 3, "k": 0, "model_psi": {}, "cost_weight": 0.0})
    )
    (tmp_path / "router_config_current.txt").write_text("3")

    embedder = _embedder_with_blobs(k=2, dim=8)
    prompts = _prompts_across_blobs(k=2, per_blob=120)
    trainer = KMeansTrainer(embedder, num_clusters=2)
    result = trainer.train(prompts, fitted_from={"source": "synthetic"})

    path = snapshot_clusters_only(result, versions_dir=tmp_path)
    assert path.name == "router_config_v4.json"
    payload = json.loads(path.read_text())
    assert payload["version"] == 4


# ---------------------------------------------------------------------------
# KMeansPlusPlusInitializer (parity with reference)
# ---------------------------------------------------------------------------


def test_kmeanspp_initializer_returns_k_distinct_centroids():
    rng = np.random.default_rng(seed=7)
    embeddings = rng.standard_normal((100, 8))
    centroids = KMeansPlusPlusInitializer.initialize(embeddings, k=4, random_state=42)
    assert centroids.shape == (4, 8)
    # No two centroids are identical.
    seen = {tuple(c.tolist()) for c in centroids}
    assert len(seen) == 4
