"""Integration tests for the full download-to-route pipeline.

Tests the real end-to-end flow using MockEmbeddingProvider to avoid
requiring SentenceTransformers at test time, and tests with the real
SentenceTransformerProvider when available.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from opentracy.core.embeddings import PromptEmbedder, MockEmbeddingProvider
from opentracy.core.clustering import KMeansClusterAssigner
from opentracy.models.llm_profile import LLMProfile
from opentracy.models.llm_registry import LLMRegistry
from opentracy.models.llm_client import MockLLMClient, create_client
from opentracy.router.uniroute import UniRouteRouter
from opentracy.data.dataset import PromptDataset, PromptSample
from opentracy.hub import Hub
from opentracy.weights import (
    download_weights,
    get_weights_path,
    list_available_weights,
    WEIGHTS_REGISTRY,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_test_weights(path: Path, num_clusters=10, embedding_dim=384, num_models=3):
    """Create a realistic weights directory structure for testing."""
    # Create clusters
    clusters_dir = path / "clusters"
    clusters_dir.mkdir(parents=True)
    centroids = np.random.randn(num_clusters, embedding_dim).astype(np.float32)
    np.savez(clusters_dir / "mmlu_full.npz", type="kmeans", centroids=centroids)

    # Create profiles
    profiles_dir = path / "profiles"
    profiles_dir.mkdir()
    models = [
        ("gpt-4o", 0.00625),
        ("gpt-4o-mini", 0.000375),
        ("mistral-small-latest", 0.0002),
    ][:num_models]

    for model_id, cost in models:
        np.random.seed(hash(model_id) % (2**31))
        psi = np.random.uniform(0.0, 0.5, num_clusters)
        counts = np.random.randint(10, 50, num_clusters)
        profile = LLMProfile(
            model_id=model_id,
            psi_vector=psi,
            cost_per_1k_tokens=cost,
            num_validation_samples=int(counts.sum()),
            cluster_sample_counts=counts,
            metadata={"provider": model_id.split("-")[0]},
        )
        safe_name = model_id.replace("/", "_")
        profile.save(profiles_dir / f"{safe_name}.json")

    return centroids, models


# ── Full pipeline with mock embeddings ────────────────────────────────────────

class TestFullPipelineMock:
    """End-to-end tests using MockEmbeddingProvider (no GPU/network needed)."""

    def test_load_router_from_weights_directory(self, tmp_path):
        """Simulate load_router by loading weights from a directory."""
        centroids, models = create_test_weights(tmp_path)

        # Load components (same steps as loader.py)
        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)

        cluster_path = tmp_path / "clusters" / "mmlu_full.npz"
        data = np.load(cluster_path, allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])

        registry = LLMRegistry()
        for profile_file in (tmp_path / "profiles").glob("*.json"):
            profile = LLMProfile.load(profile_file)
            registry.register(profile)

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
            cost_weight=0.0,
        )

        assert router.cluster_assigner.num_clusters == 10
        assert len(router.registry) == 3

    def test_route_multiple_prompts(self, tmp_path):
        """Route diverse prompts and verify all return valid decisions."""
        create_test_weights(tmp_path)

        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
        )

        prompts = [
            "What is 2 + 2?",
            "Explain quantum computing in simple terms",
            "Write a Python function to sort a list",
            "Translate hello to French",
            "What is the meaning of life?",
        ]

        model_ids = {p.model_id for p in registry.get_all()}
        for prompt in prompts:
            decision = router.route(prompt)
            assert decision.selected_model in model_ids
            assert 0 <= decision.cluster_id < 10
            assert decision.expected_error >= 0
            assert len(decision.all_scores) == 3

    def test_cost_weight_shifts_selection(self, tmp_path):
        """Verify that increasing cost_weight favors cheaper models."""
        num_clusters = 10

        # Create clusters
        clusters_dir = tmp_path / "clusters"
        clusters_dir.mkdir(parents=True)
        np.random.seed(42)
        centroids = np.random.randn(num_clusters, 384).astype(np.float32)
        np.savez(clusters_dir / "mmlu_full.npz", type="kmeans", centroids=centroids)

        # Create profiles with IDENTICAL error rates but DIFFERENT costs
        # This isolates the cost_weight effect
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        uniform_psi = np.full(num_clusters, 0.3)
        uniform_counts = np.full(num_clusters, 20)

        for model_id, cost in [("expensive-model", 0.01), ("cheap-model", 0.0001)]:
            profile = LLMProfile(
                model_id=model_id,
                psi_vector=uniform_psi.copy(),
                cost_per_1k_tokens=cost,
                num_validation_samples=200,
                cluster_sample_counts=uniform_counts.copy(),
            )
            profile.save(profiles_dir / f"{model_id}.json")

        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in profiles_dir.glob("*.json"):
            registry.register(LLMProfile.load(f))

        # With cost_weight=0, both models are tied on error — either can be selected
        router_quality = UniRouteRouter(
            embedder=embedder, cluster_assigner=assigner,
            registry=registry, cost_weight=0.0,
        )

        # With cost_weight > 0, cheap model must win (same error, lower cost)
        router_cost = UniRouteRouter(
            embedder=embedder, cluster_assigner=assigner,
            registry=registry, cost_weight=1.0,
        )

        prompt = "Explain machine learning"
        d_cost = router_cost.route(prompt)

        assert d_cost.selected_model == "cheap-model"

        # Verify the scores differ: cheap model's cost-adjusted score < expensive model's
        assert d_cost.all_scores["cheap-model"] < d_cost.all_scores["expensive-model"]

    def test_stats_accumulate_across_requests(self, tmp_path):
        """Verify stats tracking over multiple routing requests."""
        create_test_weights(tmp_path)

        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
        )

        prompts = [f"prompt_{i}" for i in range(20)]
        for p in prompts:
            router.route(p)

        stats = router.stats
        assert stats.total_requests == 20
        assert sum(stats.model_selections.values()) == 20
        assert sum(stats.cluster_distributions.values()) == 20
        assert 0 <= stats.avg_expected_error <= 1.0

    def test_allowed_models_restricts_routing(self, tmp_path):
        """Verify allowed_models constrains which model can be selected."""
        create_test_weights(tmp_path)

        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
            allowed_models=["gpt-4o-mini"],
        )

        for i in range(10):
            decision = router.route(f"test prompt {i}")
            assert decision.selected_model == "gpt-4o-mini"

    def test_profile_save_load_roundtrip(self, tmp_path):
        """Verify profiles survive a save/load cycle with no data loss."""
        create_test_weights(tmp_path)

        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        # Save to new directory
        out_dir = tmp_path / "profiles_copy"
        registry.save(out_dir)

        # Reload
        loaded = LLMRegistry.load(out_dir)
        assert len(loaded) == len(registry)
        for model_id in registry.get_model_ids():
            orig = registry.get(model_id)
            copy = loaded.get(model_id)
            assert copy is not None
            np.testing.assert_array_almost_equal(orig.psi_vector, copy.psi_vector)
            assert orig.cost_per_1k_tokens == copy.cost_per_1k_tokens

    def test_mock_client_in_full_flow(self, tmp_path):
        """Test mock client generation after routing decision."""
        create_test_weights(tmp_path)

        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
        )

        # Route, then call the selected model
        prompt = "What is the capital of France?"
        decision = router.route(prompt)
        client = create_client("mock", decision.selected_model)
        response = client.generate(prompt)

        assert response.model_id == decision.selected_model
        assert len(response.text) > 0
        assert response.tokens_used > 0
        assert response.latency_ms >= 0

    def test_dataset_to_routing_flow(self, tmp_path):
        """Test routing prompts loaded from a PromptDataset."""
        create_test_weights(tmp_path)

        provider = MockEmbeddingProvider(dimension=384)
        embedder = PromptEmbedder(provider, cache_enabled=True)
        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
        )

        dataset = PromptDataset([
            {"prompt": "What is 2+2?", "ground_truth": "4"},
            {"prompt": "Explain gravity", "ground_truth": "Force of attraction"},
            {"prompt": "Write a haiku", "ground_truth": "A poem"},
        ], name="test_set")

        results = []
        for prompt, _gt in dataset:
            decision = router.route(prompt)
            results.append(decision)

        assert len(results) == len(dataset)
        assert all(r.selected_model in registry.get_model_ids() for r in results)


# ── Hub integration ───────────────────────────────────────────────────────────

class TestHubIntegration:
    def test_install_verify_remove_cycle(self, tmp_path):
        """Test the full hub lifecycle: install -> verify -> info -> remove."""
        hub = Hub(data_home=tmp_path)

        # Create a fake installed package
        pkg_dir = tmp_path / "weights-mmlu-v1"
        pkg_dir.mkdir()
        (pkg_dir / "manifest.json").write_text(json.dumps({"package": {"id": "weights-mmlu-v1"}}))
        clusters_dir = pkg_dir / "clusters"
        clusters_dir.mkdir()
        profiles_dir = pkg_dir / "profiles"
        profiles_dir.mkdir()

        assert hub.is_installed("weights-mmlu-v1")
        assert hub.verify("weights-mmlu-v1")

        info = hub.info("weights-mmlu-v1")
        assert info is not None
        assert info["installed"] is True

        hub.remove("weights-mmlu-v1", quiet=True)
        assert not hub.is_installed("weights-mmlu-v1")
        assert not pkg_dir.exists()

    def test_weights_path_consistency(self):
        """get_weights_path aliases should resolve — with bundled weights
        shipped in the wheel, ``default`` is an alias for ``mmlu-v1`` and
        both land on the same on-disk bundle."""
        path_default = get_weights_path("default")
        path_mmlu = get_weights_path("mmlu-v1")

        # Both aliases must resolve to the bundled weights-mmlu-v1 directory.
        assert "weights-mmlu-v1" in str(path_default)
        assert "weights-mmlu-v1" in str(path_mmlu)
        assert path_default == path_mmlu


# ── Real SentenceTransformers (skipped if not installed) ──────────────────────

def _has_sentence_transformers():
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed",
)
class TestRealEmbeddings:
    """Integration tests with real SentenceTransformerProvider."""

    def test_real_embeddings_route(self, tmp_path):
        from opentracy.core.embeddings import SentenceTransformerProvider

        create_test_weights(tmp_path)

        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        embedder = PromptEmbedder(provider, cache_enabled=True)

        data = np.load(tmp_path / "clusters" / "mmlu_full.npz", allow_pickle=True)
        assigner = KMeansClusterAssigner(data["centroids"])
        registry = LLMRegistry()
        for f in (tmp_path / "profiles").glob("*.json"):
            registry.register(LLMProfile.load(f))

        router = UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
        )

        d1 = router.route("What is machine learning?")
        d2 = router.route("Write a sonnet about the moon")

        # Both should return valid decisions
        assert d1.selected_model in registry.get_model_ids()
        assert d2.selected_model in registry.get_model_ids()

        # Semantically different prompts may (but don't have to) get different clusters
        # At minimum, both should have valid cluster assignments
        assert 0 <= d1.cluster_id < 10
        assert 0 <= d2.cluster_id < 10

    def test_real_load_router_from_hub(self):
        """Test load_router with real weights from hub (if downloaded)."""
        from opentracy import load_router

        weights_path = get_weights_path("default")
        if not weights_path.exists():
            pytest.skip("Default weights not downloaded")

        # Force the Python backend so this test doesn't depend on the Go
        # binary being installed in the test environment. Backend selection
        # is covered elsewhere; this test is about weight loading + routing.
        router = load_router(weights_path=weights_path, verbose=False, engine="python")
        decision = router.route("Explain quantum computing")

        assert decision.selected_model is not None
        assert decision.expected_error >= 0
        assert decision.cluster_id >= 0
