"""Tests for UniRouteRouter and RoutingDecision."""

import numpy as np
import pytest

from lunar_router.router.uniroute import UniRouteRouter, RoutingDecision, RoutingStats
from lunar_router.core.clustering import KMeansClusterAssigner
from lunar_router.core.embeddings import PromptEmbedder
from lunar_router.models.llm_profile import LLMProfile
from lunar_router.models.llm_registry import LLMRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeEmbeddingProvider:
    """Returns deterministic embeddings for testing."""

    def __init__(self, dim=4):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(self._dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


def build_router(
    num_clusters=3,
    embedding_dim=4,
    cost_weight=0.0,
    models=None,
    use_soft=True,
):
    """Build a router with mock components."""
    if models is None:
        models = {
            "cheap": {"cost": 0.001, "errors": [0.5, 0.3, 0.7]},
            "expensive": {"cost": 0.01, "errors": [0.1, 0.1, 0.1]},
        }

    centroids = np.random.randn(num_clusters, embedding_dim)
    assigner = KMeansClusterAssigner(centroids)

    provider = FakeEmbeddingProvider(dim=embedding_dim)
    embedder = PromptEmbedder(provider, cache_enabled=False)

    registry = LLMRegistry()
    for model_id, info in models.items():
        profile = LLMProfile(
            model_id=model_id,
            psi_vector=np.array(info["errors"]),
            cost_per_1k_tokens=info["cost"],
            num_validation_samples=100,
            cluster_sample_counts=np.full(num_clusters, 20),
        )
        registry.register(profile)

    return UniRouteRouter(
        embedder=embedder,
        cluster_assigner=assigner,
        registry=registry,
        cost_weight=cost_weight,
        use_soft_assignment=use_soft,
    )


# ── RoutingDecision ───────────────────────────────────────────────────────────

class TestRoutingDecision:
    def test_to_dict(self):
        d = RoutingDecision(
            selected_model="test",
            expected_error=0.1,
            cost_adjusted_score=0.15,
            all_scores={"test": 0.15},
            cluster_id=2,
            cluster_probabilities=np.array([0.0, 0.0, 1.0]),
        )
        result = d.to_dict()
        assert result["selected_model"] == "test"
        assert result["expected_error"] == 0.1
        assert result["cluster_id"] == 2
        assert isinstance(result["cluster_probabilities"], list)


# ── RoutingStats ──────────────────────────────────────────────────────────────

class TestRoutingStats:
    def test_initial_state(self):
        stats = RoutingStats()
        assert stats.total_requests == 0
        assert stats.model_selections == {}

    def test_update(self):
        stats = RoutingStats()
        decision = RoutingDecision(
            selected_model="model-a",
            expected_error=0.2,
            cost_adjusted_score=0.3,
            all_scores={"model-a": 0.3},
            cluster_id=1,
            cluster_probabilities=np.array([0.0, 1.0, 0.0]),
        )
        stats.update(decision)
        assert stats.total_requests == 1
        assert stats.model_selections["model-a"] == 1
        assert stats.cluster_distributions[1] == 1
        assert abs(stats.avg_expected_error - 0.2) < 1e-6

    def test_multiple_updates(self):
        stats = RoutingStats()
        for i in range(3):
            d = RoutingDecision(
                selected_model="m",
                expected_error=0.1 * (i + 1),
                cost_adjusted_score=0.1 * (i + 1),
                all_scores={"m": 0.1},
                cluster_id=0,
                cluster_probabilities=np.array([1.0]),
            )
            stats.update(d)
        assert stats.total_requests == 3
        assert stats.model_selections["m"] == 3
        assert abs(stats.avg_expected_error - 0.2) < 1e-6  # (0.1+0.2+0.3)/3


# ── UniRouteRouter ────────────────────────────────────────────────────────────

class TestUniRouteRouter:
    def test_route_returns_decision(self):
        router = build_router()
        decision = router.route("test prompt")
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model in ("cheap", "expensive")
        assert decision.expected_error >= 0
        assert decision.cluster_id >= 0

    def test_route_quality_mode(self):
        """With cost_weight=0, should prefer the model with lowest error."""
        router = build_router(cost_weight=0.0)
        decision = router.route("any prompt")
        # expensive model has uniform 0.1 error vs cheap's higher errors
        assert decision.selected_model == "expensive"

    def test_route_cost_mode(self):
        """With high cost_weight, should prefer cheaper model."""
        router = build_router(cost_weight=100.0)
        decision = router.route("any prompt")
        assert decision.selected_model == "cheap"

    def test_cost_weight_override(self):
        router = build_router(cost_weight=0.0)
        # Default should pick expensive (better quality)
        d1 = router.route("prompt")
        assert d1.selected_model == "expensive"
        # Override with high cost weight should pick cheap
        d2 = router.route("prompt", cost_weight_override=100.0)
        assert d2.selected_model == "cheap"

    def test_stats_tracked(self):
        router = build_router()
        assert router.stats.total_requests == 0
        router.route("test")
        assert router.stats.total_requests == 1
        router.route("test2")
        assert router.stats.total_requests == 2

    def test_reset_stats(self):
        router = build_router()
        router.route("test")
        router.reset_stats()
        assert router.stats.total_requests == 0

    def test_hard_assignment(self):
        router = build_router(use_soft=False)
        decision = router.route("test")
        # With hard assignment, cluster_probabilities should be one-hot
        assert decision.cluster_probabilities.sum() == 1.0
        assert np.max(decision.cluster_probabilities) == 1.0

    def test_all_scores_populated(self):
        router = build_router()
        decision = router.route("test")
        assert "cheap" in decision.all_scores
        assert "expensive" in decision.all_scores

    def test_available_models_filter(self):
        router = build_router()
        decision = router.route("test", available_models=["cheap"])
        assert decision.selected_model == "cheap"

    def test_no_models_raises(self):
        router = build_router()
        with pytest.raises(ValueError, match="No models available"):
            router.route("test", available_models=["nonexistent"])

    def test_allowed_models_constructor(self):
        router = build_router()
        restricted = UniRouteRouter(
            embedder=router.embedder,
            cluster_assigner=router.cluster_assigner,
            registry=router.registry,
            cost_weight=0.0,
            allowed_models=["cheap"],
        )
        decision = restricted.route("test")
        assert decision.selected_model == "cheap"

    def test_invalid_allowed_models_raises(self):
        router = build_router()
        with pytest.raises(ValueError, match="Models not found"):
            UniRouteRouter(
                embedder=router.embedder,
                cluster_assigner=router.cluster_assigner,
                registry=router.registry,
                allowed_models=["nonexistent"],
            )
