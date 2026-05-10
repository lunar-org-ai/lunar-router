"""Decision-math tests for UniRouteRouter.

Uses FakeEmbedder + FakeAssigner so tests stay fast (no model download).
"""

from __future__ import annotations

import numpy as np
import pytest

from router.core.clustering import ClusterAssigner, ClusterResult
from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder
from router.errors import RouterColdStartError
from router.models.llm_profile import LLMProfile
from router.models.llm_registry import LLMRegistry
from router.uniroute import RoutingStats, UniRouteRouter


# --- Fakes ---


class FakeAssigner(ClusterAssigner):
    """Returns the same canned ClusterResult for every embedding."""

    def __init__(self, probabilities: np.ndarray):
        self._probs = np.asarray(probabilities)
        self._cluster_id = int(np.argmax(self._probs))

    @property
    def num_clusters(self) -> int:
        return len(self._probs)

    def assign(self, embedding: np.ndarray) -> ClusterResult:
        return ClusterResult(cluster_id=self._cluster_id, probabilities=self._probs.copy())

    def save(self, path):  # pragma: no cover — not used in tests
        raise NotImplementedError

    @classmethod
    def load(cls, path):  # pragma: no cover
        raise NotImplementedError


def _make_profile(model_id: str, psi: tuple[float, ...], cost: float) -> LLMProfile:
    psi_arr = np.array(psi, dtype=float)
    return LLMProfile(
        model_id=model_id,
        psi_vector=psi_arr,
        cost_per_1k_tokens=cost,
        num_validation_samples=int(len(psi) * 10),
        cluster_sample_counts=np.full(len(psi), 10, dtype=int),
    )


def _embedder() -> PromptEmbedder:
    return PromptEmbedder(MockEmbeddingProvider(dimension=8), cache_enabled=False)


def _registry(*profiles: LLMProfile) -> LLMRegistry:
    reg = LLMRegistry()
    for p in profiles:
        reg.register(p)
    return reg


# --- Construction guards ---


def test_cold_start_when_registry_empty():
    """Empty registry → RouterColdStartError on construction."""
    with pytest.raises(RouterColdStartError):
        UniRouteRouter(
            embedder=_embedder(),
            cluster_assigner=FakeAssigner(np.array([1.0, 0.0, 0.0])),
            registry=LLMRegistry(),
        )


def test_cold_start_when_assigner_has_zero_clusters():
    """Assigner with K=0 → RouterColdStartError."""

    class EmptyAssigner(FakeAssigner):
        @property
        def num_clusters(self) -> int:
            return 0

    reg = _registry(_make_profile("a", (0.1, 0.2, 0.3), 0.001))
    with pytest.raises(RouterColdStartError):
        UniRouteRouter(
            embedder=_embedder(),
            cluster_assigner=EmptyAssigner(np.array([1.0, 0.0, 0.0])),
            registry=reg,
        )


def test_construction_validates_allowed_models():
    """allowed_models containing unknown IDs raises ValueError."""
    reg = _registry(_make_profile("a", (0.1, 0.2, 0.3), 0.001))
    with pytest.raises(ValueError):
        UniRouteRouter(
            embedder=_embedder(),
            cluster_assigner=FakeAssigner(np.array([1.0, 0.0, 0.0])),
            registry=reg,
            allowed_models=["a", "ghost-model"],
        )


# --- Decision math ---


def test_route_picks_lowest_error_at_zero_lambda():
    """With cost_weight=0, picks the model with lowest expected error."""
    a = _make_profile("a", (0.1, 0.5, 0.3), cost=0.01)   # cluster 0 error 0.1
    b = _make_profile("b", (0.5, 0.1, 0.3), cost=0.001)  # cluster 0 error 0.5
    reg = _registry(a, b)

    # Hard cluster 0 → A wins
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0, 0.0])),
        registry=reg,
        cost_weight=0.0,
    )
    decision = router.route("anything")
    assert decision.selected_model == "a"
    assert decision.expected_error == pytest.approx(0.1)
    assert decision.cluster_id == 0

    # Hard cluster 1 → B wins
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([0.0, 1.0, 0.0])),
        registry=reg,
        cost_weight=0.0,
    )
    decision = router.route("anything")
    assert decision.selected_model == "b"
    assert decision.expected_error == pytest.approx(0.1)


def test_route_factors_cost_at_high_lambda():
    """High cost_weight tips toward cheaper model even when accuracy is worse."""
    # A is cheaper but worse on cluster 0; B is better but expensive.
    a = _make_profile("a", (0.20, 0.20, 0.20), cost=0.0)   # always 0.2 error, free
    b = _make_profile("b", (0.10, 0.10, 0.10), cost=10.0)  # always 0.1 error, expensive
    reg = _registry(a, b)

    # λ=0 → B wins (lower error).
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0, 0.0])),
        registry=reg,
        cost_weight=0.0,
    )
    assert router.route("x").selected_model == "b"

    # λ=1 → score_a = 0.2 + 0; score_b = 0.1 + 10 → A wins.
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0, 0.0])),
        registry=reg,
        cost_weight=1.0,
    )
    assert router.route("x").selected_model == "a"


def test_route_uses_soft_assignment_by_default():
    """Soft phi blends per-cluster errors before argmin."""
    # A: low on cluster 0, high on cluster 1; B: opposite.
    a = _make_profile("a", (0.1, 0.9, 0.5), cost=0.0)
    b = _make_profile("b", (0.9, 0.1, 0.5), cost=0.0)
    reg = _registry(a, b)

    # phi = [0.6, 0.4]: A's expected = 0.6*0.1 + 0.4*0.9 = 0.42
    #                   B's expected = 0.6*0.9 + 0.4*0.1 = 0.58
    # → A wins under soft assignment.
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([0.6, 0.4, 0.0])),
        registry=reg,
        use_soft_assignment=True,
        cost_weight=0.0,
    )
    decision = router.route("x")
    assert decision.selected_model == "a"
    assert decision.expected_error == pytest.approx(0.42)


def test_route_hard_assignment_uses_one_hot():
    """use_soft_assignment=False collapses phi to one-hot of dominant cluster."""
    a = _make_profile("a", (0.1, 0.9), cost=0.0)
    b = _make_profile("b", (0.9, 0.1), cost=0.0)
    reg = _registry(a, b)

    # Soft phi=[0.51, 0.49] dominant=0 → one-hot [1, 0]
    # Hard: A error = 0.1; B error = 0.9 → A wins clearly.
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([0.51, 0.49])),
        registry=reg,
        use_soft_assignment=False,
        cost_weight=0.0,
    )
    decision = router.route("x")
    assert decision.selected_model == "a"
    assert decision.expected_error == pytest.approx(0.1)


def test_route_respects_allowed_models_argument():
    """available_models filters even when A would otherwise win."""
    a = _make_profile("a", (0.1, 0.5), cost=0.0)
    b = _make_profile("b", (0.5, 0.1), cost=0.0)
    reg = _registry(a, b)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0])),
        registry=reg,
    )
    # A would win without the filter; restricting to [b] forces B.
    decision = router.route("x", available_models=["b"])
    assert decision.selected_model == "b"


def test_route_raises_when_no_models_after_filter():
    """available_models referencing only unknown IDs leaves zero profiles → ValueError."""
    a = _make_profile("a", (0.1,), cost=0.0)
    reg = _registry(a)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0])),
        registry=reg,
    )
    with pytest.raises(ValueError):
        router.route("x", available_models=["nope"])


# --- Stats / batch / distribution ---


def test_routing_stats_update_on_each_decision():
    """Sequential decisions update model_selections + cluster_distributions."""
    a = _make_profile("a", (0.1, 0.5), cost=0.0)
    b = _make_profile("b", (0.5, 0.1), cost=0.0)
    reg = _registry(a, b)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0])),
        registry=reg,
    )
    router.route("p1")
    router.route("p2")
    stats = router.stats
    assert isinstance(stats, RoutingStats)
    assert stats.total_requests == 2
    assert stats.model_selections.get("a") == 2
    assert stats.cluster_distributions.get(0) == 2

    router.reset_stats()
    assert router.stats.total_requests == 0


def test_get_best_model_for_cluster_matches_route():
    """For hard phi=cluster c, get_best_model_for_cluster(c) matches route()."""
    a = _make_profile("a", (0.1, 0.9), cost=0.0)
    b = _make_profile("b", (0.9, 0.1), cost=0.0)
    reg = _registry(a, b)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0])),
        registry=reg,
    )
    assert router.get_best_model_for_cluster(0) == "a"
    assert router.get_best_model_for_cluster(1) == "b"


def test_route_batch_returns_one_decision_per_prompt():
    """route_batch is just a loop over route() — one decision per input."""
    a = _make_profile("a", (0.1, 0.9), cost=0.0)
    reg = _registry(a)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0])),
        registry=reg,
    )
    decisions = router.route_batch(["a", "b", "c"])
    assert len(decisions) == 3
    assert all(d.selected_model == "a" for d in decisions)


def test_analyze_routing_distribution_smoke():
    """analyze_routing_distribution returns dict with expected keys + sums."""
    a = _make_profile("a", (0.1, 0.5), cost=0.0)
    b = _make_profile("b", (0.5, 0.1), cost=0.0)
    reg = _registry(a, b)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0])),
        registry=reg,
    )
    dist = router.analyze_routing_distribution(["x", "y", "z"])
    assert dist["num_prompts"] == 3
    assert sum(dist["model_counts"].values()) == 3
    assert pytest.approx(sum(dist["model_distribution"].values()), abs=1e-9) == 1.0
    assert dist["avg_expected_error"] >= 0


def test_routing_decision_to_dict_round_trip_keys():
    """to_dict has the keys the API response model expects."""
    a = _make_profile("a", (0.1, 0.5), cost=0.0)
    reg = _registry(a)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=FakeAssigner(np.array([1.0, 0.0])),
        registry=reg,
    )
    decision = router.route("x")
    d = decision.to_dict()
    expected_keys = {
        "selected_model",
        "expected_error",
        "cost_adjusted_score",
        "all_scores",
        "cluster_id",
        "cluster_probabilities",
        "reasoning",
    }
    assert expected_keys.issubset(d.keys())
    # cluster_probabilities is JSON-friendly (list, not numpy).
    assert isinstance(d["cluster_probabilities"], list)
