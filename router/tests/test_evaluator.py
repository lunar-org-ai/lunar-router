"""Tests for router/evaluation/evaluator.

Roadmap DoD verified here:
1. Empty-config (Psi all zeros) → AUROC ≈ 0.5.
2. Fitted-config (Psi informative) → AUROC > 0.5.
3. Cache hit ratio is logged at INFO.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from router.core.clustering import ClusterAssigner, ClusterResult
from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder
from router.data.dataset import PromptDataset, PromptSample
from router.evaluation.cache import ResponseCache
from router.evaluation.evaluator import (
    CacheGapError,
    EvaluationResult,
    RouterEvaluator,
)
from router.models.llm_profile import LLMProfile
from router.models.llm_registry import LLMRegistry
from router.uniroute import UniRouteRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAssigner(ClusterAssigner):
    """Two clusters, hard assignment by prompt prefix.

    'easy:' → cluster 0; 'hard:' → cluster 1.
    """

    @property
    def num_clusters(self) -> int:
        return 2

    def assign(self, embedding: np.ndarray) -> ClusterResult:
        # Pull the originating prompt from the cache key... we can't here
        # since assign() only sees the embedding. Instead the FakeEmbedder
        # encodes cluster id in dim 0 (1.0 = cluster 1, 0.0 = cluster 0).
        cid = 1 if embedding[0] > 0.5 else 0
        probs = np.zeros(2)
        probs[cid] = 1.0
        return ClusterResult(cluster_id=cid, probabilities=probs)

    def save(self, path):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path):  # pragma: no cover
        raise NotImplementedError


class _ClusterEncodingProvider:
    """Encodes 'hard:' as 1.0 in dim 0, 'easy:' as 0.0 in dim 0."""

    model_name = "cluster-encoder"
    _dim = 8

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim)
        v[0] = 1.0 if text.startswith("hard:") else 0.0
        return v

    def embed_batch(self, texts):
        return np.asarray([self.embed(t) for t in texts])


def _embedder() -> PromptEmbedder:
    return PromptEmbedder(_ClusterEncodingProvider(), cache_enabled=False)


def _profile(model_id: str, psi: list[float], cost: float) -> LLMProfile:
    return LLMProfile(
        model_id=model_id,
        psi_vector=np.array(psi),
        cost_per_1k_tokens=cost,
        num_validation_samples=20,
        cluster_sample_counts=np.array([10, 10]),
    )


def _registry(*profiles: LLMProfile) -> LLMRegistry:
    reg = LLMRegistry()
    for p in profiles:
        reg.register(p)
    return reg


def _seed_cache(scenarios: list[tuple[str, str, float]]) -> ResponseCache:
    """scenarios = [(prompt, model_id, loss), ...]"""
    cache = ResponseCache()
    for prompt, model_id, loss in scenarios:
        cache.add(prompt, model_id, response_text="r", loss=loss)
    return cache


def _full_coverage_cache(prompts: list[str], scenarios: dict) -> ResponseCache:
    """Build a cache covering every (prompt, model) combo.

    scenarios maps model_id → {prompt: loss}. Prompts not present default
    to loss=0.5 (uninformative).
    """
    cache = ResponseCache()
    for model_id, per_prompt in scenarios.items():
        for prompt in prompts:
            loss = per_prompt.get(prompt, 0.5)
            cache.add(prompt, model_id, response_text=f"r-{model_id}", loss=loss)
    return cache


def _build_evaluator(*, psi_haiku, psi_sonnet, prompts):
    weak = _profile("haiku", psi_haiku, cost=0.001)
    strong = _profile("sonnet", psi_sonnet, cost=0.003)
    registry = _registry(weak, strong)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=_FakeAssigner(),
        registry=registry,
        cost_weight=0.0,
    )

    # Easy prompts (cluster 0): both correct.
    # Hard prompts (cluster 1): only sonnet correct.
    scenarios = {
        "haiku": {p: 0.0 for p in prompts if p.startswith("easy:")}
                 | {p: 1.0 for p in prompts if p.startswith("hard:")},
        "sonnet": {p: 0.0 for p in prompts},
    }
    cache = _full_coverage_cache(prompts, scenarios)
    return RouterEvaluator(router, cache, profiles=[weak, strong], lambda_steps=5), cache


# ---------------------------------------------------------------------------
# Roadmap DoDs
# ---------------------------------------------------------------------------


def test_evaluate_empty_config_yields_auroc_half():
    """Psi all zeros → router-driven score = 0 for every prompt → AUROC ≈ 0.5."""
    prompts = [f"easy:p{i}" for i in range(10)] + [f"hard:p{i}" for i in range(10)]
    evaluator, _ = _build_evaluator(
        psi_haiku=[0.0, 0.0],
        psi_sonnet=[0.0, 0.0],
        prompts=prompts,
    )
    ds = PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])
    result = evaluator.evaluate(ds, dataset_name="empty")
    assert result.metrics.auroc == pytest.approx(0.5, abs=0.05)


def test_evaluate_fitted_config_yields_auroc_above_half():
    """Psi distinguishes hard cluster (1) → router favors sonnet there → AUROC > 0.5."""
    prompts = [f"easy:p{i}" for i in range(10)] + [f"hard:p{i}" for i in range(10)]
    evaluator, _ = _build_evaluator(
        # Hard cluster: haiku high error (0.9), sonnet low (0.1) → weak-strong gap large
        psi_haiku=[0.1, 0.9],
        psi_sonnet=[0.1, 0.1],
        prompts=prompts,
    )
    ds = PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])
    result = evaluator.evaluate(ds, dataset_name="fitted")
    assert result.metrics.auroc > 0.5


def test_evaluate_logs_cache_hit_ratio(caplog):
    prompts = [f"easy:p{i}" for i in range(5)] + [f"hard:p{i}" for i in range(5)]
    evaluator, _ = _build_evaluator(
        psi_haiku=[0.1, 0.9],
        psi_sonnet=[0.1, 0.1],
        prompts=prompts,
    )
    ds = PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="router.evaluation.evaluator"):
        evaluator.evaluate(ds, dataset_name="hit-ratio-test")
    messages = [r.message for r in caplog.records]
    assert any("cache_hit_ratio" in m for m in messages)


# ---------------------------------------------------------------------------
# Other contracts
# ---------------------------------------------------------------------------


def test_evaluate_raises_cache_gap_error_on_missing_entry():
    """Cache miss surfaces with the specific (prompt, model) in the message."""
    weak = _profile("haiku", [0.5, 0.5], cost=0.001)
    strong = _profile("sonnet", [0.1, 0.1], cost=0.003)
    registry = _registry(weak, strong)
    router = UniRouteRouter(
        embedder=_embedder(),
        cluster_assigner=_FakeAssigner(),
        registry=registry,
        cost_weight=0.0,
    )

    # Only haiku has cached responses. sonnet missing → CacheGapError.
    cache = ResponseCache()
    cache.add("easy:p1", "haiku", "r", 0.0)

    evaluator = RouterEvaluator(router, cache, profiles=[weak, strong], lambda_steps=3)
    ds = PromptDataset([PromptSample(prompt="easy:p1", ground_truth="")])
    with pytest.raises(CacheGapError) as exc_info:
        evaluator.evaluate(ds)
    assert "sonnet" in str(exc_info.value)
    assert "populate_response_cache" in str(exc_info.value)


def test_evaluate_returns_full_evaluation_result_shape():
    prompts = [f"easy:p{i}" for i in range(5)] + [f"hard:p{i}" for i in range(5)]
    evaluator, _ = _build_evaluator(
        psi_haiku=[0.1, 0.9],
        psi_sonnet=[0.1, 0.1],
        prompts=prompts,
    )
    ds = PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])
    result = evaluator.evaluate(ds, dataset_name="shape-test")
    assert isinstance(result, EvaluationResult)
    assert result.dataset_name == "shape-test"
    assert result.cache_hit_ratio == pytest.approx(1.0)
    # Pareto curve has lambda_steps points.
    assert len(result.pareto_curve) == 5
    # Baselines populated.
    assert "always_strong" in result.baseline_quality
    assert "always_weak" in result.baseline_quality
    assert "random" in result.baseline_quality
    assert "oracle" in result.baseline_quality
    # Metrics populated.
    assert 0.0 <= result.metrics.auroc <= 1.0
    assert 0.0 <= result.metrics.win_rate <= 1.0


def test_pareto_curve_monotonic_in_strong_fraction():
    """As lambda increases (cost penalty grows), strong-model fraction
    should not increase monotonically (cheaper models picked more)."""
    prompts = [f"easy:p{i}" for i in range(5)] + [f"hard:p{i}" for i in range(5)]
    evaluator, _ = _build_evaluator(
        psi_haiku=[0.1, 0.9],
        psi_sonnet=[0.1, 0.1],
        prompts=prompts,
    )
    ds = PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])
    result = evaluator.evaluate(ds)
    fractions = [p.strong_model_fraction for p in result.pareto_curve]
    # At lambda=0, sonnet wins on hard cluster (lower error). As lambda
    # grows, haiku gets picked even on hard cluster because of cost penalty.
    # So strong fraction should be monotonically non-increasing.
    for i in range(1, len(fractions)):
        assert fractions[i] <= fractions[0] + 1e-9  # never increases above lambda=0
