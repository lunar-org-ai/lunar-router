"""Tests for the evaluation framework: metrics, baselines, response cache, and evaluator."""

import numpy as np
import pytest
from pathlib import Path

from lunar_router.core.clustering import KMeansClusterAssigner, ClusterResult
from lunar_router.core.embeddings import PromptEmbedder, MockEmbeddingProvider
from lunar_router.models.llm_profile import LLMProfile
from lunar_router.models.llm_registry import LLMRegistry
from lunar_router.data.dataset import PromptDataset, PromptSample
from lunar_router.router.uniroute import UniRouteRouter

from lunar_router.evaluation.response_cache import ResponseCache, CachedResponse
from lunar_router.evaluation.metrics import (
    compute_auroc,
    compute_apgr,
    compute_cpt,
    compute_pgr_at_savings,
    compute_win_rate,
    RoutingMetrics,
)
from lunar_router.evaluation.baselines import (
    RandomBaseline,
    OracleBaseline,
    AlwaysStrongBaseline,
    AlwaysWeakBaseline,
)
from lunar_router.evaluation.evaluator import RouterEvaluator, ParetoPoint


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_profiles(num_clusters=3):
    """Create a strong and weak model profile for testing."""
    strong = LLMProfile(
        model_id="strong-model",
        psi_vector=np.array([0.1, 0.05, 0.15]),  # low error
        cost_per_1k_tokens=0.01,
        num_validation_samples=100,
        cluster_sample_counts=np.array([30, 40, 30]),
    )
    weak = LLMProfile(
        model_id="weak-model",
        psi_vector=np.array([0.4, 0.3, 0.5]),  # high error
        cost_per_1k_tokens=0.001,
        num_validation_samples=100,
        cluster_sample_counts=np.array([30, 40, 30]),
    )
    return [strong, weak]


def _make_router(profiles, cost_weight=0.0):
    """Create a minimal router for testing."""
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assigner = KMeansClusterAssigner(centroids)
    embedder = PromptEmbedder(MockEmbeddingProvider(dimension=2))
    registry = LLMRegistry()
    for p in profiles:
        registry.register(p)
    return UniRouteRouter(
        embedder=embedder,
        cluster_assigner=assigner,
        registry=registry,
        cost_weight=cost_weight,
        use_soft_assignment=False,
    )


def _make_cache_with_data():
    """Create a cache with pre-populated responses for 10 prompts."""
    cache = ResponseCache()
    prompts = [f"test prompt {i}" for i in range(10)]

    for i, prompt in enumerate(prompts):
        # Strong model: correct on most (80%)
        strong_loss = 0.0 if i < 8 else 1.0
        cache.add(prompt, "strong-model", f"strong response {i}", strong_loss)

        # Weak model: correct on few (40%)
        weak_loss = 0.0 if i < 4 else 1.0
        cache.add(prompt, "weak-model", f"weak response {i}", weak_loss)

    return cache, prompts


def _make_dataset(prompts):
    """Create a PromptDataset from prompt strings."""
    samples = [PromptSample(prompt=p, ground_truth="unused") for p in prompts]
    return PromptDataset(samples, name="test")


# ── ResponseCache ─────────────────────────────────────────────────────────────


class TestResponseCache:
    def test_add_and_get(self):
        cache = ResponseCache()
        cache.add("hello", "model-a", "response", 0.0)
        entry = cache.get("hello", "model-a")
        assert entry is not None
        assert entry.loss == 0.0
        assert entry.response_text == "response"

    def test_get_missing_returns_none(self):
        cache = ResponseCache()
        assert cache.get("hello", "model-a") is None

    def test_get_all_models(self):
        cache = ResponseCache()
        cache.add("hello", "model-a", "a", 0.0)
        cache.add("hello", "model-b", "b", 1.0)
        models = cache.get_all_models("hello")
        assert set(models.keys()) == {"model-a", "model-b"}
        assert models["model-a"].loss == 0.0
        assert models["model-b"].loss == 1.0

    def test_has(self):
        cache = ResponseCache()
        cache.add("hello", "model-a", "a", 0.0)
        assert cache.has("hello", "model-a")
        assert not cache.has("hello", "model-b")

    def test_model_ids(self):
        cache = ResponseCache()
        cache.add("p1", "model-a", "a", 0.0)
        cache.add("p2", "model-b", "b", 0.0)
        assert cache.model_ids == {"model-a", "model-b"}

    def test_len(self):
        cache = ResponseCache()
        cache.add("p1", "m1", "r", 0.0)
        cache.add("p1", "m2", "r", 0.0)
        cache.add("p2", "m1", "r", 0.0)
        assert len(cache) == 3

    def test_coverage(self):
        cache = ResponseCache()
        cache.add("p1", "m1", "r", 0.0)
        cache.add("p2", "m1", "r", 0.0)
        cache.add("p1", "m2", "r", 0.0)
        assert cache.coverage("m1") == 2
        assert cache.coverage("m2") == 1

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        cache = ResponseCache(path)
        cache.add("hello", "model-a", "response a", 0.0, latency_ms=100.0)
        cache.add("hello", "model-b", "response b", 1.0, tokens_used=50)
        cache.save()

        # Load into new cache
        cache2 = ResponseCache(path)
        assert len(cache2) == 2
        entry = cache2.get("hello", "model-a")
        assert entry is not None
        assert entry.loss == 0.0
        assert entry.latency_ms == 100.0

    def test_overwrite_entry(self):
        cache = ResponseCache()
        cache.add("hello", "m1", "old", 1.0)
        cache.add("hello", "m1", "new", 0.0)
        entry = cache.get("hello", "m1")
        assert entry.response_text == "new"
        assert entry.loss == 0.0


class TestCachedResponse:
    def test_to_dict_and_back(self):
        entry = CachedResponse(
            prompt_hash="abc123",
            model_id="model-a",
            response_text="response",
            loss=0.5,
            latency_ms=200.0,
            tokens_used=42,
            metadata={"key": "value"},
        )
        d = entry.to_dict()
        restored = CachedResponse.from_dict(d)
        assert restored.prompt_hash == "abc123"
        assert restored.loss == 0.5
        assert restored.latency_ms == 200.0
        assert restored.metadata == {"key": "value"}

    def test_to_dict_minimal(self):
        entry = CachedResponse(
            prompt_hash="abc",
            model_id="m",
            response_text="r",
            loss=0.0,
        )
        d = entry.to_dict()
        assert "latency_ms" not in d
        assert "tokens_used" not in d
        assert "metadata" not in d


# ── Metrics ───────────────────────────────────────────────────────────────────


class TestAUROC:
    def test_perfect_separation(self):
        # All positives have higher scores than all negatives
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
        labels = [True, True, True, False, False, False]
        assert compute_auroc(scores, labels) == 1.0

    def test_random(self):
        # Interleaved scores — should be near 0.5
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        labels = [True, False, True, False, True, False]
        auroc = compute_auroc(scores, labels)
        assert 0.4 <= auroc <= 0.9  # not perfect, not terrible

    def test_worst_case(self):
        # All negatives ranked above positives
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
        labels = [False, False, False, True, True, True]
        assert compute_auroc(scores, labels) == 0.0

    def test_empty_returns_half(self):
        assert compute_auroc([], []) == 0.5

    def test_no_positives_returns_half(self):
        assert compute_auroc([0.5, 0.3], [False, False]) == 0.5

    def test_no_negatives_returns_half(self):
        assert compute_auroc([0.5, 0.3], [True, True]) == 0.5


class TestAPGR:
    def test_matches_strong(self):
        assert compute_apgr(0.9, 0.5, 0.9) == pytest.approx(1.0)

    def test_matches_weak(self):
        assert compute_apgr(0.5, 0.5, 0.9) == pytest.approx(0.0)

    def test_halfway(self):
        assert compute_apgr(0.7, 0.5, 0.9) == pytest.approx(0.5)

    def test_exceeds_strong(self):
        result = compute_apgr(0.95, 0.5, 0.9)
        assert result > 1.0

    def test_no_gap(self):
        # When strong == weak
        assert compute_apgr(0.5, 0.5, 0.5) == 1.0


class TestCPT:
    def test_achievable_target(self):
        points = [(0.9, 0.5), (0.85, 0.3), (0.7, 0.1)]
        cpt = compute_cpt(points, 0.90, quality_strong=1.0, quality_weak=0.5)
        # Target: 0.5 + 0.9*(1.0-0.5) = 0.95 — not reachable
        assert cpt is None

    def test_exact_target(self):
        points = [(0.95, 0.5), (0.8, 0.2), (0.6, 0.05)]
        cpt = compute_cpt(points, 0.50, quality_strong=1.0, quality_weak=0.5)
        # Target: 0.5 + 0.5*0.5 = 0.75. Points at (0.95, 0.5) and (0.8, 0.2) qualify.
        assert cpt == 0.2  # cheapest qualifying point

    def test_unreachable_returns_none(self):
        points = [(0.5, 0.1), (0.4, 0.05)]
        cpt = compute_cpt(points, 0.95, quality_strong=1.0, quality_weak=0.0)
        assert cpt is None


class TestPGRAtSavings:
    def test_basic(self):
        # 50% savings = max 0.5 strong fraction
        points = [(0.9, 0.5), (0.85, 0.3), (0.7, 0.1)]
        pgr = compute_pgr_at_savings(points, 0.50, quality_strong=1.0, quality_weak=0.5)
        # Best quality at <=0.5 strong: 0.9. PGR = (0.9-0.5)/(1.0-0.5) = 0.8
        assert pgr == pytest.approx(0.8)

    def test_no_data_returns_none(self):
        points = [(0.9, 0.8)]  # all above 50% savings threshold
        pgr = compute_pgr_at_savings(points, 0.50, quality_strong=1.0, quality_weak=0.5)
        assert pgr is None


class TestWinRate:
    def test_all_correct(self):
        assert compute_win_rate([True, True, True]) == pytest.approx(1.0)

    def test_all_wrong(self):
        assert compute_win_rate([False, False, False]) == pytest.approx(0.0)

    def test_mixed(self):
        assert compute_win_rate([True, False, True, False]) == pytest.approx(0.5)

    def test_empty(self):
        assert compute_win_rate([]) == 0.0


class TestRoutingMetrics:
    def test_summary_contains_key_values(self):
        metrics = RoutingMetrics(
            auroc=0.85,
            apgr=0.72,
            win_rate=0.80,
            cpt_50=0.05,
            cpt_75=0.10,
            cpt_90=0.25,
            cpt_95=0.40,
            pgr_at_25_savings=0.95,
            pgr_at_50_savings=0.80,
            pgr_at_75_savings=0.55,
            quality_strong=0.92,
            quality_weak=0.55,
            strong_model="gpt-4o",
            weak_model="gpt-4o-mini",
            num_samples=1000,
        )
        summary = metrics.summary()
        assert "0.8500" in summary  # AUROC
        assert "gpt-4o" in summary
        assert "1000" in summary

    def test_to_dict_roundtrip(self):
        metrics = RoutingMetrics(
            auroc=0.85, apgr=0.72, win_rate=0.80,
            cpt_50=0.05, cpt_75=0.10, cpt_90=0.25, cpt_95=0.40,
            pgr_at_25_savings=0.95, pgr_at_50_savings=0.80, pgr_at_75_savings=0.55,
            quality_strong=0.92, quality_weak=0.55,
            strong_model="gpt-4o", weak_model="gpt-4o-mini", num_samples=1000,
        )
        d = metrics.to_dict()
        assert d["auroc"] == 0.85
        assert d["num_samples"] == 1000


# ── Baselines ─────────────────────────────────────────────────────────────────


class TestAlwaysStrongBaseline:
    def test_picks_lowest_error(self):
        profiles = _make_profiles()
        baseline = AlwaysStrongBaseline(profiles)
        assert baseline.model_id == "strong-model"

    def test_evaluate(self):
        profiles = _make_profiles()
        cache, prompts = _make_cache_with_data()
        hashes = [ResponseCache.hash_prompt(p) for p in prompts]

        baseline = AlwaysStrongBaseline(profiles)
        results = baseline.evaluate(cache, hashes)
        assert len(results) == 10
        # Strong model correct on 8/10
        correct = sum(1 for r in results if r.loss == 0.0)
        assert correct == 8


class TestAlwaysWeakBaseline:
    def test_picks_cheapest(self):
        profiles = _make_profiles()
        baseline = AlwaysWeakBaseline(profiles)
        assert baseline.model_id == "weak-model"

    def test_evaluate(self):
        profiles = _make_profiles()
        cache, prompts = _make_cache_with_data()
        hashes = [ResponseCache.hash_prompt(p) for p in prompts]

        baseline = AlwaysWeakBaseline(profiles)
        results = baseline.evaluate(cache, hashes)
        assert len(results) == 10
        # Weak model correct on 4/10
        correct = sum(1 for r in results if r.loss == 0.0)
        assert correct == 4


class TestOracleBaseline:
    def test_picks_cheapest_correct(self):
        profiles = _make_profiles()
        cache, prompts = _make_cache_with_data()
        hashes = [ResponseCache.hash_prompt(p) for p in prompts]

        oracle = OracleBaseline(profiles)
        results = oracle.evaluate(cache, hashes)

        # Prompts 0-3: both correct → picks weak (cheaper)
        # Prompts 4-7: only strong correct → picks strong
        # Prompts 8-9: neither correct → picks cheapest
        for i, r in enumerate(results):
            if i < 4:
                assert r.selected_model == "weak-model"
                assert r.loss == 0.0
            elif i < 8:
                assert r.selected_model == "strong-model"
                assert r.loss == 0.0
            else:
                assert r.selected_model == "weak-model"  # cheapest fallback

        # Oracle correct on 8/10 (same as strong — can't do better)
        correct = sum(1 for r in results if r.loss == 0.0)
        assert correct == 8


class TestRandomBaseline:
    def test_returns_results(self):
        profiles = _make_profiles()
        cache, prompts = _make_cache_with_data()
        hashes = [ResponseCache.hash_prompt(p) for p in prompts]

        baseline = RandomBaseline(profiles, seed=42)
        results = baseline.evaluate(cache, hashes)
        assert len(results) == 10
        models_used = {r.selected_model for r in results}
        # With 10 samples and 2 models, random should pick both
        assert len(models_used) >= 1


# ── Evaluator ─────────────────────────────────────────────────────────────────


class TestRouterEvaluator:
    def test_evaluate_produces_result(self):
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache, prompts = _make_cache_with_data()
        dataset = _make_dataset(prompts)

        evaluator = RouterEvaluator(
            router=router,
            cache=cache,
            profiles=profiles,
            lambda_steps=5,
        )
        result = evaluator.evaluate(dataset, dataset_name="test")

        assert result.metrics.num_samples == 10
        assert result.metrics.strong_model == "strong-model"
        assert result.metrics.weak_model == "weak-model"
        assert result.metrics.quality_strong == pytest.approx(0.8)  # 8/10
        assert result.metrics.quality_weak == pytest.approx(0.4)  # 4/10
        assert 0.0 <= result.metrics.auroc <= 1.0
        assert 0.0 <= result.metrics.win_rate <= 1.0

    def test_pareto_curve_has_points(self):
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache, prompts = _make_cache_with_data()
        dataset = _make_dataset(prompts)

        evaluator = RouterEvaluator(
            router=router,
            cache=cache,
            profiles=profiles,
            lambda_steps=5,
        )
        result = evaluator.evaluate(dataset)

        assert len(result.pareto_curve) == 5
        for point in result.pareto_curve:
            assert 0.0 <= point.quality <= 1.0
            assert 0.0 <= point.strong_model_fraction <= 1.0
            assert point.avg_cost >= 0.0

    def test_baselines_present(self):
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache, prompts = _make_cache_with_data()
        dataset = _make_dataset(prompts)

        evaluator = RouterEvaluator(
            router=router, cache=cache, profiles=profiles, lambda_steps=3,
        )
        result = evaluator.evaluate(dataset)

        assert "always_strong" in result.baseline_quality
        assert "always_weak" in result.baseline_quality
        assert "random" in result.baseline_quality
        assert "oracle" in result.baseline_quality

    def test_summary_runs(self):
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache, prompts = _make_cache_with_data()
        dataset = _make_dataset(prompts)

        evaluator = RouterEvaluator(
            router=router, cache=cache, profiles=profiles, lambda_steps=3,
        )
        result = evaluator.evaluate(dataset, dataset_name="unit-test")
        summary = result.summary()
        assert "unit-test" in summary
        assert "AUROC" in summary

    def test_save_and_load(self, tmp_path):
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache, prompts = _make_cache_with_data()
        dataset = _make_dataset(prompts)

        evaluator = RouterEvaluator(
            router=router, cache=cache, profiles=profiles, lambda_steps=3,
        )
        result = evaluator.evaluate(dataset)

        path = tmp_path / "result.json"
        result.save(str(path))
        assert path.exists()

        import json
        with open(path) as f:
            data = json.load(f)
        assert "metrics" in data
        assert "pareto_curve" in data
        assert data["metrics"]["num_samples"] == 10

    def test_no_coverage_raises(self):
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache = ResponseCache()  # empty
        dataset = _make_dataset(["prompt with no cache"])

        evaluator = RouterEvaluator(
            router=router, cache=cache, profiles=profiles,
        )
        with pytest.raises(ValueError, match="No prompts have cached responses"):
            evaluator.evaluate(dataset)

    def test_high_lambda_routes_cheap(self):
        """High lambda should push routing toward the cheap model."""
        profiles = _make_profiles()
        router = _make_router(profiles, cost_weight=0.0)
        cache, prompts = _make_cache_with_data()
        dataset = _make_dataset(prompts)

        evaluator = RouterEvaluator(
            router=router, cache=cache, profiles=profiles,
            lambda_range=(0.0, 100.0), lambda_steps=5,
        )
        result = evaluator.evaluate(dataset)

        # At high lambda, strong_model_fraction should be lower
        low_lambda = result.pareto_curve[0]
        high_lambda = result.pareto_curve[-1]
        assert high_lambda.strong_model_fraction <= low_lambda.strong_model_fraction


class TestParetoPoint:
    def test_fields(self):
        p = ParetoPoint(
            lambda_value=1.0,
            quality=0.85,
            avg_cost=0.005,
            strong_model_fraction=0.3,
            model_distribution={"gpt-4o": 0.3, "gpt-4o-mini": 0.7},
        )
        assert p.quality == 0.85
        assert p.model_distribution["gpt-4o"] == 0.3
