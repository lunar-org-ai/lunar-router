"""Tests for auto-training: judge, augmenter, preferences, feedback loop, and orchestrator."""

import numpy as np
import pytest
import random

from lunar_router.core.clustering import KMeansClusterAssigner
from lunar_router.core.embeddings import PromptEmbedder, MockEmbeddingProvider
from lunar_router.models.llm_profile import LLMProfile
from lunar_router.models.llm_registry import LLMRegistry
from lunar_router.models.llm_client import MockLLMClient
from lunar_router.data.dataset import PromptDataset, PromptSample
from lunar_router.evaluation.response_cache import ResponseCache

from lunar_router.augmentation.judge import (
    LLMJudge, JudgeVerdict, PointwiseScore, _parse_pairwise, _parse_pointwise,
)
from lunar_router.augmentation.preference_data import PreferencePair, PreferenceDataset
from lunar_router.augmentation.golden_augmenter import GoldenAugmenter, AugmentedSample
from lunar_router.feedback.trace_to_training import TraceToTraining, TraceRecord
from lunar_router.feedback.drift_detector import DriftDetector, DriftReport
from lunar_router.training.auto_trainer import AutoTrainer, AutoTrainConfig, AutoTrainResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_profiles():
    return [
        LLMProfile(
            model_id="strong", psi_vector=np.array([0.1, 0.05, 0.08]),
            cost_per_1k_tokens=0.01, num_validation_samples=100,
            cluster_sample_counts=np.array([30, 40, 30]),
        ),
        LLMProfile(
            model_id="weak", psi_vector=np.array([0.4, 0.3, 0.5]),
            cost_per_1k_tokens=0.001, num_validation_samples=100,
            cluster_sample_counts=np.array([30, 40, 30]),
        ),
    ]


def _make_cache_and_dataset():
    random.seed(42)
    profiles = _make_profiles()
    cache = ResponseCache()
    samples = []

    for i in range(50):
        prompt = f"test prompt {i}"
        gt = f"answer_{i}"
        samples.append(PromptSample(prompt=prompt, ground_truth=gt))

        for p in profiles:
            error_rate = p.psi_vector[i % 3]
            loss = 0.0 if random.random() > error_rate else 1.0
            cache.add(prompt, p.model_id, f"resp from {p.model_id}", loss)

    dataset = PromptDataset(samples, name="test")
    return cache, dataset, profiles


def _make_router_components():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assigner = KMeansClusterAssigner(centroids)
    embedder = PromptEmbedder(MockEmbeddingProvider(dimension=2))
    return assigner, embedder


# ── Judge Parsing ─────────────────────────────────────────────────────────────

class TestJudgeParsing:
    def test_parse_pairwise_a_wins(self):
        text = "WINNER: A\nCONFIDENCE: 4\nREASON: A was more accurate"
        winner, conf, reason = _parse_pairwise(text)
        assert winner == "A"
        assert conf == 4
        assert "accurate" in reason

    def test_parse_pairwise_tie(self):
        winner, conf, _ = _parse_pairwise("WINNER: TIE\nCONFIDENCE: 2\nREASON: both ok")
        assert winner == "TIE"
        assert conf == 2

    def test_parse_pairwise_fallback(self):
        winner, conf, _ = _parse_pairwise("unparseable output")
        assert winner == "TIE"
        assert conf == 3

    def test_parse_pointwise(self):
        score, reason = _parse_pointwise("SCORE: 5\nREASON: excellent")
        assert score == 5
        assert "excellent" in reason

    def test_parse_pointwise_clamped(self):
        score, _ = _parse_pointwise("SCORE: 9")
        assert score == 5  # clamped


class TestJudgeVerdict:
    def test_winner_model(self):
        v = JudgeVerdict("p", "m1", "m2", "A", 4, "reason", "judge")
        assert v.winner_model == "m1"
        assert v.loser_model == "m2"

    def test_tie_no_winner(self):
        v = JudgeVerdict("p", "m1", "m2", "TIE", 3, "reason", "judge")
        assert v.winner_model is None
        assert v.loser_model is None


class TestPointwiseScore:
    def test_loss_conversion(self):
        assert PointwiseScore("p", "m", 5, "r", "j").loss == pytest.approx(0.0)
        assert PointwiseScore("p", "m", 1, "r", "j").loss == pytest.approx(1.0)
        assert PointwiseScore("p", "m", 3, "r", "j").loss == pytest.approx(0.5)


class TestLLMJudge:
    def test_compare_with_mock(self):
        mock = MockLLMClient(model="judge", default_response="WINNER: A\nCONFIDENCE: 4\nREASON: better")
        judge = LLMJudge(mock)
        verdict = judge.compare("prompt", "m1", "resp1", "m2", "resp2")
        assert verdict.winner == "A"
        assert verdict.winner_model == "m1"

    def test_rate_with_mock(self):
        mock = MockLLMClient(model="judge", default_response="SCORE: 4\nREASON: good response")
        judge = LLMJudge(mock)
        score = judge.rate("prompt", "model-a", "response text")
        assert score.score == 4
        assert score.loss == pytest.approx(0.25)


# ── Preference Data ───────────────────────────────────────────────────────────

class TestPreferencePair:
    def test_fields(self):
        p = PreferencePair("prompt", "winner", "loser", source="judge", confidence=0.8)
        assert p.winner_model == "winner"
        assert p.source == "judge"


class TestPreferenceDataset:
    def test_add_from_cache(self):
        cache, _, profiles = _make_cache_and_dataset()
        ds = PreferenceDataset()
        count = ds.add_from_cache(cache)
        assert count > 0
        assert len(ds) == count

    def test_model_win_rates(self):
        cache, _, _ = _make_cache_and_dataset()
        ds = PreferenceDataset()
        ds.add_from_cache(cache)
        rates = ds.model_win_rates()
        assert "strong" in rates
        assert "weak" in rates
        assert rates["strong"] > rates["weak"]

    def test_filter_by_source(self):
        ds = PreferenceDataset([
            PreferencePair("p1", "a", "b", source="benchmark"),
            PreferencePair("p2", "a", "b", source="judge"),
        ])
        bench = ds.filter_by_source("benchmark")
        assert len(bench) == 1

    def test_save_and_load(self, tmp_path):
        ds = PreferenceDataset([
            PreferencePair("p1", "a", "b", confidence=0.9),
            PreferencePair("p2", "b", "a", confidence=0.7),
        ])
        path = tmp_path / "prefs.jsonl"
        ds.save(path)

        loaded = PreferenceDataset.load(path)
        assert len(loaded) == 2
        assert loaded.pairs[0].winner_model == "a"

    def test_add_from_verdicts(self):
        verdicts = [
            JudgeVerdict("p1", "m1", "m2", "A", 4, "reason", "judge"),
            JudgeVerdict("p2", "m1", "m2", "TIE", 3, "reason", "judge"),
        ]
        ds = PreferenceDataset()
        count = ds.add_from_verdicts(verdicts)
        assert count == 1  # TIE doesn't produce a pair


# ── Golden Augmenter ──────────────────────────────────────────────────────────

class TestGoldenAugmenter:
    def test_augment_from_cache(self):
        cache, dataset, profiles = _make_cache_and_dataset()
        augmenter = GoldenAugmenter(llm_clients=[], judge=None, metric_fn=None)
        prefs = augmenter.augment_from_cache(
            cache, [s.prompt for s in dataset.samples], use_judge=False
        )
        assert len(prefs) > 0

    def test_augmented_sample_best_model(self):
        sample = AugmentedSample(
            prompt="test", ground_truth="a",
            model_responses={"m1": "r1", "m2": "r2"},
            ground_truth_losses={"m1": 0.0, "m2": 1.0},
            judge_scores={"m1": 4, "m2": 2},
        )
        assert sample.best_model_by_gt == "m1"
        assert sample.best_model_by_judge == "m1"


# ── Trace to Training ────────────────────────────────────────────────────────

class TestTraceToTraining:
    def test_compute_psi_updates(self):
        converter = TraceToTraining(num_clusters=3)
        traces = [
            TraceRecord("r1", "model-a", 0, False, 100.0, 0.01),
            TraceRecord("r2", "model-a", 0, True, 200.0, 0.01),
            TraceRecord("r3", "model-a", 1, False, 150.0, 0.01),
            TraceRecord("r4", "model-b", 0, False, 100.0, 0.005),
            TraceRecord("r5", "model-b", 2, True, 100.0, 0.005),
        ]
        converter.add_traces(traces)
        updates = converter.compute_psi_updates()

        assert len(updates) == 2
        a_update = next(u for u in updates if u.model_id == "model-a")
        assert a_update.total_traces == 3
        assert a_update.psi_vector[0] == pytest.approx(0.5)  # 1/2 errors in cluster 0
        assert a_update.psi_vector[1] == pytest.approx(0.0)  # 0/1 errors in cluster 1

    def test_blend_with_profiles(self):
        profiles = _make_profiles()
        converter = TraceToTraining(num_clusters=3)

        # Add traces only for "strong" model
        for i in range(30):
            converter.add_trace(TraceRecord(
                f"r{i}", "strong", i % 3, i % 5 == 0, 100.0, 0.01
            ))

        blended = converter.blend_with_profiles(profiles, alpha=0.3)
        assert len(blended) == 2

        # Strong model should have changed
        strong_old = profiles[0]
        strong_new = next(p for p in blended if p.model_id == "strong")
        assert not np.array_equal(strong_old.psi_vector, strong_new.psi_vector)

        # Weak model should be unchanged (no traces)
        weak_new = next(p for p in blended if p.model_id == "weak")
        np.testing.assert_array_equal(weak_new.psi_vector, profiles[1].psi_vector)

    def test_latency_threshold_counts_as_error(self):
        converter = TraceToTraining(num_clusters=2, latency_threshold_ms=500.0)
        converter.add_trace(TraceRecord("r1", "m", 0, False, 1000.0, 0.01))  # slow = error
        converter.add_trace(TraceRecord("r2", "m", 0, False, 100.0, 0.01))  # fast = ok
        updates = converter.compute_psi_updates()
        assert updates[0].psi_vector[0] == pytest.approx(0.5)

    def test_reset(self):
        converter = TraceToTraining(num_clusters=2)
        converter.add_trace(TraceRecord("r1", "m", 0, True, 100.0, 0.01))
        converter.reset()
        assert converter.compute_psi_updates() == []


# ── Drift Detector ────────────────────────────────────────────────────────────

class TestDriftDetector:
    def test_no_drift_on_normal_data(self):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        assigner = KMeansClusterAssigner(centroids)

        # Embeddings close to centroids
        embeddings = centroids[np.random.randint(0, 3, 50)] + np.random.randn(50, 2) * 0.1

        detector = DriftDetector(assigner, drift_threshold=2.0)
        report = detector.check(embeddings)
        assert not report.needs_reclustering
        assert report.drift_ratio == pytest.approx(1.0)

    def test_drift_detected(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
        assigner = KMeansClusterAssigner(centroids)

        # First check sets baseline
        normal = np.random.randn(30, 2) * 0.1
        detector = DriftDetector(assigner, drift_threshold=1.5)
        detector.check(normal)

        # Drifted embeddings far from centroids
        drifted = np.random.randn(30, 2) * 0.1 + np.array([10.0, 10.0])
        report = detector.check(drifted)
        assert report.needs_reclustering
        assert report.drift_ratio > 1.5

    def test_empty_embeddings(self):
        centroids = np.array([[0.0, 0.0]])
        assigner = KMeansClusterAssigner(centroids)
        detector = DriftDetector(assigner)
        report = detector.check(np.array([]).reshape(0, 1))
        assert not report.needs_reclustering
        assert report.num_embeddings == 0

    def test_summary(self):
        report = DriftReport(
            avg_distance=0.5, baseline_distance=0.3, drift_ratio=1.67,
            outlier_fraction=0.15, needs_reclustering=True,
            num_embeddings=100, cluster_usage={0: 50, 1: 50},
        )
        s = report.summary()
        assert "1.67x" in s
        assert "True" in s


# ── AutoTrainer ───────────────────────────────────────────────────────────────

class TestAutoTrainer:
    def _make_trainer(self):
        cache, dataset, profiles = _make_cache_and_dataset()
        assigner, embedder = _make_router_components()
        return AutoTrainer(
            embedder=embedder,
            cluster_assigner=assigner,
            profiles=profiles,
            eval_dataset=dataset,
            eval_cache=cache,
            config=AutoTrainConfig(
                use_judge=False,
                lambda_steps=5,
                min_auroc_improvement=0.0,
                min_win_rate=0.0,
            ),
        )

    def test_train_from_cache(self):
        trainer = self._make_trainer()
        result = trainer.train_from_cache()
        assert isinstance(result, AutoTrainResult)
        assert result.baseline_metrics is not None
        assert result.new_metrics is not None
        assert result.preference_pairs_generated > 0

    def test_train_from_traces(self):
        trainer = self._make_trainer()
        traces = [
            TraceRecord(f"r{i}", "strong" if i % 2 == 0 else "weak",
                        i % 3, i % 7 == 0, 100.0, 0.01)
            for i in range(100)
        ]
        result = trainer.train_from_traces(traces)
        assert result.production_traces_used == 100

    def test_quality_gate_rejects_low_win_rate(self):
        trainer = self._make_trainer()
        trainer.config.min_win_rate = 0.99  # unrealistically high
        result = trainer.train_from_cache()
        assert not result.promoted
        assert "Win rate" in result.reason

    def test_get_router(self):
        trainer = self._make_trainer()
        router = trainer.get_router()
        decision = router.route("test prompt")
        assert decision.selected_model in {"strong", "weak"}

    def test_history_tracking(self):
        trainer = self._make_trainer()
        trainer.train_from_cache()
        trainer.train_from_cache()
        assert len(trainer.history) == 2

    def test_evaluate(self):
        trainer = self._make_trainer()
        result = trainer.evaluate()
        assert result.metrics.num_samples > 0

    def test_summary(self):
        trainer = self._make_trainer()
        result = trainer.train_from_cache()
        summary = result.summary()
        assert "AUROC" in summary
        assert "Duration" in summary
