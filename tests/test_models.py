"""Tests for LLM models: LLMProfile, LLMRegistry, LLMResponse, MockLLMClient."""

import json
import numpy as np
import pytest
from pathlib import Path

from lunar_router.models.llm_profile import LLMProfile
from lunar_router.models.llm_registry import LLMRegistry
from lunar_router.models.llm_client import (
    LLMResponse,
    MockLLMClient,
    create_client,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_profile(model_id="gpt-4o", cost=0.00625, num_clusters=5, error_rate=0.2):
    """Create a test LLMProfile with uniform error rate."""
    psi = np.full(num_clusters, error_rate)
    counts = np.full(num_clusters, 20)
    return LLMProfile(
        model_id=model_id,
        psi_vector=psi,
        cost_per_1k_tokens=cost,
        num_validation_samples=100,
        cluster_sample_counts=counts,
    )


# ── LLMProfile ────────────────────────────────────────────────────────────────

class TestLLMProfile:
    def test_create_profile(self):
        profile = make_profile()
        assert profile.model_id == "gpt-4o"
        assert profile.num_clusters == 5
        assert profile.cost_per_1k_tokens == 0.00625

    def test_overall_error_and_accuracy(self):
        profile = make_profile(error_rate=0.3)
        assert abs(profile.overall_error_rate - 0.3) < 1e-6
        assert abs(profile.overall_accuracy - 0.7) < 1e-6

    def test_expected_error(self):
        profile = make_profile(num_clusters=3, error_rate=0.0)
        # Set specific per-cluster errors
        profile.psi_vector = np.array([0.1, 0.5, 0.9])
        # One-hot assignment to cluster 0
        phi = np.array([1.0, 0.0, 0.0])
        assert abs(profile.get_expected_error(phi) - 0.1) < 1e-6
        # Soft assignment
        phi = np.array([0.5, 0.3, 0.2])
        expected = 0.5 * 0.1 + 0.3 * 0.5 + 0.2 * 0.9
        assert abs(profile.get_expected_error(phi) - expected) < 1e-6

    def test_expected_error_wrong_dimensions(self):
        profile = make_profile(num_clusters=3)
        with pytest.raises(ValueError, match="phi length"):
            profile.get_expected_error(np.array([1.0, 0.0]))

    def test_cluster_error_and_accuracy(self):
        profile = make_profile(num_clusters=3)
        profile.psi_vector = np.array([0.1, 0.5, 0.9])
        assert abs(profile.get_cluster_error(0) - 0.1) < 1e-6
        assert abs(profile.get_cluster_accuracy(0) - 0.9) < 1e-6

    def test_cluster_error_out_of_range(self):
        profile = make_profile(num_clusters=3)
        with pytest.raises(ValueError, match="out of range"):
            profile.get_cluster_error(5)
        with pytest.raises(ValueError, match="out of range"):
            profile.get_cluster_error(-1)

    def test_strongest_and_weakest_clusters(self):
        profile = make_profile(num_clusters=5)
        profile.psi_vector = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
        strongest = profile.strongest_clusters(2)
        assert strongest[0][0] == 1  # cluster 1 has lowest error
        assert strongest[1][0] == 3  # cluster 3 is next
        weakest = profile.weakest_clusters(2)
        assert weakest[0][0] == 2  # cluster 2 has highest error

    def test_save_and_load(self, tmp_path):
        profile = make_profile(model_id="test-model", num_clusters=4)
        filepath = tmp_path / "profile.json"
        profile.save(filepath)

        loaded = LLMProfile.load(filepath)
        assert loaded.model_id == "test-model"
        assert loaded.num_clusters == 4
        np.testing.assert_array_almost_equal(loaded.psi_vector, profile.psi_vector)
        np.testing.assert_array_almost_equal(
            loaded.cluster_sample_counts, profile.cluster_sample_counts
        )

    def test_to_dict_and_from_dict(self):
        profile = make_profile()
        d = profile.to_dict()
        restored = LLMProfile.from_dict(d)
        assert restored.model_id == profile.model_id
        np.testing.assert_array_almost_equal(restored.psi_vector, profile.psi_vector)

    def test_mismatched_vector_lengths(self):
        with pytest.raises(ValueError, match="psi_vector length"):
            LLMProfile(
                model_id="bad",
                psi_vector=np.array([0.1, 0.2]),
                cost_per_1k_tokens=0.001,
                num_validation_samples=10,
                cluster_sample_counts=np.array([5, 5, 5]),
            )

    def test_repr(self):
        profile = make_profile()
        r = repr(profile)
        assert "gpt-4o" in r


# ── LLMRegistry ───────────────────────────────────────────────────────────────

class TestLLMRegistry:
    def test_register_and_get(self):
        reg = LLMRegistry()
        p = make_profile("model-a")
        reg.register(p)
        assert "model-a" in reg
        assert reg.get("model-a") is p
        assert len(reg) == 1

    def test_first_registered_becomes_default(self):
        reg = LLMRegistry()
        reg.register(make_profile("first"))
        reg.register(make_profile("second"))
        assert reg.default_model_id == "first"

    def test_unregister(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        reg.register(make_profile("b"))
        removed = reg.unregister("a")
        assert removed is not None
        assert "a" not in reg
        assert len(reg) == 1
        # Default should update
        assert reg.default_model_id == "b"

    def test_unregister_nonexistent(self):
        reg = LLMRegistry()
        assert reg.unregister("nope") is None

    def test_set_default(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        reg.register(make_profile("b"))
        reg.set_default("b")
        assert reg.default_model_id == "b"

    def test_set_default_nonexistent_raises(self):
        reg = LLMRegistry()
        with pytest.raises(ValueError, match="not registered"):
            reg.set_default("nonexistent")

    def test_get_all_and_model_ids(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        reg.register(make_profile("b"))
        assert len(reg.get_all()) == 2
        assert set(reg.get_model_ids()) == {"a", "b"}

    def test_get_available_models_with_filter(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        reg.register(make_profile("b"))
        reg.register(make_profile("c"))
        result = reg.get_available_models(["a", "c"])
        assert len(result) == 2
        assert {p.model_id for p in result} == {"a", "c"}

    def test_get_available_models_none_returns_all(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        reg.register(make_profile("b"))
        result = reg.get_available_models(None)
        assert len(result) == 2

    def test_filter_by_cost(self):
        reg = LLMRegistry()
        reg.register(make_profile("cheap", cost=0.001))
        reg.register(make_profile("expensive", cost=0.01))
        cheap = reg.filter_by_cost(0.005)
        assert len(cheap) == 1
        assert cheap[0].model_id == "cheap"

    def test_filter_by_accuracy(self):
        reg = LLMRegistry()
        reg.register(make_profile("good", error_rate=0.1))
        reg.register(make_profile("bad", error_rate=0.6))
        good = reg.filter_by_accuracy(0.8)
        assert len(good) == 1
        assert good[0].model_id == "good"

    def test_get_cheapest_and_most_accurate(self):
        reg = LLMRegistry()
        reg.register(make_profile("cheap", cost=0.001, error_rate=0.5))
        reg.register(make_profile("accurate", cost=0.01, error_rate=0.1))
        assert reg.get_cheapest().model_id == "cheap"
        assert reg.get_most_accurate().model_id == "accurate"

    def test_get_best_for_cluster(self):
        reg = LLMRegistry()
        p1 = make_profile("a", num_clusters=3)
        p1.psi_vector = np.array([0.1, 0.9, 0.5])
        p2 = make_profile("b", num_clusters=3)
        p2.psi_vector = np.array([0.9, 0.1, 0.5])
        reg.register(p1)
        reg.register(p2)
        assert reg.get_best_for_cluster(0).model_id == "a"
        assert reg.get_best_for_cluster(1).model_id == "b"

    def test_empty_registry(self):
        reg = LLMRegistry()
        assert reg.get_cheapest() is None
        assert reg.get_most_accurate() is None
        assert reg.get_default() is None
        assert reg.get_best_for_cluster(0) is None

    def test_save_and_load(self, tmp_path):
        reg = LLMRegistry()
        reg.register(make_profile("m1", num_clusters=4))
        reg.register(make_profile("m2", num_clusters=4))
        reg.set_default("m2")

        reg.save(tmp_path / "profiles")
        loaded = LLMRegistry.load(tmp_path / "profiles")
        assert len(loaded) == 2
        assert "m1" in loaded
        assert "m2" in loaded
        assert loaded.default_model_id == "m2"

    def test_iteration(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        reg.register(make_profile("b"))
        ids = {p.model_id for p in reg}
        assert ids == {"a", "b"}

    def test_summary(self):
        reg = LLMRegistry()
        reg.register(make_profile("a"))
        s = reg.summary()
        assert "a" in s
        assert "1 models" in s


# ── LLMResponse ───────────────────────────────────────────────────────────────

class TestLLMResponse:
    def test_basic_fields(self):
        r = LLMResponse(
            text="Hello",
            tokens_used=10,
            latency_ms=50.0,
            model_id="test",
        )
        assert r.text == "Hello"
        assert r.tokens_used == 10
        assert r.latency_ms == 50.0
        assert r.input_tokens is None
        assert r.output_tokens is None

    def test_cost_returns_none(self):
        r = LLMResponse(text="", tokens_used=0, latency_ms=0, model_id="test")
        assert r.cost is None


# ── MockLLMClient ─────────────────────────────────────────────────────────────

class TestMockLLMClient:
    def test_default_response(self):
        client = MockLLMClient(model="mock", default_response="Hi!")
        resp = client.generate("anything")
        assert resp.text == "Hi!"
        assert resp.model_id == "mock"
        assert resp.tokens_used > 0
        assert resp.latency_ms >= 0

    def test_custom_responses(self):
        client = MockLLMClient(
            responses={"hello": "world"},
            default_response="fallback",
        )
        assert client.generate("hello").text == "world"
        assert client.generate("other").text == "fallback"

    def test_properties(self):
        client = MockLLMClient(model="my-model", cost_per_1k=0.005)
        assert client.model_id == "my-model"
        assert client.cost_per_1k_tokens == 0.005


# ── create_client ─────────────────────────────────────────────────────────────

class TestCreateClient:
    def test_mock_provider(self):
        client = create_client("mock", "test-model")
        assert client.model_id == "test-model"
        assert isinstance(client, MockLLMClient)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client("nonexistent", "model")
