"""Smoke tests for router/models — LLMProfile, LLMRegistry, and client surfaces."""

import json
from pathlib import Path

import numpy as np
import pytest

from router.models.llm_profile import LLMProfile
from router.models.llm_registry import LLMRegistry
from router.models.llm_client import (
    LLMClient,
    LLMResponse,
    AnthropicClient,
    OpenAIClient,
    MistralClient,
    MockLLMClient,
)


# --- LLMProfile ---


def _make_profile(
    model_id: str = "test-model",
    psi=(0.1, 0.2, 0.3),
    cost: float = 0.001,
    counts=(30, 40, 30),
) -> LLMProfile:
    return LLMProfile(
        model_id=model_id,
        psi_vector=np.array(psi),
        cost_per_1k_tokens=cost,
        num_validation_samples=int(sum(counts)),
        cluster_sample_counts=np.array(counts),
    )


def test_llm_profile_basic():
    """Construct + read num_clusters + per-cluster error."""
    p = _make_profile()
    assert p.num_clusters == 3
    assert p.get_cluster_error(0) == pytest.approx(0.1)
    assert p.get_cluster_error(2) == pytest.approx(0.3)
    assert p.get_cluster_accuracy(0) == pytest.approx(0.9)


def test_llm_profile_overall_error_rate_weighted():
    """overall_error_rate is a sample-count-weighted average of per-cluster Ψ."""
    p = _make_profile(psi=(0.0, 1.0, 0.5), counts=(10, 10, 0))
    # Weighted: (0*10 + 1*10 + 0.5*0) / 20 = 0.5
    assert p.overall_error_rate == pytest.approx(0.5)
    assert p.overall_accuracy == pytest.approx(0.5)


def test_llm_profile_get_expected_error():
    """γ(x, h) = Φ(x)ᵀ · Ψ(h) on a one-hot phi."""
    p = _make_profile(psi=(0.1, 0.5, 0.3))
    one_hot_cluster_1 = np.array([0.0, 1.0, 0.0])
    assert p.get_expected_error(one_hot_cluster_1) == pytest.approx(0.5)


def test_llm_profile_dim_mismatch_raises():
    """psi_vector and cluster_sample_counts must have matching length."""
    with pytest.raises(ValueError):
        LLMProfile(
            model_id="bad",
            psi_vector=np.array([0.1, 0.2, 0.3]),
            cost_per_1k_tokens=0.0,
            num_validation_samples=10,
            cluster_sample_counts=np.array([5, 5]),  # length 2 != 3
        )


def test_llm_profile_save_load_round_trip(tmp_path: Path):
    """save → load preserves Ψ, costs, sample counts, metadata."""
    p = _make_profile(model_id="round-trip", psi=(0.11, 0.22, 0.33), cost=0.0042)
    p.metadata = {"provider": "test", "version": "1"}

    path = tmp_path / "p.json"
    p.save(path)

    raw = json.loads(path.read_text())
    assert raw["model_id"] == "round-trip"
    assert "_stats" in raw  # save adds derived stats

    q = LLMProfile.load(path)
    assert q.model_id == p.model_id
    assert q.cost_per_1k_tokens == pytest.approx(p.cost_per_1k_tokens)
    assert np.array_equal(q.psi_vector, p.psi_vector)
    assert np.array_equal(q.cluster_sample_counts, p.cluster_sample_counts)
    assert q.metadata == {"provider": "test", "version": "1"}


def test_llm_profile_strongest_and_weakest_clusters():
    """strongest = lowest error; weakest = highest error."""
    p = _make_profile(psi=(0.1, 0.5, 0.3, 0.05), counts=(25, 25, 25, 25))
    strongest = p.strongest_clusters(n=2)
    assert strongest[0][0] == 3  # cluster 3 has 0.05 (lowest error)
    weakest = p.weakest_clusters(n=2)
    assert weakest[0][0] == 1  # cluster 1 has 0.5 (highest error)


# --- LLMRegistry ---


def test_registry_register_get_contains():
    """register() + get() + __contains__ + __len__ + iteration."""
    reg = LLMRegistry()
    assert len(reg) == 0
    p = _make_profile(model_id="a")
    reg.register(p)
    assert len(reg) == 1
    assert "a" in reg
    assert reg.get("a") is p
    assert list(reg) == [p]


def test_registry_first_registered_becomes_default():
    """First profile becomes default."""
    reg = LLMRegistry()
    a = _make_profile(model_id="a")
    b = _make_profile(model_id="b")
    reg.register(a)
    reg.register(b)
    assert reg.default_model_id == "a"
    assert reg.get_default() is a


def test_registry_set_default_validates():
    """set_default raises when model isn't registered."""
    reg = LLMRegistry()
    reg.register(_make_profile(model_id="a"))
    with pytest.raises(ValueError):
        reg.set_default("missing")


def test_registry_get_available_models_filters():
    """get_available_models(['a']) returns only matching profile."""
    reg = LLMRegistry()
    reg.register(_make_profile(model_id="a"))
    reg.register(_make_profile(model_id="b"))
    out = reg.get_available_models(["a"])
    assert len(out) == 1
    assert out[0].model_id == "a"


def test_registry_get_best_for_cluster():
    """get_best_for_cluster picks the model with the lowest error in that cluster."""
    reg = LLMRegistry()
    reg.register(_make_profile(model_id="a", psi=(0.1, 0.9, 0.5)))
    reg.register(_make_profile(model_id="b", psi=(0.9, 0.1, 0.5)))
    assert reg.get_best_for_cluster(0).model_id == "a"
    assert reg.get_best_for_cluster(1).model_id == "b"


def test_registry_save_load(tmp_path: Path):
    """Round-trip the whole registry through a directory."""
    reg = LLMRegistry()
    reg.register(_make_profile(model_id="a"))
    reg.register(_make_profile(model_id="b"))
    reg.save(tmp_path)

    reloaded = LLMRegistry.load(tmp_path)
    assert len(reloaded) == 2
    assert "a" in reloaded
    assert "b" in reloaded
    # Default preserved.
    assert reloaded.default_model_id == "a"


# --- LLMClient surface ---


def test_anthropic_client_init_no_network():
    """AnthropicClient instantiates with a dummy key without making a call."""
    c = AnthropicClient(model="claude-haiku-4-5", api_key="dummy")
    assert c.model_id == "claude-haiku-4-5"
    assert c.cost_per_1k_tokens == 0.001  # known cost from COSTS dict
    assert "AnthropicClient" in repr(c)


def test_anthropic_client_unknown_model_uses_default_cost():
    """Unknown model defaults to 0.003."""
    c = AnthropicClient(model="claude-future", api_key="dummy")
    assert c.cost_per_1k_tokens == 0.003


def test_openai_client_stub_raises():
    """OpenAIClient is deferred — instantiation raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        OpenAIClient(model="gpt-4")
    assert "deferred" in str(exc_info.value).lower()


def test_mistral_client_stub_raises():
    """MistralClient is deferred — instantiation raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as exc_info:
        MistralClient(model="mistral-small-latest")
    assert "deferred" in str(exc_info.value).lower()


def test_mock_client_returns_canned_response():
    """MockLLMClient honors the responses dict + falls back to default_response."""
    c = MockLLMClient(
        model="m",
        cost_per_1k=0.0,
        responses={"ping": "pong"},
        default_response="hello",
    )
    assert c.model_id == "m"
    r1 = c.generate("ping")
    assert r1.text == "pong"
    assert isinstance(r1, LLMResponse)
    r2 = c.generate("anything else")
    assert r2.text == "hello"


def test_llm_client_is_abstract():
    """LLMClient is an ABC — direct instantiation fails."""
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]
