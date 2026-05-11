"""Tests for router/evaluation/baselines."""

from __future__ import annotations

import numpy as np

from router.evaluation.baselines import (
    AlwaysStrongBaseline,
    AlwaysWeakBaseline,
    OracleBaseline,
    RandomBaseline,
)
from router.evaluation.cache import ResponseCache
from router.models.llm_profile import LLMProfile


def _profile(model_id: str, error: float, cost: float) -> LLMProfile:
    return LLMProfile(
        model_id=model_id,
        psi_vector=np.array([error, error]),
        cost_per_1k_tokens=cost,
        num_validation_samples=20,
        cluster_sample_counts=np.array([10, 10]),
    )


def _seed_cache(prompts: list[str], answers: dict[tuple[str, str], float]) -> ResponseCache:
    """Seed a cache; answers is {(prompt, model_id): loss}."""
    cache = ResponseCache()
    for (prompt, model_id), loss in answers.items():
        cache.add(prompt, model_id, response_text="resp", loss=loss)
    return cache


def test_always_strong_picks_lowest_error_profile():
    weak = _profile("haiku", error=0.4, cost=0.001)
    strong = _profile("sonnet", error=0.1, cost=0.003)
    bs = AlwaysStrongBaseline([weak, strong])
    assert bs.model_id == "sonnet"
    assert bs.cost_per_1k_tokens == 0.003


def test_always_weak_picks_cheapest_profile():
    weak = _profile("haiku", error=0.4, cost=0.001)
    strong = _profile("sonnet", error=0.1, cost=0.003)
    bw = AlwaysWeakBaseline([weak, strong])
    assert bw.model_id == "haiku"
    assert bw.cost_per_1k_tokens == 0.001


def test_oracle_picks_cheapest_correct_per_prompt():
    weak = _profile("haiku", error=0.4, cost=0.001)
    strong = _profile("sonnet", error=0.1, cost=0.003)

    # On p1: only sonnet correct → oracle picks sonnet.
    # On p2: both correct → oracle picks haiku (cheaper).
    cache = _seed_cache(
        ["p1", "p2"],
        {
            ("p1", "haiku"): 1.0, ("p1", "sonnet"): 0.0,
            ("p2", "haiku"): 0.0, ("p2", "sonnet"): 0.0,
        },
    )
    hashes = [ResponseCache.hash_prompt(p) for p in ["p1", "p2"]]
    oracle = OracleBaseline([weak, strong])
    out = oracle.evaluate(cache, hashes)
    assert {r.selected_model for r in out} == {"sonnet", "haiku"}
    by_hash = {hashes[i]: out[i] for i in range(len(out))}
    # The oracle is order-preserving with prompt_hashes input — verify by content.
    p1_result = next(r for r in out if r.selected_model == "sonnet")
    p2_result = next(r for r in out if r.selected_model == "haiku")
    assert p1_result.loss == 0.0
    assert p2_result.loss == 0.0


def test_oracle_falls_back_to_cheapest_when_no_one_correct():
    weak = _profile("haiku", error=0.4, cost=0.001)
    strong = _profile("sonnet", error=0.1, cost=0.003)
    cache = _seed_cache(
        ["p1"],
        {
            ("p1", "haiku"): 1.0, ("p1", "sonnet"): 1.0,
        },
    )
    hashes = [ResponseCache.hash_prompt("p1")]
    out = OracleBaseline([weak, strong]).evaluate(cache, hashes)
    assert len(out) == 1
    # Both wrong → cheapest = haiku.
    assert out[0].selected_model == "haiku"


def test_random_baseline_yields_one_result_per_prompt():
    weak = _profile("haiku", error=0.4, cost=0.001)
    strong = _profile("sonnet", error=0.1, cost=0.003)
    cache = _seed_cache(
        ["p1", "p2", "p3"],
        {
            ("p1", "haiku"): 0.0, ("p1", "sonnet"): 0.0,
            ("p2", "haiku"): 1.0, ("p2", "sonnet"): 0.0,
            ("p3", "haiku"): 0.0, ("p3", "sonnet"): 1.0,
        },
    )
    hashes = [ResponseCache.hash_prompt(p) for p in ["p1", "p2", "p3"]]
    out = RandomBaseline([weak, strong], seed=0).evaluate(cache, hashes)
    assert len(out) == 3
    for r in out:
        assert r.selected_model in {"haiku", "sonnet"}
