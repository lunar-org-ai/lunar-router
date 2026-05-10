"""Tests for router.augmentation.goldens.

Uses a FakeJudge + MockLLMClient so no real brain or LLM is hit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from router.augmentation.goldens import (
    AugmentationResult,
    AugmentedSample,
    GoldenAugmenter,
)
from router.augmentation.judge import LLMJudge, PointwiseScore
from router.data.dataset import PromptDataset, PromptSample
from router.models.llm_client import MockLLMClient


class _FakeJudge:
    """Stand-in for LLMJudge that returns scores from a dict {(prompt, model_id): score}."""

    def __init__(self, scores: dict):
        self._scores = scores
        self.calls: list[tuple[str, str]] = []

    def rate(self, prompt: str, model_id: str, response: str) -> PointwiseScore:
        self.calls.append((prompt, model_id))
        score = self._scores.get((prompt, model_id), 3)
        return PointwiseScore(
            prompt=prompt,
            model_id=model_id,
            score=score,
            reasoning="fake",
            judge_model="fake",
        )


def _client(model_id: str, responses: dict | None = None, default: str = "ok") -> MockLLMClient:
    return MockLLMClient(
        model=model_id,
        cost_per_1k=0.0,
        responses=responses or {},
        default_response=default,
    )


def _ds(prompts: list[str]) -> PromptDataset:
    return PromptDataset([PromptSample(prompt=p, ground_truth="") for p in prompts])


# ---------------------------------------------------------------------------


def test_augment_basic_returns_result(tmp_path: Path):
    """Smoke: 3 prompts × 2 models → preference pairs + persisted JSONL."""
    a = _client("haiku")
    b = _client("sonnet")
    judge = _FakeJudge({
        ("Q1", "haiku"): 5, ("Q1", "sonnet"): 2,  # haiku wins
        ("Q2", "haiku"): 2, ("Q2", "sonnet"): 5,  # sonnet wins
        ("Q3", "haiku"): 4, ("Q3", "sonnet"): 4,  # tie → no pair
    })
    augmenter = GoldenAugmenter(
        llm_clients=[a, b],
        judge=judge,
        max_samples=500,
        output_dir=tmp_path,
    )

    result = augmenter.augment(_ds(["Q1", "Q2", "Q3"]))
    assert isinstance(result, AugmentationResult)
    assert len(result.samples) == 3
    assert all(isinstance(s, AugmentedSample) for s in result.samples)
    # Two pairs (Q3 tied).
    assert len(result.preference_dataset) == 2
    winners = {p.winner_model for p in result.preference_dataset.pairs}
    assert winners == {"haiku", "sonnet"}

    # Persisted JSONL exists + parses.
    assert result.persisted_path is not None
    assert result.persisted_path.exists()
    lines = result.persisted_path.read_text().strip().split("\n")
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert "winner_model" in payload and "loser_model" in payload


def test_augment_caps_at_max_samples(tmp_path: Path, caplog):
    """Inputs above max_samples are truncated *before* any LLM calls."""
    a = _client("haiku")
    judge = _FakeJudge({})
    augmenter = GoldenAugmenter(
        llm_clients=[a],
        judge=judge,
        max_samples=4,
        output_dir=tmp_path,
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="router.augmentation.goldens"):
        result = augmenter.augment(_ds([f"q{i}" for i in range(20)]))

    # Only 4 samples were processed → judge was called at most 4 times.
    assert len(result.samples) == 4
    assert any("truncating" in r.message for r in caplog.records)


def test_augment_persist_false_returns_none_path(tmp_path: Path):
    a = _client("haiku")
    augmenter = GoldenAugmenter(
        llm_clients=[a],
        judge=None,
        output_dir=tmp_path,
    )
    result = augmenter.augment(_ds(["Q1"]), persist=False)
    assert result.persisted_path is None
    # Output dir should have nothing written.
    assert list(tmp_path.glob("pp_*.jsonl")) == []


def test_augment_uses_metric_fn_for_gt_signal(tmp_path: Path):
    """When metric_fn is set + ground_truth present, ground-truth signal
    drives the preference pair (overrides judge tie)."""
    a = MockLLMClient(model="a", default_response="42", cost_per_1k=0.0)
    b = MockLLMClient(model="b", default_response="wrong", cost_per_1k=0.0)
    judge = _FakeJudge({
        ("Q1", "a"): 3, ("Q1", "b"): 3,  # judge tied
    })
    metric_fn = lambda pred, gt: 0.0 if pred == gt else 1.0

    augmenter = GoldenAugmenter(
        llm_clients=[a, b],
        judge=judge,
        metric_fn=metric_fn,
        output_dir=tmp_path,
    )

    ds = PromptDataset([PromptSample(prompt="Q1", ground_truth="42")])
    result = augmenter.augment(ds)
    assert len(result.preference_dataset) == 1
    pair = result.preference_dataset.pairs[0]
    assert pair.winner_model == "a"
    assert pair.loser_model == "b"
    assert pair.source == "benchmark"  # GT signal beats judge


def test_augment_from_cache_only_judges_no_generation(tmp_path: Path):
    """augment_from_cache reads from the cache, optionally calls judge,
    skips LLM generation."""
    from types import SimpleNamespace

    class FakeCache:
        def __init__(self):
            self._map = {
                "Q1": {
                    "a": SimpleNamespace(response_text="ans-a", loss=0.0),
                    "b": SimpleNamespace(response_text="ans-b", loss=1.0),
                },
                "Q2": {
                    "a": SimpleNamespace(response_text="ans-a2", loss=1.0),
                    "b": SimpleNamespace(response_text="ans-b2", loss=0.0),
                },
            }

        def get_all_models(self, prompt):
            return self._map.get(prompt, {})

    judge = _FakeJudge({})
    augmenter = GoldenAugmenter(
        llm_clients=[],
        judge=judge,
        output_dir=tmp_path,
    )
    result = augmenter.augment_from_cache(
        FakeCache(), prompts=["Q1", "Q2"], use_judge=False
    )
    # Two pairs (one per prompt) from the GT signal in the cache.
    assert len(result.preference_dataset) == 2
    assert {p.winner_model for p in result.preference_dataset.pairs} == {"a", "b"}
