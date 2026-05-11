"""Tests for router.augmentation.preference."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from router.augmentation.judge import JudgeVerdict, PointwiseScore
from router.augmentation.preference import PreferenceDataset, PreferencePair


def test_preference_pair_basic():
    p = PreferencePair(
        prompt="Q?",
        winner_model="haiku",
        loser_model="sonnet",
        source="judge",
        confidence=0.8,
    )
    assert p.prompt == "Q?"
    assert p.winner_model == "haiku"
    assert p.confidence == 0.8


def test_dataset_add_and_iter():
    ds = PreferenceDataset()
    ds.add(PreferencePair(prompt="p1", winner_model="a", loser_model="b"))
    ds.add(PreferencePair(prompt="p2", winner_model="b", loser_model="c"))
    assert len(ds) == 2
    pairs = list(ds)
    assert pairs[0].prompt == "p1"


def test_dataset_add_from_verdicts_skips_ties_and_parse_errors():
    """Verdicts with winner == 'TIE' or '' → no pair added."""
    verdicts = [
        JudgeVerdict("p1", "a", "b", winner="A", confidence=4, reasoning="x", judge_model="j"),
        JudgeVerdict("p2", "a", "b", winner="TIE", confidence=3, reasoning="x", judge_model="j"),
        JudgeVerdict("p3", "a", "b", winner="", confidence=1, reasoning="x", judge_model="j"),  # parse error
        JudgeVerdict("p4", "a", "b", winner="B", confidence=5, reasoning="x", judge_model="j"),
    ]
    ds = PreferenceDataset()
    n = ds.add_from_verdicts(verdicts)
    assert n == 2  # only A and B winners
    assert len(ds) == 2
    assert ds.pairs[0].source == "judge"


def test_dataset_add_from_pointwise_scores_threshold():
    """Pairs only created when |score_a - score_b| >= threshold."""
    scores = [
        PointwiseScore("p1", "a", 5, "x", "j"),
        PointwiseScore("p1", "b", 2, "x", "j"),  # diff 3 → pair
        PointwiseScore("p2", "a", 4, "x", "j"),
        PointwiseScore("p2", "b", 3, "x", "j"),  # diff 1 → pair (default threshold 0.5)
        PointwiseScore("p3", "a", 3, "x", "j"),
        PointwiseScore("p3", "b", 3, "x", "j"),  # diff 0 → no pair
    ]
    ds = PreferenceDataset()
    n = ds.add_from_pointwise_scores(scores, threshold=0.5)
    assert n == 2


def test_dataset_save_load_round_trip(tmp_path: Path):
    ds = PreferenceDataset()
    ds.add(PreferencePair(prompt="p1", winner_model="a", loser_model="b", source="judge", confidence=0.9))
    ds.add(PreferencePair(prompt="p2", winner_model="c", loser_model="d", source="benchmark", confidence=1.0))
    path = tmp_path / "pp.jsonl"
    ds.save(path)

    # Each line is a valid JSON object with the documented keys.
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["winner_model"] == "a"
    assert parsed[1]["source"] == "benchmark"

    reloaded = PreferenceDataset.load(path)
    assert len(reloaded) == 2
    assert reloaded.pairs[0].prompt == "p1"
    assert reloaded.pairs[1].confidence == 1.0


def test_model_win_rates():
    ds = PreferenceDataset()
    ds.add(PreferencePair(prompt="p1", winner_model="a", loser_model="b"))
    ds.add(PreferencePair(prompt="p2", winner_model="a", loser_model="c"))
    ds.add(PreferencePair(prompt="p3", winner_model="b", loser_model="c"))
    rates = ds.model_win_rates()
    # a: won 2 of 2 appearances → 1.0
    # b: won 1 of 2 appearances → 0.5
    # c: won 0 of 2 appearances → 0.0
    assert rates["a"] == pytest.approx(1.0)
    assert rates["b"] == pytest.approx(0.5)
    assert rates["c"] == pytest.approx(0.0)


def test_filter_by_source_and_confidence():
    ds = PreferenceDataset()
    ds.add(PreferencePair(prompt="p1", winner_model="a", loser_model="b", source="judge", confidence=0.9))
    ds.add(PreferencePair(prompt="p2", winner_model="c", loser_model="d", source="benchmark", confidence=1.0))
    ds.add(PreferencePair(prompt="p3", winner_model="e", loser_model="f", source="judge", confidence=0.4))

    only_judge = ds.filter_by_source("judge")
    assert len(only_judge) == 2
    high_conf = ds.filter_by_confidence(0.5)
    assert len(high_conf) == 2  # 0.9 + 1.0


def test_dataset_add_from_cache_duck_typed(tmp_path: Path):
    """add_from_cache pulls from a duck-typed cache (P15.3.6 will provide
    the real ResponseCache; here we just check the surface contract)."""
    # Two prompts, two models. A wins on prompt h1, B wins on prompt h2.
    entry = lambda loss, txt="r": SimpleNamespace(loss=loss, response_text=txt)

    class FakeCache:
        model_ids = ["a", "b"]
        prompt_hashes = ["h1", "h2"]

        def get_all_models_by_hash(self, h):
            if h == "h1":
                return {"a": entry(0.0), "b": entry(0.5)}  # A wins
            return {"a": entry(0.5), "b": entry(0.0)}     # B wins

    ds = PreferenceDataset()
    n = ds.add_from_cache(FakeCache())
    assert n == 2
    sources = {p.source for p in ds.pairs}
    assert sources == {"benchmark"}
