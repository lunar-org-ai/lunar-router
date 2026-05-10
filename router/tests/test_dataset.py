"""Smoke tests for router/data/dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router.data.dataset import PromptDataset, PromptSample


def test_prompt_dataset_basic():
    """Construct from list of dicts; len + iter work."""
    ds = PromptDataset(
        [
            {"prompt": "p1", "ground_truth": "g1", "category": "math"},
            {"prompt": "p2", "ground_truth": "g2"},
        ],
        name="S_tr",
    )
    assert len(ds) == 2
    pairs = list(ds)
    assert pairs == [("p1", "g1"), ("p2", "g2")]
    assert ds.get_prompts() == ["p1", "p2"]
    assert ds.get_categories() == {"math"}
    assert "S_tr" in repr(ds)


def test_prompt_dataset_filter_by_category():
    ds = PromptDataset(
        [
            PromptSample(prompt="p1", ground_truth="g1", category="a"),
            PromptSample(prompt="p2", ground_truth="g2", category="b"),
            PromptSample(prompt="p3", ground_truth="g3", category="a"),
        ]
    )
    only_a = ds.filter_by_category("a")
    assert len(only_a) == 2
    assert only_a.get_prompts() == ["p1", "p3"]


def test_prompt_dataset_split_ratio():
    samples = [
        PromptSample(prompt=f"p{i}", ground_truth=f"g{i}") for i in range(10)
    ]
    ds = PromptDataset(samples)
    train, val = ds.split(val_fraction=0.2, seed=7)
    assert len(train) + len(val) == 10
    assert len(val) == 2
    # Disjoint.
    train_prompts = set(train.get_prompts())
    val_prompts = set(val.get_prompts())
    assert train_prompts.isdisjoint(val_prompts)


def test_prompt_dataset_split_stratified():
    samples = [PromptSample(prompt=f"a{i}", ground_truth="g", category="a") for i in range(8)]
    samples += [PromptSample(prompt=f"b{i}", ground_truth="g", category="b") for i in range(2)]
    ds = PromptDataset(samples)
    train, val = ds.split(val_fraction=0.5, seed=0, stratify_by_category=True)
    # Stratified guarantees at least 1 sample of each category in val.
    assert {s.category for s in val.samples} == {"a", "b"}


def test_prompt_dataset_save_load_round_trip(tmp_path: Path):
    samples = [
        PromptSample(prompt="p1", ground_truth="g1", category="math", metadata={"k": 1}),
        PromptSample(prompt="p2", ground_truth="g2", category="reasoning", metadata={}),
    ]
    ds = PromptDataset(samples, name="round-trip")
    path = tmp_path / "ds.json"
    ds.save(path)

    raw = json.loads(path.read_text())
    assert raw["name"] == "round-trip"
    assert raw["num_samples"] == 2

    reloaded = PromptDataset.load(path)
    assert len(reloaded) == 2
    assert reloaded.name == "round-trip"
    assert reloaded.samples[0].metadata == {"k": 1}


def test_prompt_dataset_from_list():
    ds = PromptDataset.from_list([("p1", "g1"), ("p2", "g2")], name="from_list")
    assert len(ds) == 2
    assert ds.samples[0].prompt == "p1"
    assert ds.samples[0].ground_truth == "g1"


def test_prompt_dataset_sample():
    ds = PromptDataset([PromptSample(prompt=f"p{i}", ground_truth="g") for i in range(20)])
    sub = ds.sample(5, seed=11)
    assert len(sub) == 5
    sub2 = ds.sample(5, seed=11)
    # Same seed → same selection.
    assert sub.get_prompts() == sub2.get_prompts()
