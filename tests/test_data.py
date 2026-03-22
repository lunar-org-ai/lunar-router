"""Tests for data components: PromptDataset, PromptSample."""

import json
import pytest
from pathlib import Path

from lunar_router.data.dataset import PromptDataset, PromptSample


class TestPromptSample:
    def test_basic_creation(self):
        s = PromptSample(prompt="What is 2+2?", ground_truth="4")
        assert s.prompt == "What is 2+2?"
        assert s.ground_truth == "4"
        assert s.category is None
        assert s.metadata == {}

    def test_with_metadata(self):
        s = PromptSample(
            prompt="test",
            ground_truth="answer",
            category="math",
            metadata={"source": "mmlu"},
        )
        assert s.category == "math"
        assert s.metadata["source"] == "mmlu"


class TestPromptDataset:
    def test_from_prompt_samples(self):
        samples = [
            PromptSample(prompt="q1", ground_truth="a1"),
            PromptSample(prompt="q2", ground_truth="a2"),
        ]
        ds = PromptDataset(samples)
        assert len(ds) == 2

    def test_from_dicts(self):
        data = [
            {"prompt": "q1", "ground_truth": "a1"},
            {"prompt": "q2", "ground_truth": "a2"},
        ]
        ds = PromptDataset(data)
        assert len(ds) == 2
        assert ds[0].prompt == "q1"
        assert ds[0].ground_truth == "a1"

    def test_from_dicts_with_optional_fields(self):
        data = [
            {"prompt": "q1", "ground_truth": "a1", "category": "math"},
            {"prompt": "q2"},  # ground_truth defaults to ""
        ]
        ds = PromptDataset(data)
        assert ds[0].category == "math"
        assert ds[1].ground_truth == ""

    def test_iteration(self):
        data = [
            {"prompt": "q1", "ground_truth": "a1"},
            {"prompt": "q2", "ground_truth": "a2"},
        ]
        ds = PromptDataset(data)
        pairs = list(ds)
        assert pairs == [("q1", "a1"), ("q2", "a2")]

    def test_getitem(self):
        data = [{"prompt": "q1", "ground_truth": "a1"}]
        ds = PromptDataset(data)
        sample = ds[0]
        assert isinstance(sample, PromptSample)
        assert sample.prompt == "q1"

    def test_empty_dataset(self):
        ds = PromptDataset([])
        assert len(ds) == 0
        assert list(ds) == []

    def test_name(self):
        ds = PromptDataset([], name="train")
        assert ds.name == "train"
