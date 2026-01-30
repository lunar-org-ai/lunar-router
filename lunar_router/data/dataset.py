"""
Dataset classes for UniRoute.

Handles prompt datasets used for training (S_tr) and validation (S_val).
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Any
import json
import random
from pathlib import Path


@dataclass
class PromptSample:
    """
    A single sample in the dataset.

    Attributes:
        prompt: The input prompt text.
        ground_truth: The expected correct answer/output.
        category: Optional category/task type (e.g., "math", "reasoning").
        metadata: Optional additional metadata about the sample.
    """
    prompt: str
    ground_truth: str
    category: Optional[str] = None
    metadata: Optional[dict[str, Any]] = field(default_factory=dict)


class PromptDataset:
    """
    Dataset of prompts with ground truths.

    Used for both S_tr (training) and S_val (validation) in the UniRoute framework.

    Attributes:
        samples: List of PromptSample objects.
        name: Name of the dataset (e.g., "S_tr", "S_val").
    """

    def __init__(self, samples: list[PromptSample | dict], name: str = "dataset"):
        """
        Initialize the dataset.

        Args:
            samples: List of PromptSample objects or dicts with 'prompt' and 'ground_truth' keys.
            name: Name identifier for the dataset.
        """
        # Convert dicts to PromptSample if needed
        converted_samples = []
        for s in samples:
            if isinstance(s, dict):
                converted_samples.append(PromptSample(
                    prompt=s["prompt"],
                    ground_truth=s.get("ground_truth", ""),
                    category=s.get("category"),
                    metadata=s.get("metadata", {}),
                ))
            else:
                converted_samples.append(s)

        self.samples = converted_samples
        self.name = name

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """
        Iterate over samples, yielding (prompt, ground_truth) tuples.

        This is the primary iteration interface used by the profiler and trainer.
        """
        for sample in self.samples:
            yield sample.prompt, sample.ground_truth

    def __getitem__(self, idx: int) -> PromptSample:
        """Get a sample by index."""
        return self.samples[idx]

    def get_prompts(self) -> list[str]:
        """Return list of all prompts."""
        return [s.prompt for s in self.samples]

    def get_categories(self) -> set[str]:
        """Return set of unique categories in the dataset."""
        return {s.category for s in self.samples if s.category is not None}

    def filter_by_category(self, category: str) -> "PromptDataset":
        """
        Create a new dataset with only samples from the given category.

        Args:
            category: The category to filter by.

        Returns:
            A new PromptDataset containing only matching samples.
        """
        filtered = [s for s in self.samples if s.category == category]
        return PromptDataset(filtered, name=f"{self.name}_{category}")

    def split(
        self,
        val_fraction: float = 0.1,
        seed: int = 42,
        stratify_by_category: bool = False,
    ) -> tuple["PromptDataset", "PromptDataset"]:
        """
        Split dataset into training and validation sets.

        Args:
            val_fraction: Fraction of data to use for validation (0 to 1).
            seed: Random seed for reproducibility.
            stratify_by_category: If True, maintain category proportions in both sets.

        Returns:
            Tuple of (training_dataset, validation_dataset).
        """
        random.seed(seed)

        if stratify_by_category and any(s.category for s in self.samples):
            # Stratified split
            train_samples = []
            val_samples = []

            # Group by category
            by_category: dict[Optional[str], list[PromptSample]] = {}
            for sample in self.samples:
                cat = sample.category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(sample)

            # Split each category
            for cat_samples in by_category.values():
                random.shuffle(cat_samples)
                val_size = max(1, int(len(cat_samples) * val_fraction))
                val_samples.extend(cat_samples[:val_size])
                train_samples.extend(cat_samples[val_size:])
        else:
            # Simple random split
            samples_copy = self.samples.copy()
            random.shuffle(samples_copy)

            val_size = int(len(samples_copy) * val_fraction)
            val_samples = samples_copy[:val_size]
            train_samples = samples_copy[val_size:]

        return (
            PromptDataset(train_samples, name="S_tr"),
            PromptDataset(val_samples, name="S_val"),
        )

    def sample(self, n: int, seed: Optional[int] = None) -> "PromptDataset":
        """
        Return a random sample of n items from the dataset.

        Args:
            n: Number of samples to return.
            seed: Optional random seed.

        Returns:
            A new PromptDataset with n samples.
        """
        if seed is not None:
            random.seed(seed)

        n = min(n, len(self.samples))
        sampled = random.sample(self.samples, n)
        return PromptDataset(sampled, name=f"{self.name}_sample_{n}")

    def save(self, path: str | Path) -> None:
        """
        Save dataset to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)

        data = {
            "name": self.name,
            "num_samples": len(self.samples),
            "samples": [
                {
                    "prompt": s.prompt,
                    "ground_truth": s.ground_truth,
                    "category": s.category,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "PromptDataset":
        """
        Load dataset from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A PromptDataset instance.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = [
            PromptSample(
                prompt=s["prompt"],
                ground_truth=s["ground_truth"],
                category=s.get("category"),
                metadata=s.get("metadata", {}),
            )
            for s in data["samples"]
        ]

        return cls(samples, name=data.get("name", "dataset"))

    @classmethod
    def from_list(
        cls,
        items: list[tuple[str, str]],
        name: str = "dataset",
    ) -> "PromptDataset":
        """
        Create dataset from a list of (prompt, ground_truth) tuples.

        Args:
            items: List of (prompt, ground_truth) tuples.
            name: Name for the dataset.

        Returns:
            A PromptDataset instance.
        """
        samples = [
            PromptSample(prompt=prompt, ground_truth=gt)
            for prompt, gt in items
        ]
        return cls(samples, name=name)

    def __repr__(self) -> str:
        return f"PromptDataset(name='{self.name}', num_samples={len(self)})"
