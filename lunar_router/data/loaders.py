"""
Dataset Loaders: Load and preprocess benchmark datasets.

Supports loading from:
- HuggingFace Datasets
- Local files (JSON, CSV, JSONL)
- Standard benchmarks (MMLU, GSM8K, TruthfulQA, ARC, etc.)
"""

from pathlib import Path
from typing import Optional, Callable
import json
import csv

from .dataset import PromptDataset, PromptSample


class DatasetLoader:
    """Base class for dataset loaders."""

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict]:
        """Load a JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def load_json(path: str | Path) -> list[dict]:
        """Load a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_csv(path: str | Path) -> list[dict]:
        """Load a CSV file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data


class MMLULoader(DatasetLoader):
    """
    Loader for MMLU (Massive Multitask Language Understanding) dataset.

    MMLU contains 57 subjects across STEM, humanities, social sciences, etc.
    Format: question, choices (A/B/C/D), answer
    """

    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]

    @classmethod
    def from_huggingface(
        cls,
        split: str = "test",
        subjects: Optional[list[str]] = None,
        max_samples_per_subject: Optional[int] = None,
    ) -> PromptDataset:
        """
        Load MMLU from HuggingFace datasets.

        Args:
            split: Dataset split ("test", "validation", "dev", "auxiliary_train").
            subjects: List of subjects to load. None = all subjects.
            max_samples_per_subject: Max samples per subject for limiting size.

        Returns:
            PromptDataset with MMLU questions.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required. Install with: pip install datasets")

        subjects = subjects or cls.SUBJECTS
        samples = []

        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)

                for i, item in enumerate(dataset):
                    if max_samples_per_subject and i >= max_samples_per_subject:
                        break

                    # Format: question with choices
                    choices = item["choices"]
                    question = item["question"]
                    answer_idx = item["answer"]

                    # Create prompt with multiple choice format
                    prompt = f"{question}\n"
                    prompt += f"A) {choices[0]}\n"
                    prompt += f"B) {choices[1]}\n"
                    prompt += f"C) {choices[2]}\n"
                    prompt += f"D) {choices[3]}\n"
                    prompt += "Answer:"

                    # Ground truth is the letter
                    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                    ground_truth = answer_map[answer_idx]

                    samples.append(PromptSample(
                        prompt=prompt,
                        ground_truth=ground_truth,
                        category=subject,
                        metadata={"choices": choices, "answer_idx": answer_idx},
                    ))

            except Exception as e:
                print(f"Warning: Could not load subject {subject}: {e}")

        return PromptDataset(samples, name=f"mmlu_{split}")

    @classmethod
    def from_local(
        cls,
        data_dir: str | Path,
        split: str = "test",
        subjects: Optional[list[str]] = None,
    ) -> PromptDataset:
        """
        Load MMLU from local CSV files.

        Expected structure:
            data_dir/
            ├── test/
            │   ├── abstract_algebra_test.csv
            │   └── ...
            └── dev/
                └── ...

        Args:
            data_dir: Root directory containing MMLU data.
            split: Dataset split.
            subjects: Subjects to load.

        Returns:
            PromptDataset with MMLU questions.
        """
        data_dir = Path(data_dir)
        subjects = subjects or cls.SUBJECTS
        samples = []

        for subject in subjects:
            file_path = data_dir / split / f"{subject}_{split}.csv"

            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            rows = cls.load_csv(file_path)

            for row in rows:
                # CSV columns: question, A, B, C, D, answer
                question = row.get("question", row.get("0", ""))
                choices = [
                    row.get("A", row.get("1", "")),
                    row.get("B", row.get("2", "")),
                    row.get("C", row.get("3", "")),
                    row.get("D", row.get("4", "")),
                ]
                answer = row.get("answer", row.get("5", ""))

                prompt = f"{question}\n"
                prompt += f"A) {choices[0]}\n"
                prompt += f"B) {choices[1]}\n"
                prompt += f"C) {choices[2]}\n"
                prompt += f"D) {choices[3]}\n"
                prompt += "Answer:"

                samples.append(PromptSample(
                    prompt=prompt,
                    ground_truth=answer,
                    category=subject,
                ))

        return PromptDataset(samples, name=f"mmlu_{split}")


class GSM8KLoader(DatasetLoader):
    """
    Loader for GSM8K (Grade School Math 8K) dataset.

    GSM8K contains grade school math word problems with step-by-step solutions.
    """

    @classmethod
    def from_huggingface(
        cls,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> PromptDataset:
        """
        Load GSM8K from HuggingFace.

        Args:
            split: Dataset split ("train" or "test").
            max_samples: Maximum number of samples to load.

        Returns:
            PromptDataset with math problems.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        dataset = load_dataset("gsm8k", "main", split=split, trust_remote_code=True)
        samples = []

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            question = item["question"]
            answer_full = item["answer"]

            # Extract final numerical answer (after ####)
            if "####" in answer_full:
                final_answer = answer_full.split("####")[-1].strip()
            else:
                final_answer = answer_full.strip()

            prompt = f"Solve the following math problem step by step:\n\n{question}\n\nAnswer:"

            samples.append(PromptSample(
                prompt=prompt,
                ground_truth=final_answer,
                category="math",
                metadata={"full_solution": answer_full},
            ))

        return PromptDataset(samples, name=f"gsm8k_{split}")


class TruthfulQALoader(DatasetLoader):
    """
    Loader for TruthfulQA dataset.

    TruthfulQA tests model tendency to generate truthful answers.
    """

    @classmethod
    def from_huggingface(
        cls,
        split: str = "validation",
        task: str = "multiple_choice",
        max_samples: Optional[int] = None,
    ) -> PromptDataset:
        """
        Load TruthfulQA from HuggingFace.

        Args:
            split: Dataset split.
            task: Task type ("multiple_choice" or "generation").
            max_samples: Maximum samples to load.

        Returns:
            PromptDataset with questions.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        dataset = load_dataset("truthful_qa", task, split=split, trust_remote_code=True)
        samples = []

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            question = item["question"]
            category = item.get("category", "general")

            if task == "multiple_choice":
                choices = item["mc1_targets"]["choices"]
                labels = item["mc1_targets"]["labels"]

                # Find correct answer
                correct_idx = labels.index(1) if 1 in labels else 0

                prompt = f"{question}\n"
                for j, choice in enumerate(choices):
                    prompt += f"{chr(65 + j)}) {choice}\n"
                prompt += "Answer:"

                ground_truth = chr(65 + correct_idx)
            else:
                # Generation task
                correct_answers = item.get("correct_answers", [])
                ground_truth = correct_answers[0] if correct_answers else ""
                prompt = f"{question}\n\nAnswer:"

            samples.append(PromptSample(
                prompt=prompt,
                ground_truth=ground_truth,
                category=category,
            ))

        return PromptDataset(samples, name=f"truthfulqa_{split}")


class ARCLoader(DatasetLoader):
    """
    Loader for ARC (AI2 Reasoning Challenge) dataset.

    ARC contains science questions from standardized tests.
    """

    @classmethod
    def from_huggingface(
        cls,
        split: str = "test",
        difficulty: str = "ARC-Challenge",  # or "ARC-Easy"
        max_samples: Optional[int] = None,
    ) -> PromptDataset:
        """
        Load ARC from HuggingFace.

        Args:
            split: Dataset split.
            difficulty: "ARC-Challenge" or "ARC-Easy".
            max_samples: Maximum samples to load.

        Returns:
            PromptDataset with science questions.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        dataset = load_dataset("ai2_arc", difficulty, split=split, trust_remote_code=True)
        samples = []

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            # Choices format: {"text": [...], "label": [...]}
            choice_texts = choices["text"]
            choice_labels = choices["label"]

            prompt = f"{question}\n"
            for label, text in zip(choice_labels, choice_texts):
                prompt += f"{label}) {text}\n"
            prompt += "Answer:"

            samples.append(PromptSample(
                prompt=prompt,
                ground_truth=answer_key,
                category="science",
                metadata={"difficulty": difficulty},
            ))

        return PromptDataset(samples, name=f"arc_{difficulty.lower()}_{split}")


class HellaSwagLoader(DatasetLoader):
    """
    Loader for HellaSwag dataset.

    HellaSwag tests commonsense reasoning with sentence completion.
    """

    @classmethod
    def from_huggingface(
        cls,
        split: str = "validation",
        max_samples: Optional[int] = None,
    ) -> PromptDataset:
        """
        Load HellaSwag from HuggingFace.

        Args:
            split: Dataset split.
            max_samples: Maximum samples to load.

        Returns:
            PromptDataset with completion questions.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        dataset = load_dataset("hellaswag", split=split, trust_remote_code=True)
        samples = []

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            context = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])

            prompt = f"Complete the following:\n\n{context}\n\n"
            for j, ending in enumerate(endings):
                prompt += f"{chr(65 + j)}) {ending}\n"
            prompt += "\nAnswer:"

            ground_truth = chr(65 + label)

            samples.append(PromptSample(
                prompt=prompt,
                ground_truth=ground_truth,
                category="commonsense",
            ))

        return PromptDataset(samples, name=f"hellaswag_{split}")


class WinograndeLoader(DatasetLoader):
    """
    Loader for Winogrande dataset.

    Winogrande tests commonsense reasoning with pronoun resolution.
    """

    @classmethod
    def from_huggingface(
        cls,
        split: str = "validation",
        size: str = "winogrande_xl",
        max_samples: Optional[int] = None,
    ) -> PromptDataset:
        """
        Load Winogrande from HuggingFace.

        Args:
            split: Dataset split.
            size: Dataset size variant.
            max_samples: Maximum samples to load.

        Returns:
            PromptDataset with pronoun resolution questions.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        dataset = load_dataset("winogrande", size, split=split, trust_remote_code=True)
        samples = []

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            sentence = item["sentence"]
            option1 = item["option1"]
            option2 = item["option2"]
            answer = item["answer"]

            prompt = f"{sentence}\n\n"
            prompt += f"A) {option1}\n"
            prompt += f"B) {option2}\n"
            prompt += "\nWhich option best fills the blank? Answer:"

            # Answer is "1" or "2", convert to A/B
            ground_truth = "A" if answer == "1" else "B"

            samples.append(PromptSample(
                prompt=prompt,
                ground_truth=ground_truth,
                category="commonsense",
            ))

        return PromptDataset(samples, name=f"winogrande_{split}")


# Convenience functions
def load_benchmark(
    name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    **kwargs,
) -> PromptDataset:
    """
    Load a benchmark dataset by name.

    Args:
        name: Benchmark name ("mmlu", "gsm8k", "truthfulqa", "arc", "hellaswag", "winogrande").
        split: Dataset split.
        max_samples: Maximum samples to load.
        **kwargs: Additional arguments for the specific loader.

    Returns:
        PromptDataset with benchmark data.

    Raises:
        ValueError: If benchmark name is not recognized.
    """
    loaders = {
        "mmlu": lambda: MMLULoader.from_huggingface(split=split, max_samples_per_subject=max_samples, **kwargs),
        "gsm8k": lambda: GSM8KLoader.from_huggingface(split=split, max_samples=max_samples, **kwargs),
        "truthfulqa": lambda: TruthfulQALoader.from_huggingface(split="validation", max_samples=max_samples, **kwargs),
        "arc": lambda: ARCLoader.from_huggingface(split=split, max_samples=max_samples, **kwargs),
        "arc-easy": lambda: ARCLoader.from_huggingface(split=split, difficulty="ARC-Easy", max_samples=max_samples, **kwargs),
        "arc-challenge": lambda: ARCLoader.from_huggingface(split=split, difficulty="ARC-Challenge", max_samples=max_samples, **kwargs),
        "hellaswag": lambda: HellaSwagLoader.from_huggingface(split="validation", max_samples=max_samples, **kwargs),
        "winogrande": lambda: WinograndeLoader.from_huggingface(split="validation", max_samples=max_samples, **kwargs),
    }

    name_lower = name.lower()
    if name_lower not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(loaders.keys())}")

    return loaders[name_lower]()


def load_combined_benchmark(
    benchmarks: list[str],
    split: str = "test",
    max_samples_per_benchmark: Optional[int] = None,
) -> PromptDataset:
    """
    Load and combine multiple benchmarks into a single dataset.

    Args:
        benchmarks: List of benchmark names.
        split: Dataset split.
        max_samples_per_benchmark: Max samples per benchmark.

    Returns:
        Combined PromptDataset.
    """
    all_samples = []

    for name in benchmarks:
        try:
            dataset = load_benchmark(name, split=split, max_samples=max_samples_per_benchmark)
            all_samples.extend(dataset.samples)
            print(f"Loaded {len(dataset)} samples from {name}")
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")

    return PromptDataset(all_samples, name="combined_benchmark")
