"""
Loss and evaluation metrics for UniRoute.

These functions return LOSS values (0.0 = correct, 1.0 = error),
not accuracy values. This aligns with the paper's formulation where
we minimize expected error.
"""

from enum import Enum
from typing import Callable


class MetricType(Enum):
    """Available metric types for evaluation."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    F1 = "f1"
    NORMALIZED_EXACT = "normalized_exact"
    MMLU = "mmlu"  # Extracts letter choice from response


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact string match loss.

    Returns:
        0.0 if prediction equals ground_truth (after strip/lower), 1.0 otherwise.
    """
    pred_clean = prediction.strip().lower()
    truth_clean = ground_truth.strip().lower()
    return 0.0 if pred_clean == truth_clean else 1.0


def normalized_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Normalized exact match that handles common formatting differences.

    Removes punctuation and extra whitespace before comparing.

    Returns:
        0.0 if normalized strings match, 1.0 otherwise.
    """
    import re

    def normalize(text: str) -> str:
        # Remove punctuation, lowercase, collapse whitespace
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    return 0.0 if normalize(prediction) == normalize(ground_truth) else 1.0


def contains_match(prediction: str, ground_truth: str) -> float:
    """
    Check if ground_truth is contained within prediction.

    Useful for cases where the model might output additional text
    around the correct answer.

    Returns:
        0.0 if ground_truth is found in prediction, 1.0 otherwise.
    """
    return 0.0 if ground_truth.lower() in prediction.lower() else 1.0


def f1_score_loss(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score loss (1 - F1).

    Computes F1 based on word overlap between prediction and ground truth.

    Returns:
        1 - F1 score (0.0 = perfect match, 1.0 = no overlap).
    """
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not truth_tokens:
        return 1.0

    intersection = pred_tokens & truth_tokens

    if len(intersection) == 0:
        return 1.0

    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return 1.0 - f1


def mmlu_match(prediction: str, ground_truth: str) -> float:
    """
    MMLU-specific metric that extracts letter choice (A/B/C/D) from response.

    Handles various response formats:
    - "A"
    - "The answer is A"
    - "A) option text"
    - "(A)"
    - "Answer: A"

    Returns:
        0.0 if extracted letter matches ground_truth, 1.0 otherwise.
    """
    import re

    # Ground truth should be a single letter
    truth = ground_truth.strip().upper()
    if truth not in ['A', 'B', 'C', 'D']:
        # Fall back to contains match if ground truth isn't a letter
        return contains_match(prediction, ground_truth)

    pred = prediction.strip()

    # Try various patterns to extract the letter
    patterns = [
        r'^([A-D])\b',  # Starts with letter
        r'\b([A-D])\)',  # Letter followed by )
        r'\(([A-D])\)',  # Letter in parentheses
        r'answer\s*(?:is|:)?\s*([A-D])\b',  # "answer is A" or "answer: A"
        r'([A-D])\s*(?:is correct|is the correct)',  # "A is correct"
        r'correct\s*(?:answer|option)\s*(?:is)?\s*([A-D])',  # "correct answer is A"
        r'\*\*([A-D])\*\*',  # **A** (markdown bold)
        r'^([A-D])[\.\:\)]',  # A. or A: or A)
    ]

    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            extracted = match.group(1).upper()
            return 0.0 if extracted == truth else 1.0

    # Last resort: check if only one letter A-D appears
    letters_found = re.findall(r'\b([A-D])\b', pred, re.IGNORECASE)
    if len(letters_found) == 1:
        return 0.0 if letters_found[0].upper() == truth else 1.0

    # If we can't extract, consider it wrong
    return 1.0


def get_metric(metric_type: MetricType) -> Callable[[str, str], float]:
    """
    Factory function to get a metric by type.

    Args:
        metric_type: The type of metric to retrieve.

    Returns:
        A callable that takes (prediction, ground_truth) and returns loss.

    Raises:
        ValueError: If metric_type is not recognized.
    """
    metrics: dict[MetricType, Callable[[str, str], float]] = {
        MetricType.EXACT_MATCH: exact_match,
        MetricType.NORMALIZED_EXACT: normalized_exact_match,
        MetricType.CONTAINS: contains_match,
        MetricType.F1: f1_score_loss,
        MetricType.MMLU: mmlu_match,
    }

    if metric_type not in metrics:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metrics[metric_type]


def compute_accuracy(losses: list[float]) -> float:
    """
    Compute accuracy from a list of losses.

    Args:
        losses: List of loss values (0.0 = correct, 1.0 = error).

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if not losses:
        return 0.0
    return 1.0 - (sum(losses) / len(losses))
