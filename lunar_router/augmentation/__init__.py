"""
Data augmentation for router training.

Provides LLM-as-judge evaluation, golden label augmentation,
and preference pair generation for improving routing quality.
"""

from .judge import LLMJudge, JudgeVerdict
from .preference_data import PreferencePair, PreferenceDataset
from .golden_augmenter import GoldenAugmenter, AugmentedSample

__all__ = [
    "LLMJudge",
    "JudgeVerdict",
    "PreferencePair",
    "PreferenceDataset",
    "GoldenAugmenter",
    "AugmentedSample",
]
