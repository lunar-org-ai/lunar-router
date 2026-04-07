"""
Baseline routing strategies for comparison.

Every evaluation should include these baselines to contextualize
how well the router is performing relative to trivial strategies.
"""

from dataclasses import dataclass
from typing import Optional
import random

from ..models.llm_profile import LLMProfile
from .response_cache import ResponseCache


@dataclass
class BaselineResult:
    """Result of evaluating a baseline on one sample."""

    selected_model: str
    loss: float  # 0.0 = correct, 1.0 = error
    cost_per_1k_tokens: float


class RandomBaseline:
    """
    Uniformly random model selection.

    The straight-line baseline — any good router should beat this.
    At N models, expected quality = average quality across all models.
    """

    def __init__(self, profiles: list[LLMProfile], seed: int = 42):
        self._profiles = profiles
        self._rng = random.Random(seed)

    def select(self) -> LLMProfile:
        return self._rng.choice(self._profiles)

    def evaluate(
        self,
        cache: ResponseCache,
        prompt_hashes: list[str],
    ) -> list[BaselineResult]:
        """Evaluate random selection on cached prompts."""
        results = []
        for ph in prompt_hashes:
            profile = self.select()
            entry = cache.get_by_hash(ph, profile.model_id)
            if entry is not None:
                results.append(BaselineResult(
                    selected_model=profile.model_id,
                    loss=entry.loss,
                    cost_per_1k_tokens=profile.cost_per_1k_tokens,
                ))
        return results


class OracleBaseline:
    """
    Always picks the cheapest correct model (upper bound).

    This is the theoretical best any router can achieve — it has
    perfect knowledge of which models will answer correctly.
    """

    def __init__(self, profiles: list[LLMProfile]):
        self._profiles = {p.model_id: p for p in profiles}

    def evaluate(
        self,
        cache: ResponseCache,
        prompt_hashes: list[str],
    ) -> list[BaselineResult]:
        """Evaluate oracle on cached prompts."""
        results = []
        for ph in prompt_hashes:
            models = cache.get_all_models_by_hash(ph)
            if not models:
                continue

            # Find cheapest correct model
            best: Optional[BaselineResult] = None
            for mid, entry in models.items():
                profile = self._profiles.get(mid)
                if profile is None:
                    continue

                if entry.loss == 0.0:  # correct
                    if best is None or profile.cost_per_1k_tokens < best.cost_per_1k_tokens:
                        best = BaselineResult(
                            selected_model=mid,
                            loss=0.0,
                            cost_per_1k_tokens=profile.cost_per_1k_tokens,
                        )

            # If no model got it right, pick cheapest anyway
            if best is None:
                cheapest_mid = min(
                    models.keys(),
                    key=lambda m: self._profiles[m].cost_per_1k_tokens
                    if m in self._profiles else float("inf"),
                )
                entry = models[cheapest_mid]
                profile = self._profiles.get(cheapest_mid)
                if profile:
                    best = BaselineResult(
                        selected_model=cheapest_mid,
                        loss=entry.loss,
                        cost_per_1k_tokens=profile.cost_per_1k_tokens,
                    )

            if best is not None:
                results.append(best)

        return results


class AlwaysStrongBaseline:
    """
    Always picks the most accurate (most expensive) model.

    Quality ceiling at maximum cost. The router should approach this
    quality while spending far less.
    """

    def __init__(self, profiles: list[LLMProfile]):
        # Strong = lowest overall error rate
        self._strong = min(profiles, key=lambda p: p.overall_error_rate)

    @property
    def model_id(self) -> str:
        return self._strong.model_id

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._strong.cost_per_1k_tokens

    def evaluate(
        self,
        cache: ResponseCache,
        prompt_hashes: list[str],
    ) -> list[BaselineResult]:
        """Evaluate always-strong on cached prompts."""
        results = []
        for ph in prompt_hashes:
            entry = cache.get_by_hash(ph, self._strong.model_id)
            if entry is not None:
                results.append(BaselineResult(
                    selected_model=self._strong.model_id,
                    loss=entry.loss,
                    cost_per_1k_tokens=self._strong.cost_per_1k_tokens,
                ))
        return results


class AlwaysWeakBaseline:
    """
    Always picks the cheapest model.

    Cost floor at minimum quality. The router should greatly exceed
    this quality while keeping costs close.
    """

    def __init__(self, profiles: list[LLMProfile]):
        # Weak = cheapest
        self._weak = min(profiles, key=lambda p: p.cost_per_1k_tokens)

    @property
    def model_id(self) -> str:
        return self._weak.model_id

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._weak.cost_per_1k_tokens

    def evaluate(
        self,
        cache: ResponseCache,
        prompt_hashes: list[str],
    ) -> list[BaselineResult]:
        """Evaluate always-weak on cached prompts."""
        results = []
        for ph in prompt_hashes:
            entry = cache.get_by_hash(ph, self._weak.model_id)
            if entry is not None:
                results.append(BaselineResult(
                    selected_model=self._weak.model_id,
                    loss=entry.loss,
                    cost_per_1k_tokens=self._weak.cost_per_1k_tokens,
                ))
        return results
