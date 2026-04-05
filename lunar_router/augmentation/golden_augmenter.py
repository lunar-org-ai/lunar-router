"""
Golden label augmentation for router training.

Generates model responses on benchmark data, evaluates with both
ground-truth metrics and LLM-judge, and produces augmented training
sets with richer signal than ground-truth alone.

This is the highest-impact data improvement: RouteLLM showed even
1,500 golden-label samples improved all router architectures by 3-8% AUROC.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import logging

from ..data.dataset import PromptDataset, PromptSample
from ..models.llm_client import LLMClient
from ..evaluation.response_cache import ResponseCache
from .judge import LLMJudge, JudgeVerdict, PointwiseScore
from .preference_data import PreferenceDataset

logger = logging.getLogger(__name__)


@dataclass
class AugmentedSample:
    """A prompt with responses and scores from multiple models."""

    prompt: str
    ground_truth: Optional[str]
    model_responses: dict[str, str]  # model_id -> response text
    ground_truth_losses: dict[str, float]  # model_id -> loss (0=correct)
    judge_scores: dict[str, int]  # model_id -> 1-5 score
    category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_model_by_gt(self) -> Optional[str]:
        """Model with lowest ground-truth loss."""
        if not self.ground_truth_losses:
            return None
        return min(self.ground_truth_losses, key=self.ground_truth_losses.get)

    @property
    def best_model_by_judge(self) -> Optional[str]:
        """Model with highest judge score."""
        if not self.judge_scores:
            return None
        return max(self.judge_scores, key=self.judge_scores.get)


class GoldenAugmenter:
    """
    Generates augmented training data by running models on benchmarks
    and evaluating with both ground-truth and LLM-judge.

    The augmented data provides:
    1. Richer Psi vectors (judge scores on production-like prompts)
    2. Preference pairs for matrix factorization training
    3. Difficulty signals (which prompts trip up which models)

    Usage:
        augmenter = GoldenAugmenter(
            llm_clients=[gpt4o, gpt4o_mini, llama],
            judge=LLMJudge(judge_client),
            metric_fn=exact_match,
        )

        # Run augmentation
        samples, preferences = augmenter.augment(dataset)
        print(f"Generated {len(preferences)} preference pairs")
    """

    def __init__(
        self,
        llm_clients: list[LLMClient],
        judge: Optional[LLMJudge] = None,
        metric_fn: Optional[Callable[[str, str], float]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        """
        Args:
            llm_clients: Models to generate responses.
            judge: Optional LLM judge for quality scoring.
            metric_fn: Ground-truth loss function (0=correct, 1=error).
            max_tokens: Max tokens per generation.
            temperature: Generation temperature.
        """
        self.llm_clients = {c.model_id: c for c in llm_clients}
        self.judge = judge
        self.metric_fn = metric_fn
        self.max_tokens = max_tokens
        self.temperature = temperature

    def augment(
        self,
        dataset: PromptDataset,
        use_judge: bool = True,
        cache: Optional[ResponseCache] = None,
    ) -> tuple[list[AugmentedSample], PreferenceDataset]:
        """
        Run full augmentation: generate, evaluate, create preference pairs.

        Args:
            dataset: Source dataset with prompts (and optionally ground_truth).
            use_judge: Whether to run LLM judge scoring.
            cache: Optional ResponseCache to populate and reuse.

        Returns:
            Tuple of (augmented_samples, preference_dataset).
        """
        if cache is None:
            cache = ResponseCache()

        samples: list[AugmentedSample] = []
        preferences = PreferenceDataset(name="augmented")

        total = len(dataset)
        for idx, sample in enumerate(dataset.samples):
            prompt = sample.prompt
            gt = sample.ground_truth

            if (idx + 1) % 50 == 0:
                logger.info(f"Augmenting {idx + 1}/{total}...")

            # Step 1: Generate responses from all models
            model_responses = {}
            gt_losses = {}

            for model_id, client in self.llm_clients.items():
                # Check cache first
                cached = cache.get(prompt, model_id)
                if cached:
                    model_responses[model_id] = cached.response_text
                    gt_losses[model_id] = cached.loss
                    continue

                try:
                    response = client.generate(
                        prompt, max_tokens=self.max_tokens, temperature=self.temperature
                    )
                    model_responses[model_id] = response.text

                    # Compute ground-truth loss if available
                    loss = 0.5  # default: unknown
                    if gt and self.metric_fn:
                        loss = self.metric_fn(response.text, gt)
                    gt_losses[model_id] = loss

                    # Populate cache
                    cache.add(prompt, model_id, response.text, loss,
                              latency_ms=response.latency_ms,
                              tokens_used=response.tokens_used)

                except Exception as e:
                    logger.warning(f"Failed {model_id} on prompt {idx}: {e}")

            # Step 2: LLM judge scoring
            judge_scores: dict[str, int] = {}
            if use_judge and self.judge and model_responses:
                for model_id, resp in model_responses.items():
                    try:
                        ps = self.judge.rate(prompt, model_id, resp)
                        judge_scores[model_id] = ps.score
                    except Exception as e:
                        logger.warning(f"Judge failed on {model_id}: {e}")

            # Step 3: Build augmented sample
            aug = AugmentedSample(
                prompt=prompt,
                ground_truth=gt,
                model_responses=model_responses,
                ground_truth_losses=gt_losses,
                judge_scores=judge_scores,
                category=sample.category,
            )
            samples.append(aug)

            # Step 4: Generate preference pairs
            model_ids = list(model_responses.keys())
            for i, mid_a in enumerate(model_ids):
                for mid_b in model_ids[i + 1:]:
                    pair = self._create_preference(mid_a, mid_b, gt_losses, judge_scores, prompt)
                    if pair:
                        preferences.add(pair)

        logger.info(
            f"Augmentation complete: {len(samples)} samples, "
            f"{len(preferences)} preference pairs"
        )
        return samples, preferences

    def augment_from_cache(
        self,
        cache: ResponseCache,
        prompts: list[str],
        use_judge: bool = True,
    ) -> PreferenceDataset:
        """
        Generate preference pairs from an existing response cache.

        Faster than full augmentation — no LLM generation calls needed,
        only optional judge calls.

        Args:
            cache: Pre-populated ResponseCache.
            prompts: List of prompts to process.
            use_judge: Whether to run judge on cached responses.

        Returns:
            PreferenceDataset with generated pairs.
        """
        preferences = PreferenceDataset(name="cache_augmented")

        for prompt in prompts:
            models = cache.get_all_models(prompt)
            if len(models) < 2:
                continue

            gt_losses = {mid: e.loss for mid, e in models.items()}
            judge_scores: dict[str, int] = {}

            if use_judge and self.judge:
                for mid, entry in models.items():
                    try:
                        ps = self.judge.rate(prompt, mid, entry.response_text)
                        judge_scores[mid] = ps.score
                    except Exception:
                        pass

            model_ids = list(models.keys())
            for i, mid_a in enumerate(model_ids):
                for mid_b in model_ids[i + 1:]:
                    pair = self._create_preference(
                        mid_a, mid_b, gt_losses, judge_scores, prompt
                    )
                    if pair:
                        preferences.add(pair)

        return preferences

    def _create_preference(
        self,
        model_a: str,
        model_b: str,
        gt_losses: dict[str, float],
        judge_scores: dict[str, int],
        prompt: str,
    ) -> Optional:
        """Create a preference pair from ground-truth and/or judge scores."""
        from .preference_data import PreferencePair

        loss_a = gt_losses.get(model_a)
        loss_b = gt_losses.get(model_b)
        score_a = judge_scores.get(model_a)
        score_b = judge_scores.get(model_b)

        winner = None
        loser = None
        source = "benchmark"
        confidence = 1.0

        # Ground-truth signal (strongest)
        if loss_a is not None and loss_b is not None:
            if loss_a == 0.0 and loss_b > 0.0:
                winner, loser = model_a, model_b
            elif loss_b == 0.0 and loss_a > 0.0:
                winner, loser = model_b, model_a

        # Judge signal (use when GT is tied or unavailable)
        if winner is None and score_a is not None and score_b is not None:
            diff = abs(score_a - score_b)
            if diff >= 1:  # need at least 1 point difference
                if score_a > score_b:
                    winner, loser = model_a, model_b
                else:
                    winner, loser = model_b, model_a
                source = "judge"
                confidence = min(diff / 4.0, 1.0)

        if winner and loser:
            return PreferencePair(
                prompt=prompt,
                winner_model=winner,
                loser_model=loser,
                source=source,
                confidence=confidence,
            )
        return None
