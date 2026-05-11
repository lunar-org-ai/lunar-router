"""Golden label augmentation for router training.

Generates model responses on benchmark data, evaluates with both
ground-truth metrics and the LLM judge, and produces a richer training
set than ground-truth labels alone.

Sources of preference signal (in priority order):
1. Ground-truth losses from ``metric_fn`` — strongest.
2. Judge pointwise scores — used when GT is tied or unavailable.

Cache parameter is duck-typed so we can drop in P15.3.6's ``ResponseCache``
later without circular imports. Anything with these methods works:
- ``.get(prompt, model_id) -> entry | None`` (entry has ``.response_text``,
  ``.loss``)
- ``.add(prompt, model_id, response_text, loss, **kw) -> None``
- ``.get_all_models(prompt) -> dict[model_id, entry]``

P15.3.5's deliverable adds **file persistence**: ``augment()`` returns a
written JSONL alongside the in-memory dataset, so downstream phases can
reload preference pairs without re-running the judge.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from router.augmentation.judge import LLMJudge
from router.augmentation.preference import PreferenceDataset, PreferencePair
from router.data.dataset import PromptDataset
from router.models.llm_client import LLMClient


logger = logging.getLogger("router.augmentation.goldens")


DEFAULT_OUTPUT_DIR = Path("evals") / "preference_pairs"
DEFAULT_MAX_SAMPLES = 500


@dataclass
class AugmentedSample:
    """A prompt with responses and scores from multiple models."""

    prompt: str
    ground_truth: Optional[str]
    model_responses: dict[str, str]
    ground_truth_losses: dict[str, float]
    judge_scores: dict[str, int]
    category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_model_by_gt(self) -> Optional[str]:
        if not self.ground_truth_losses:
            return None
        return min(self.ground_truth_losses, key=self.ground_truth_losses.get)

    @property
    def best_model_by_judge(self) -> Optional[str]:
        if not self.judge_scores:
            return None
        return max(self.judge_scores, key=self.judge_scores.get)


@dataclass
class AugmentationResult:
    """Output of GoldenAugmenter.augment().

    Carries the in-memory ``samples`` + ``preference_dataset`` plus the
    on-disk path of the persisted JSONL (None when ``persist=False``).
    """

    samples: list[AugmentedSample]
    preference_dataset: PreferenceDataset
    persisted_path: Optional[Path]


class GoldenAugmenter:
    """Generates augmented training data via model generation + judging."""

    def __init__(
        self,
        llm_clients: list[LLMClient],
        judge: Optional[LLMJudge] = None,
        metric_fn: Optional[Callable[[str, str], float]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        """Args:
            llm_clients: Models to generate responses with.
            judge: Optional LLMJudge for quality scoring.
            metric_fn: Ground-truth loss function (0=correct, 1=error).
            max_tokens / temperature: Generation knobs.
            max_samples: Cap dataset slice before any LLM calls (locked at
                500 by the P15.3 budget; configurable for tests).
            output_dir: Where persisted preference JSONLs land. Default
                evals/preference_pairs/ matches the existing eval output
                conventions.
        """
        self.llm_clients = {c.model_id: c for c in llm_clients}
        self.judge = judge
        self.metric_fn = metric_fn
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_samples = max_samples
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Full augmentation: generate + judge + persist
    # ------------------------------------------------------------------

    def augment(
        self,
        dataset: PromptDataset,
        *,
        use_judge: bool = True,
        cache=None,
        persist: bool = True,
    ) -> AugmentationResult:
        """Run full augmentation.

        Args:
            dataset: Source dataset with prompts (and optionally
                ground_truth labels).
            use_judge: Whether to run the LLM judge for pointwise scores.
            cache: Optional duck-typed ResponseCache. When provided,
                cached (prompt, model) responses are reused instead of
                regenerating. P15.3.6's ResponseCache plugs in here.
            persist: Whether to write the resulting PreferenceDataset to
                disk. Default True (locked).

        Returns:
            AugmentationResult with samples + preference_dataset +
            persisted_path.
        """
        sliced_samples = self._slice(dataset.samples)

        samples: list[AugmentedSample] = []
        preferences = PreferenceDataset(name="augmented")

        total = len(sliced_samples)
        for idx, sample in enumerate(sliced_samples):
            prompt = sample.prompt
            gt = sample.ground_truth

            if (idx + 1) % 50 == 0:
                logger.info("augmenting %d/%d", idx + 1, total)

            model_responses, gt_losses = self._collect_responses(
                prompt, gt, cache=cache
            )

            judge_scores: dict[str, int] = {}
            if use_judge and self.judge and model_responses:
                for model_id, resp in model_responses.items():
                    try:
                        ps = self.judge.rate(prompt, model_id, resp)
                        judge_scores[model_id] = ps.score
                    except Exception as e:
                        logger.warning("judge failed on %s: %s", model_id, e)

            samples.append(
                AugmentedSample(
                    prompt=prompt,
                    ground_truth=gt,
                    model_responses=model_responses,
                    ground_truth_losses=gt_losses,
                    judge_scores=judge_scores,
                    category=sample.category,
                )
            )

            for pair in self._iter_preference_pairs(
                prompt, model_responses.keys(), gt_losses, judge_scores
            ):
                preferences.add(pair)

        persisted = self._persist(preferences) if persist else None
        logger.info(
            "augmentation complete: %d samples, %d preference pairs (persisted=%s)",
            len(samples),
            len(preferences),
            persisted,
        )
        return AugmentationResult(
            samples=samples,
            preference_dataset=preferences,
            persisted_path=persisted,
        )

    # ------------------------------------------------------------------
    # Cache-only augmentation (cheap; for re-judging existing responses)
    # ------------------------------------------------------------------

    def augment_from_cache(
        self,
        cache,
        prompts: list[str],
        *,
        use_judge: bool = True,
        persist: bool = True,
    ) -> AugmentationResult:
        """Generate preference pairs from an already-populated cache.

        Faster than ``augment()`` — no LLM generation calls needed,
        only optional judge calls.
        """
        prompts = self._slice(prompts)
        preferences = PreferenceDataset(name="cache_augmented")
        samples: list[AugmentedSample] = []

        for prompt in prompts:
            models = cache.get_all_models(prompt)
            if len(models) < 2:
                continue

            model_responses = {mid: e.response_text for mid, e in models.items()}
            gt_losses = {mid: float(e.loss) for mid, e in models.items()}

            judge_scores: dict[str, int] = {}
            if use_judge and self.judge:
                for mid, resp in model_responses.items():
                    try:
                        ps = self.judge.rate(prompt, mid, resp)
                        judge_scores[mid] = ps.score
                    except Exception as e:
                        logger.warning("judge failed on %s: %s", mid, e)

            samples.append(
                AugmentedSample(
                    prompt=prompt,
                    ground_truth=None,
                    model_responses=model_responses,
                    ground_truth_losses=gt_losses,
                    judge_scores=judge_scores,
                )
            )
            for pair in self._iter_preference_pairs(
                prompt, model_responses.keys(), gt_losses, judge_scores
            ):
                preferences.add(pair)

        persisted = self._persist(preferences) if persist else None
        return AugmentationResult(
            samples=samples,
            preference_dataset=preferences,
            persisted_path=persisted,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _slice(self, items: list) -> list:
        """Apply the max_samples cap before any expensive work."""
        if len(items) > self.max_samples:
            logger.warning(
                "augment input %d exceeds max_samples=%d; truncating",
                len(items),
                self.max_samples,
            )
            return items[: self.max_samples]
        return items

    def _collect_responses(
        self,
        prompt: str,
        gt: Optional[str],
        *,
        cache,
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Generate (or cache-hit) one response per registered client.

        Returns (model_responses, gt_losses).
        """
        model_responses: dict[str, str] = {}
        gt_losses: dict[str, float] = {}

        for model_id, client in self.llm_clients.items():
            cached = cache.get(prompt, model_id) if cache is not None else None
            if cached is not None:
                model_responses[model_id] = cached.response_text
                gt_losses[model_id] = float(cached.loss)
                continue

            try:
                response = client.generate(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                model_responses[model_id] = response.text
                loss = 0.5  # default when GT or metric_fn is absent
                if gt and self.metric_fn:
                    loss = float(self.metric_fn(response.text, gt))
                gt_losses[model_id] = loss
                if cache is not None:
                    cache.add(
                        prompt,
                        model_id,
                        response.text,
                        loss,
                        latency_ms=getattr(response, "latency_ms", 0.0),
                        tokens_used=getattr(response, "tokens_used", 0),
                    )
            except Exception as e:
                logger.warning("generate failed for %s on prompt: %s", model_id, e)

        return model_responses, gt_losses

    def _iter_preference_pairs(
        self,
        prompt: str,
        model_ids,
        gt_losses: dict[str, float],
        judge_scores: dict[str, int],
    ):
        """Yield PreferencePairs for every (a, b) combo with a clear winner."""
        ids = list(model_ids)
        for i, mid_a in enumerate(ids):
            for mid_b in ids[i + 1:]:
                pair = self._pair_from_signals(
                    prompt, mid_a, mid_b, gt_losses, judge_scores
                )
                if pair is not None:
                    yield pair

    def _pair_from_signals(
        self,
        prompt: str,
        model_a: str,
        model_b: str,
        gt_losses: dict[str, float],
        judge_scores: dict[str, int],
    ) -> Optional[PreferencePair]:
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

        # Judge signal (when GT is tied or unavailable)
        if winner is None and score_a is not None and score_b is not None:
            diff = abs(score_a - score_b)
            if diff >= 1:
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

    def _persist(self, preferences: PreferenceDataset) -> Path:
        """Write the dataset to evals/preference_pairs/pp_<date>_<hash>.jsonl."""
        utc_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        body = "|".join(
            f"{p.winner_model}>{p.loser_model}:{p.prompt[:32]}"
            for p in preferences.pairs[:50]
        )
        short_hash = hashlib.md5(
            f"{utc_date}|{len(preferences)}|{body}".encode()
        ).hexdigest()[:8]
        path = self.output_dir / f"pp_{utc_date}_{short_hash}.jsonl"
        preferences.save(path)
        return path
