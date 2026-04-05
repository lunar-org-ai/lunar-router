"""
Core router evaluator.

Sweeps cost_weight (lambda) values, producing Pareto curves and computing
all routing metrics against baselines. Works entirely from cached responses
— no LLM calls needed during evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import json
import logging
import numpy as np

from ..router.uniroute import UniRouteRouter, RoutingDecision
from ..models.llm_profile import LLMProfile
from ..data.dataset import PromptDataset
from .response_cache import ResponseCache
from .baselines import (
    RandomBaseline,
    OracleBaseline,
    AlwaysStrongBaseline,
    AlwaysWeakBaseline,
    BaselineResult,
)
from .metrics import (
    compute_auroc,
    compute_apgr,
    compute_cpt,
    compute_pgr_at_savings,
    compute_win_rate,
    RoutingMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A single point on the cost-quality Pareto curve."""

    lambda_value: float
    quality: float  # accuracy = 1 - avg_loss
    avg_cost: float  # average cost per 1k tokens
    strong_model_fraction: float  # fraction of prompts sent to strong model
    model_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result with Pareto curve and metrics."""

    metrics: RoutingMetrics
    pareto_curve: list[ParetoPoint]
    baseline_quality: dict[str, float]  # baseline_name -> quality
    baseline_cost: dict[str, float]  # baseline_name -> avg cost
    dataset_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "pareto_curve": [
                {
                    "lambda": p.lambda_value,
                    "quality": p.quality,
                    "avg_cost": p.avg_cost,
                    "strong_model_fraction": p.strong_model_fraction,
                    "model_distribution": p.model_distribution,
                }
                for p in self.pareto_curve
            ],
            "baseline_quality": self.baseline_quality,
            "baseline_cost": self.baseline_cost,
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        """Save evaluation result to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Evaluation: {self.dataset_name} ===",
            self.metrics.summary(),
            "",
            "Baselines:",
        ]
        for name in self.baseline_quality:
            q = self.baseline_quality[name]
            c = self.baseline_cost.get(name, 0.0)
            lines.append(f"  {name:20s}: quality={q:.2%}, cost=${c:.6f}/1k")
        lines.append("")
        lines.append(f"Pareto curve: {len(self.pareto_curve)} points")
        if self.pareto_curve:
            best = max(self.pareto_curve, key=lambda p: p.quality)
            cheapest = min(self.pareto_curve, key=lambda p: p.avg_cost)
            lines.append(f"  Best quality:  {best.quality:.2%} at lambda={best.lambda_value:.2f}")
            lines.append(f"  Lowest cost:   ${cheapest.avg_cost:.6f}/1k at lambda={cheapest.lambda_value:.2f}")
        return "\n".join(lines)


class RouterEvaluator:
    """
    Evaluates a router's cost-quality tradeoff on a cached benchmark.

    Usage:
        evaluator = RouterEvaluator(router, cache, profiles)
        result = evaluator.evaluate(dataset)
        print(result.summary())
        print(result.metrics.auroc)
    """

    def __init__(
        self,
        router: UniRouteRouter,
        cache: ResponseCache,
        profiles: list[LLMProfile],
        lambda_range: Optional[tuple[float, float]] = None,
        lambda_steps: int = 20,
    ):
        """
        Args:
            router: The UniRouteRouter to evaluate.
            cache: ResponseCache with pre-computed model responses.
            profiles: LLMProfiles for the models being evaluated.
            lambda_range: (min, max) for lambda sweep. Default (0.0, 10.0).
            lambda_steps: Number of lambda values to evaluate.
        """
        self.router = router
        self.cache = cache
        self.profiles = profiles
        self._profile_map = {p.model_id: p for p in profiles}
        self._lambda_min, self._lambda_max = lambda_range or (0.0, 10.0)
        self._lambda_steps = lambda_steps

    def evaluate(
        self,
        dataset: PromptDataset,
        dataset_name: str = "",
    ) -> EvaluationResult:
        """
        Run full evaluation: Pareto sweep + baselines + metrics.

        Args:
            dataset: PromptDataset with prompts (ground_truth not used directly —
                     correctness comes from the cache).
            dataset_name: Name for labeling results.

        Returns:
            EvaluationResult with Pareto curve and aggregated metrics.
        """
        prompts = [s.prompt for s in dataset.samples]
        prompt_hashes = [ResponseCache.hash_prompt(p) for p in prompts]

        # Filter to prompts that have cached responses for all models
        valid_hashes = []
        valid_prompts = []
        model_ids = {p.model_id for p in self.profiles}
        for ph, prompt in zip(prompt_hashes, prompts):
            cached = self.cache.get_all_models_by_hash(ph)
            if model_ids.issubset(cached.keys()):
                valid_hashes.append(ph)
                valid_prompts.append(prompt)

        if not valid_hashes:
            raise ValueError(
                f"No prompts have cached responses for all models. "
                f"Cache has {len(self.cache)} entries, need models: {model_ids}"
            )

        logger.info(
            f"Evaluating on {len(valid_hashes)}/{len(prompts)} prompts "
            f"with full model coverage"
        )

        # 1. Compute baselines
        strong_baseline = AlwaysStrongBaseline(self.profiles)
        weak_baseline = AlwaysWeakBaseline(self.profiles)
        random_baseline = RandomBaseline(self.profiles)
        oracle_baseline = OracleBaseline(self.profiles)

        strong_results = strong_baseline.evaluate(self.cache, valid_hashes)
        weak_results = weak_baseline.evaluate(self.cache, valid_hashes)
        random_results = random_baseline.evaluate(self.cache, valid_hashes)
        oracle_results = oracle_baseline.evaluate(self.cache, valid_hashes)

        quality_strong = _quality(strong_results)
        quality_weak = _quality(weak_results)
        quality_random = _quality(random_results)
        quality_oracle = _quality(oracle_results)

        baseline_quality = {
            "always_strong": quality_strong,
            "always_weak": quality_weak,
            "random": quality_random,
            "oracle": quality_oracle,
        }
        baseline_cost = {
            "always_strong": _avg_cost(strong_results),
            "always_weak": _avg_cost(weak_results),
            "random": _avg_cost(random_results),
            "oracle": _avg_cost(oracle_results),
        }

        # 2. Sweep lambda for Pareto curve
        lambda_values = np.linspace(
            self._lambda_min, self._lambda_max, self._lambda_steps
        ).tolist()

        pareto_curve: list[ParetoPoint] = []
        default_lambda_results: Optional[list[tuple[RoutingDecision, float]]] = None

        for lam in lambda_values:
            point, per_sample = self._evaluate_at_lambda(
                valid_prompts, valid_hashes, lam, strong_baseline.model_id
            )
            pareto_curve.append(point)

            # Store results at the router's default lambda for AUROC/metrics
            if abs(lam - self.router.cost_weight) < 1e-9:
                default_lambda_results = per_sample

        # If default lambda wasn't in the sweep, evaluate it separately
        if default_lambda_results is None:
            _, default_lambda_results = self._evaluate_at_lambda(
                valid_prompts, valid_hashes,
                self.router.cost_weight, strong_baseline.model_id,
            )

        # 3. Compute AUROC
        # Score: router's error gap (strong_error - weak_error) for each prompt
        # Label: strong model was needed (weak wrong AND strong right)
        auroc_scores = []
        auroc_labels = []
        for ph in valid_hashes:
            models = self.cache.get_all_models_by_hash(ph)
            strong_entry = models.get(strong_baseline.model_id)
            weak_entry = models.get(weak_baseline.model_id)
            if strong_entry and weak_entry:
                # Score: how much the router thinks this prompt needs the strong model
                # Use the error difference from profiles as the score
                score = weak_entry.loss - strong_entry.loss
                # Label: strong was actually needed
                label = weak_entry.loss > 0.0 and strong_entry.loss == 0.0
                auroc_scores.append(score)
                auroc_labels.append(label)

        auroc = compute_auroc(auroc_scores, auroc_labels)

        # 4. Compute metrics at default lambda
        router_correct = [loss == 0.0 for _, loss in default_lambda_results]
        default_point = next(
            (p for p in pareto_curve
             if abs(p.lambda_value - self.router.cost_weight) < 1e-9),
            pareto_curve[0] if pareto_curve else None,
        )
        quality_at_default = default_point.quality if default_point else quality_random

        # Pareto points for CPT/PGR: (quality, strong_model_fraction)
        pareto_pairs = [(p.quality, p.strong_model_fraction) for p in pareto_curve]

        metrics = RoutingMetrics(
            auroc=auroc,
            apgr=compute_apgr(quality_at_default, quality_weak, quality_strong),
            win_rate=compute_win_rate(router_correct),
            cpt_50=compute_cpt(pareto_pairs, 0.50, quality_strong, quality_weak),
            cpt_75=compute_cpt(pareto_pairs, 0.75, quality_strong, quality_weak),
            cpt_90=compute_cpt(pareto_pairs, 0.90, quality_strong, quality_weak),
            cpt_95=compute_cpt(pareto_pairs, 0.95, quality_strong, quality_weak),
            pgr_at_25_savings=compute_pgr_at_savings(
                pareto_pairs, 0.25, quality_strong, quality_weak
            ),
            pgr_at_50_savings=compute_pgr_at_savings(
                pareto_pairs, 0.50, quality_strong, quality_weak
            ),
            pgr_at_75_savings=compute_pgr_at_savings(
                pareto_pairs, 0.75, quality_strong, quality_weak
            ),
            quality_strong=quality_strong,
            quality_weak=quality_weak,
            strong_model=strong_baseline.model_id,
            weak_model=weak_baseline.model_id,
            num_samples=len(valid_hashes),
        )

        return EvaluationResult(
            metrics=metrics,
            pareto_curve=pareto_curve,
            baseline_quality=baseline_quality,
            baseline_cost=baseline_cost,
            dataset_name=dataset_name,
        )

    def _evaluate_at_lambda(
        self,
        prompts: list[str],
        prompt_hashes: list[str],
        lambda_value: float,
        strong_model_id: str,
    ) -> tuple[ParetoPoint, list[tuple[RoutingDecision, float]]]:
        """Evaluate router at a specific lambda, return Pareto point + per-sample data."""
        total_loss = 0.0
        total_cost = 0.0
        strong_count = 0
        model_counts: dict[str, int] = {}
        per_sample: list[tuple[RoutingDecision, float]] = []

        for prompt, ph in zip(prompts, prompt_hashes):
            decision = self.router.route(
                prompt, cost_weight_override=lambda_value
            )

            # Look up cached correctness for the selected model
            entry = self.cache.get_by_hash(ph, decision.selected_model)
            if entry is None:
                continue

            loss = entry.loss
            total_loss += loss

            profile = self._profile_map.get(decision.selected_model)
            cost = profile.cost_per_1k_tokens if profile else 0.0
            total_cost += cost

            if decision.selected_model == strong_model_id:
                strong_count += 1

            model_counts[decision.selected_model] = (
                model_counts.get(decision.selected_model, 0) + 1
            )
            per_sample.append((decision, loss))

        n = len(per_sample)
        if n == 0:
            return ParetoPoint(lambda_value=lambda_value, quality=0.0,
                               avg_cost=0.0, strong_model_fraction=0.0), []

        quality = 1.0 - (total_loss / n)
        avg_cost = total_cost / n
        strong_fraction = strong_count / n
        model_dist = {m: c / n for m, c in model_counts.items()}

        point = ParetoPoint(
            lambda_value=lambda_value,
            quality=quality,
            avg_cost=avg_cost,
            strong_model_fraction=strong_fraction,
            model_distribution=model_dist,
        )

        return point, per_sample


def _quality(results: list[BaselineResult]) -> float:
    """Compute accuracy from baseline results."""
    if not results:
        return 0.0
    return 1.0 - sum(r.loss for r in results) / len(results)


def _avg_cost(results: list[BaselineResult]) -> float:
    """Compute average cost from baseline results."""
    if not results:
        return 0.0
    return sum(r.cost_per_1k_tokens for r in results) / len(results)
