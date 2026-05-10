"""Core router evaluator.

Sweeps cost_weight (lambda) values, producing Pareto curves and computing
all routing metrics against baselines. Works **entirely from cached
responses** — no LLM calls during evaluation.

Adaptations vs the reference impl:
- Cache miss raises ``CacheGapError`` with the specific (prompt, model)
  pair instead of silently filtering. Operators run
  ``tools/populate_response_cache.py`` to fix.
- AUROC is computed from the **router's predicted strong-benefit**
  (Psi-driven) rather than from cache loss differences. This means
  empty-Psi configs return ~0.5 (random ranking) and fitted configs
  that learned something return > 0.5 — what P15.3.6's DoD asserts.
- Cache hit ratio is logged at INFO on every ``evaluate()`` call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from router.data.dataset import PromptDataset
from router.errors import RouterError
from router.evaluation.baselines import (
    AlwaysStrongBaseline,
    AlwaysWeakBaseline,
    BaselineResult,
    OracleBaseline,
    RandomBaseline,
)
from router.evaluation.cache import ResponseCache
from router.evaluation.metrics import (
    RoutingMetrics,
    compute_apgr,
    compute_auroc,
    compute_cpt,
    compute_pgr_at_savings,
    compute_win_rate,
)
from router.models.llm_profile import LLMProfile
from router.uniroute import RoutingDecision, UniRouteRouter


logger = logging.getLogger("router.evaluation.evaluator")


class CacheGapError(RouterError):
    """Raised when the cache is missing a (prompt, model) entry needed for eval.

    The operator should run ``tools/populate_response_cache.py`` (or the
    P15.3.7 proposer's auto-populate hook) to fill the gap.
    """


@dataclass
class ParetoPoint:
    """A single point on the cost-quality Pareto curve."""

    lambda_value: float
    quality: float
    avg_cost: float
    strong_model_fraction: float
    model_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result with Pareto curve and metrics."""

    metrics: RoutingMetrics
    pareto_curve: list[ParetoPoint]
    baseline_quality: dict[str, float]
    baseline_cost: dict[str, float]
    dataset_name: str = ""
    cache_hit_ratio: float = 1.0
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
            "cache_hit_ratio": self.cache_hit_ratio,
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        lines = [
            f"=== Evaluation: {self.dataset_name} ===",
            self.metrics.summary(),
            f"  cache_hit_ratio={self.cache_hit_ratio:.2f}",
            "",
            "Baselines:",
        ]
        for name in self.baseline_quality:
            q = self.baseline_quality[name]
            c = self.baseline_cost.get(name, 0.0)
            lines.append(f"  {name:20s}: quality={q:.2%}, cost=${c:.6f}/1k")
        if self.pareto_curve:
            best = max(self.pareto_curve, key=lambda p: p.quality)
            cheapest = min(self.pareto_curve, key=lambda p: p.avg_cost)
            lines.append("")
            lines.append(f"Pareto curve: {len(self.pareto_curve)} points")
            lines.append(
                f"  best:     quality={best.quality:.2%} at lambda={best.lambda_value:.2f}"
            )
            lines.append(
                f"  cheapest: ${cheapest.avg_cost:.6f}/1k at lambda={cheapest.lambda_value:.2f}"
            )
        return "\n".join(lines)


class RouterEvaluator:
    """Evaluates a router's cost-quality tradeoff on a cached benchmark.

    All evaluation reads from the cache; no LLM calls happen here. If a
    (prompt, model) is missing, raises ``CacheGapError``.
    """

    def __init__(
        self,
        router: UniRouteRouter,
        cache: ResponseCache,
        profiles: list[LLMProfile],
        lambda_range: Optional[tuple[float, float]] = None,
        lambda_steps: int = 20,
    ):
        self.router = router
        self.cache = cache
        self.profiles = profiles
        self._profile_map = {p.model_id: p for p in profiles}
        self._lambda_min, self._lambda_max = lambda_range or (0.0, 10.0)
        self._lambda_steps = lambda_steps

    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: PromptDataset,
        dataset_name: str = "",
    ) -> EvaluationResult:
        """Run full evaluation: Pareto sweep + baselines + metrics."""
        prompts = [s.prompt for s in dataset.samples]
        prompt_hashes = [ResponseCache.hash_prompt(p) for p in prompts]

        # Cache coverage check — fail loudly on the first missing pair.
        model_ids = {p.model_id for p in self.profiles}
        total_lookups = 0
        cache_hits = 0
        for ph, prompt in zip(prompt_hashes, prompts):
            cached = self.cache.get_all_models_by_hash(ph)
            for mid in model_ids:
                total_lookups += 1
                if mid in cached:
                    cache_hits += 1
                else:
                    raise CacheGapError(
                        f"cache miss: prompt_hash={ph[:12]}... model={mid!r} "
                        f"(prompt: {prompt[:80]!r}). "
                        "Run tools/populate_response_cache.py to fill."
                    )

        cache_hit_ratio = cache_hits / total_lookups if total_lookups else 1.0
        logger.info(
            "evaluate dataset=%s prompts=%d models=%d cache_hit_ratio=%.2f",
            dataset_name,
            len(prompts),
            len(model_ids),
            cache_hit_ratio,
        )

        # 1. Baselines
        strong_baseline = AlwaysStrongBaseline(self.profiles)
        weak_baseline = AlwaysWeakBaseline(self.profiles)
        random_baseline = RandomBaseline(self.profiles)
        oracle_baseline = OracleBaseline(self.profiles)

        strong_results = strong_baseline.evaluate(self.cache, prompt_hashes)
        weak_results = weak_baseline.evaluate(self.cache, prompt_hashes)
        random_results = random_baseline.evaluate(self.cache, prompt_hashes)
        oracle_results = oracle_baseline.evaluate(self.cache, prompt_hashes)

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

        # 2. Pareto sweep
        lambda_values = np.linspace(
            self._lambda_min, self._lambda_max, self._lambda_steps
        ).tolist()

        pareto_curve: list[ParetoPoint] = []
        default_lambda_results: Optional[list[tuple[RoutingDecision, float]]] = None
        for lam in lambda_values:
            point, per_sample = self._evaluate_at_lambda(
                prompts, prompt_hashes, lam, strong_baseline.model_id
            )
            pareto_curve.append(point)
            if abs(lam - self.router.cost_weight) < 1e-9:
                default_lambda_results = per_sample
        if default_lambda_results is None:
            _, default_lambda_results = self._evaluate_at_lambda(
                prompts,
                prompt_hashes,
                self.router.cost_weight,
                strong_baseline.model_id,
            )

        # 3. AUROC — driven by the router's Psi prediction (so empty-config = 0.5).
        auroc = self._compute_router_driven_auroc(
            prompts,
            prompt_hashes,
            strong_baseline.model_id,
            weak_baseline.model_id,
        )

        # 4. Metrics at default lambda
        router_correct = [loss == 0.0 for _, loss in default_lambda_results]
        default_point = next(
            (
                p
                for p in pareto_curve
                if abs(p.lambda_value - self.router.cost_weight) < 1e-9
            ),
            pareto_curve[0] if pareto_curve else None,
        )
        quality_at_default = (
            default_point.quality if default_point else quality_random
        )
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
            num_samples=len(prompts),
        )

        return EvaluationResult(
            metrics=metrics,
            pareto_curve=pareto_curve,
            baseline_quality=baseline_quality,
            baseline_cost=baseline_cost,
            dataset_name=dataset_name,
            cache_hit_ratio=cache_hit_ratio,
            metadata={
                "quality_oracle": quality_oracle,
                "lambda_range": [self._lambda_min, self._lambda_max],
                "lambda_steps": self._lambda_steps,
            },
        )

    # ------------------------------------------------------------------

    def _evaluate_at_lambda(
        self,
        prompts: list[str],
        prompt_hashes: list[str],
        lambda_value: float,
        strong_model_id: str,
    ) -> tuple[ParetoPoint, list[tuple[RoutingDecision, float]]]:
        total_loss = 0.0
        total_cost = 0.0
        strong_count = 0
        model_counts: dict[str, int] = {}
        per_sample: list[tuple[RoutingDecision, float]] = []

        for prompt, ph in zip(prompts, prompt_hashes):
            decision = self.router.route(
                prompt, cost_weight_override=lambda_value
            )
            entry = self.cache.get_by_hash(ph, decision.selected_model)
            if entry is None:
                raise CacheGapError(
                    f"cache miss after route: prompt_hash={ph[:12]}... "
                    f"selected={decision.selected_model!r}"
                )

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
            return (
                ParetoPoint(
                    lambda_value=lambda_value,
                    quality=0.0,
                    avg_cost=0.0,
                    strong_model_fraction=0.0,
                ),
                [],
            )

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

    def _compute_router_driven_auroc(
        self,
        prompts: list[str],
        prompt_hashes: list[str],
        strong_id: str,
        weak_id: str,
    ) -> float:
        """Score per prompt = router-predicted (weak_err - strong_err).

        With empty Psi (all zeros), every score is exactly 0 and the AUROC
        becomes 0.5 (no separation). With an informative Psi, prompts the
        router thinks need the strong model rank higher.
        """
        strong_profile = self._profile_map.get(strong_id)
        weak_profile = self._profile_map.get(weak_id)
        if strong_profile is None or weak_profile is None:
            return 0.5

        scores: list[float] = []
        labels: list[bool] = []

        for prompt, ph in zip(prompts, prompt_hashes):
            embedding = self.router.embedder.embed(prompt)
            cluster_result = self.router.cluster_assigner.assign(embedding)
            if self.router.use_soft_assignment:
                phi = cluster_result.probabilities
            else:
                phi = cluster_result.to_one_hot()

            err_strong = strong_profile.get_expected_error(phi)
            err_weak = weak_profile.get_expected_error(phi)
            score = err_weak - err_strong  # higher = router thinks strong helps more

            cached = self.cache.get_all_models_by_hash(ph)
            strong_loss = cached[strong_id].loss
            weak_loss = cached[weak_id].loss
            label = bool(weak_loss > 0.0 and strong_loss == 0.0)

            scores.append(score)
            labels.append(label)

        return compute_auroc(scores, labels)


# ---------------------------------------------------------------------------


def _quality(results: list[BaselineResult]) -> float:
    if not results:
        return 0.0
    return 1.0 - sum(r.loss for r in results) / len(results)


def _avg_cost(results: list[BaselineResult]) -> float:
    if not results:
        return 0.0
    return sum(r.cost_per_1k_tokens for r in results) / len(results)
