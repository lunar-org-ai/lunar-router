"""
Auto-Training Orchestrator for continuous router improvement.

Chains the full improvement cycle:
1. Evaluate current router (baseline)
2. Augment training data (golden labels + LLM judge)
3. Retrain Psi vectors with augmented data
4. Evaluate new router
5. Promote if improved, rollback if not

This is the "self-improving router" — run it periodically or after
collecting enough production traces.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any
import json
import logging
import time

import numpy as np

from ..core.clustering import ClusterAssigner
from ..core.embeddings import PromptEmbedder
from ..core.metrics import MetricType, get_metric
from ..data.dataset import PromptDataset
from ..models.llm_client import LLMClient
from ..models.llm_profile import LLMProfile
from ..models.llm_registry import LLMRegistry
from ..router.uniroute import UniRouteRouter
from ..evaluation.response_cache import ResponseCache
from ..evaluation.evaluator import RouterEvaluator, EvaluationResult
from ..evaluation.metrics import RoutingMetrics
from ..augmentation.judge import LLMJudge
from ..augmentation.golden_augmenter import GoldenAugmenter
from ..augmentation.preference_data import PreferenceDataset
from ..feedback.trace_to_training import TraceToTraining, TraceRecord
from ..feedback.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


@dataclass
class AutoTrainConfig:
    """Configuration for auto-training."""

    # Augmentation
    use_judge: bool = True
    max_augmentation_samples: int = 500

    # Feedback blending
    production_alpha: float = 0.3  # weight for production data in Psi blend

    # Quality gates
    min_auroc_improvement: float = 0.0  # must improve AUROC by at least this
    min_win_rate: float = 0.5  # new router must exceed this win rate
    max_error_rate_increase: float = 0.05  # any model's error rate can't jump more than this

    # Evaluation
    lambda_steps: int = 20
    lambda_range: tuple[float, float] = (0.0, 10.0)

    # Output
    output_dir: Optional[str] = None
    keep_history: bool = True


@dataclass
class AutoTrainResult:
    """Result of an auto-training cycle."""

    promoted: bool  # whether new weights were promoted
    reason: str  # why promoted or not

    baseline_metrics: Optional[RoutingMetrics] = None
    new_metrics: Optional[RoutingMetrics] = None

    preference_pairs_generated: int = 0
    augmented_samples: int = 0
    production_traces_used: int = 0

    old_profiles: list[LLMProfile] = field(default_factory=list)
    new_profiles: list[LLMProfile] = field(default_factory=list)

    duration_ms: float = 0.0
    output_path: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"Auto-Training Result: {'PROMOTED' if self.promoted else 'REJECTED'}",
            f"  Reason: {self.reason}",
            f"  Duration: {self.duration_ms / 1000:.1f}s",
        ]
        if self.baseline_metrics and self.new_metrics:
            b = self.baseline_metrics
            n = self.new_metrics
            lines.extend([
                f"  Baseline AUROC:  {b.auroc:.4f}  →  New: {n.auroc:.4f} ({n.auroc - b.auroc:+.4f})",
                f"  Baseline APGR:   {b.apgr:.2%}  →  New: {n.apgr:.2%}",
                f"  Baseline Win%:   {b.win_rate:.2%}  →  New: {n.win_rate:.2%}",
            ])
        lines.extend([
            f"  Preference pairs: {self.preference_pairs_generated}",
            f"  Augmented samples: {self.augmented_samples}",
            f"  Production traces: {self.production_traces_used}",
        ])
        if self.output_path:
            lines.append(f"  Weights: {self.output_path}")
        return "\n".join(lines)


class AutoTrainer:
    """
    Self-improving router trainer.

    Orchestrates the full cycle: evaluate → augment → retrain → evaluate → promote.

    Usage:
        trainer = AutoTrainer(
            embedder=embedder,
            cluster_assigner=assigner,
            profiles=profiles,
            eval_dataset=val_data,
            eval_cache=cache,
        )

        # Auto-train with augmented data
        result = trainer.train(
            llm_clients=[gpt4o, mini, llama],
            judge_client=gpt4o,
            augmentation_dataset=train_data,
        )
        print(result.summary())

        # Auto-train with production traces
        result = trainer.train_from_traces(traces)
        print(result.summary())

        # Get the improved router
        if result.promoted:
            router = trainer.get_router()
    """

    def __init__(
        self,
        embedder: PromptEmbedder,
        cluster_assigner: ClusterAssigner,
        profiles: list[LLMProfile],
        eval_dataset: PromptDataset,
        eval_cache: ResponseCache,
        cost_weight: float = 0.0,
        config: Optional[AutoTrainConfig] = None,
    ):
        self.embedder = embedder
        self.assigner = cluster_assigner
        self.profiles = list(profiles)
        self.eval_dataset = eval_dataset
        self.eval_cache = eval_cache
        self.cost_weight = cost_weight
        self.config = config or AutoTrainConfig()
        self._history: list[AutoTrainResult] = []

    def get_router(self, profiles: Optional[list[LLMProfile]] = None) -> UniRouteRouter:
        """Build a UniRouteRouter from current or given profiles."""
        profs = profiles or self.profiles
        registry = LLMRegistry()
        for p in profs:
            registry.register(p)
        return UniRouteRouter(
            embedder=self.embedder,
            cluster_assigner=self.assigner,
            registry=registry,
            cost_weight=self.cost_weight,
        )

    def evaluate(self, profiles: Optional[list[LLMProfile]] = None) -> EvaluationResult:
        """Evaluate a set of profiles on the eval dataset."""
        profs = profiles or self.profiles
        router = self.get_router(profs)
        evaluator = RouterEvaluator(
            router=router,
            cache=self.eval_cache,
            profiles=profs,
            lambda_range=self.config.lambda_range,
            lambda_steps=self.config.lambda_steps,
        )
        return evaluator.evaluate(self.eval_dataset)

    def train(
        self,
        llm_clients: Optional[list[LLMClient]] = None,
        judge_client: Optional[LLMClient] = None,
        augmentation_dataset: Optional[PromptDataset] = None,
        production_traces: Optional[list[TraceRecord]] = None,
        quality_flags: Optional[dict[str, bool]] = None,
        metric_fn: Optional[Callable] = None,
    ) -> AutoTrainResult:
        """
        Run a full auto-training cycle.

        Args:
            llm_clients: Models to generate augmented responses (optional).
            judge_client: LLM to judge response quality (optional).
            augmentation_dataset: Dataset for augmentation (optional).
            production_traces: Traces from ClickHouse (optional).
            quality_flags: TraceScanner quality flags (optional).
            metric_fn: Ground-truth loss function (optional).

        Returns:
            AutoTrainResult with promotion decision and metrics.
        """
        start = time.time()
        cfg = self.config

        # Step 1: Evaluate baseline
        logger.info("Step 1: Evaluating baseline router...")
        baseline_result = self.evaluate()
        baseline_metrics = baseline_result.metrics

        # Step 2: Augment data and generate preference pairs
        preferences = PreferenceDataset(name="auto_train")
        augmented_count = 0

        # 2a: Golden augmentation from dataset
        if llm_clients and augmentation_dataset:
            logger.info("Step 2a: Running golden augmentation...")
            judge = LLMJudge(judge_client) if judge_client and cfg.use_judge else None
            augmenter = GoldenAugmenter(
                llm_clients=llm_clients,
                judge=judge,
                metric_fn=metric_fn,
            )

            # Limit samples
            aug_data = augmentation_dataset
            if len(aug_data) > cfg.max_augmentation_samples:
                aug_data = PromptDataset(
                    aug_data.samples[:cfg.max_augmentation_samples],
                    name="aug_subset",
                )

            samples, aug_prefs = augmenter.augment(aug_data, use_judge=cfg.use_judge)
            augmented_count = len(samples)
            preferences.pairs.extend(aug_prefs.pairs)

        # 2b: Preferences from eval cache
        logger.info("Step 2b: Generating preferences from eval cache...")
        cache_prefs = PreferenceDataset(name="cache")
        cache_prefs.add_from_cache(self.eval_cache)
        preferences.pairs.extend(cache_prefs.pairs)

        # Step 3: Compute new Psi vectors
        logger.info("Step 3: Computing improved Psi vectors...")
        new_profiles = self._improve_profiles(preferences, production_traces, quality_flags)

        # Step 4: Evaluate new router
        logger.info("Step 4: Evaluating improved router...")
        new_result = self.evaluate(new_profiles)
        new_metrics = new_result.metrics

        # Step 5: Quality gate — promote or reject
        promoted, reason = self._quality_gate(baseline_metrics, new_metrics, new_profiles)

        if promoted:
            self.profiles = new_profiles
            logger.info(f"PROMOTED: {reason}")
        else:
            logger.info(f"REJECTED: {reason}")

        # Save if output_dir configured
        output_path = None
        if promoted and cfg.output_dir:
            output_path = self._save_weights(new_profiles, cfg.output_dir)

        duration = (time.time() - start) * 1000

        result = AutoTrainResult(
            promoted=promoted,
            reason=reason,
            baseline_metrics=baseline_metrics,
            new_metrics=new_metrics,
            preference_pairs_generated=len(preferences),
            augmented_samples=augmented_count,
            production_traces_used=len(production_traces) if production_traces else 0,
            old_profiles=self.profiles if not promoted else list(profiles for profiles in [self.profiles]),
            new_profiles=new_profiles,
            duration_ms=duration,
            output_path=str(output_path) if output_path else None,
        )

        self._history.append(result)
        return result

    def train_from_traces(
        self,
        traces: list[TraceRecord],
        quality_flags: Optional[dict[str, bool]] = None,
    ) -> AutoTrainResult:
        """
        Quick auto-training from production traces only (no LLM calls needed).

        Blends production error rates with existing Psi vectors and
        promotes if the blended router is better on the eval benchmark.
        """
        return self.train(production_traces=traces, quality_flags=quality_flags)

    def train_from_cache(self) -> AutoTrainResult:
        """
        Auto-training from eval cache only (no LLM calls needed).

        Uses existing cached responses to generate preference pairs and
        refine Psi vectors via preference-weighted updates.
        """
        return self.train()

    def _improve_profiles(
        self,
        preferences: PreferenceDataset,
        traces: Optional[list[TraceRecord]],
        quality_flags: Optional[dict[str, bool]],
    ) -> list[LLMProfile]:
        """Compute improved Psi vectors from all available signal."""
        new_profiles = list(self.profiles)

        # Apply preference-based Psi refinement
        if len(preferences) > 0:
            new_profiles = self._refine_from_preferences(new_profiles, preferences)

        # Blend with production traces
        if traces:
            num_clusters = self.profiles[0].num_clusters if self.profiles else 100
            converter = TraceToTraining(
                num_clusters=num_clusters,
                quality_flags=quality_flags or {},
            )
            converter.add_traces(traces)
            new_profiles = converter.blend_with_profiles(
                new_profiles, alpha=self.config.production_alpha
            )

        return new_profiles

    def _refine_from_preferences(
        self,
        profiles: list[LLMProfile],
        preferences: PreferenceDataset,
    ) -> list[LLMProfile]:
        """
        Refine Psi vectors using preference pairs.

        For each (prompt, winner, loser) pair:
        - Decrease winner's error estimate slightly
        - Increase loser's error estimate slightly

        This nudges the router toward preferring winners on similar prompts.
        """
        lr = 0.01  # learning rate for preference updates
        profile_map = {p.model_id: p for p in profiles}

        # Accumulate adjustments
        adjustments: dict[str, np.ndarray] = {
            p.model_id: np.zeros_like(p.psi_vector) for p in profiles
        }
        adj_counts: dict[str, np.ndarray] = {
            p.model_id: np.zeros_like(p.psi_vector) for p in profiles
        }

        for pair in preferences:
            # We don't know the exact cluster for each prompt in preferences,
            # so we apply a small global adjustment weighted by confidence
            if pair.winner_model in adjustments:
                adjustments[pair.winner_model] -= lr * pair.confidence
                adj_counts[pair.winner_model] += 1
            if pair.loser_model in adjustments:
                adjustments[pair.loser_model] += lr * pair.confidence
                adj_counts[pair.loser_model] += 1

        # Apply adjustments
        refined = []
        for profile in profiles:
            mid = profile.model_id
            count = adj_counts[mid]
            total = count.sum()

            if total > 0:
                # Normalize by number of preferences
                adj = adjustments[mid] / max(total, 1)
                new_psi = np.clip(profile.psi_vector + adj, 0.0, 1.0)
            else:
                new_psi = profile.psi_vector.copy()

            refined.append(LLMProfile(
                model_id=mid,
                psi_vector=new_psi,
                cost_per_1k_tokens=profile.cost_per_1k_tokens,
                num_validation_samples=profile.num_validation_samples,
                cluster_sample_counts=profile.cluster_sample_counts.copy(),
                metadata={**profile.metadata, "preference_refined": True},
            ))

        return refined

    def _quality_gate(
        self,
        baseline: RoutingMetrics,
        new: RoutingMetrics,
        new_profiles: list[LLMProfile],
    ) -> tuple[bool, str]:
        """Check if the new router passes quality gates."""
        cfg = self.config

        # Gate 1: Win rate minimum
        if new.win_rate < cfg.min_win_rate:
            return False, f"Win rate {new.win_rate:.2%} below minimum {cfg.min_win_rate:.2%}"

        # Gate 2: AUROC must not decrease significantly
        auroc_delta = new.auroc - baseline.auroc
        if auroc_delta < -0.01:  # allow tiny regression
            return False, f"AUROC decreased by {abs(auroc_delta):.4f}"

        # Gate 3: AUROC improvement threshold
        if auroc_delta < cfg.min_auroc_improvement:
            # Still promote if win_rate or APGR improved
            if new.win_rate > baseline.win_rate or new.apgr > baseline.apgr:
                return True, (
                    f"AUROC stable ({auroc_delta:+.4f}), "
                    f"win_rate improved ({baseline.win_rate:.2%} -> {new.win_rate:.2%})"
                )
            return False, f"AUROC improvement {auroc_delta:.4f} below threshold {cfg.min_auroc_improvement}"

        # Gate 4: No model's error rate should spike
        old_map = {p.model_id: p for p in self.profiles}
        for new_p in new_profiles:
            old_p = old_map.get(new_p.model_id)
            if old_p:
                increase = new_p.overall_error_rate - old_p.overall_error_rate
                if increase > cfg.max_error_rate_increase:
                    return False, (
                        f"{new_p.model_id} error rate spiked by {increase:.3f} "
                        f"(max allowed: {cfg.max_error_rate_increase})"
                    )

        return True, f"AUROC improved by {auroc_delta:+.4f}, all gates passed"

    def _save_weights(self, profiles: list[LLMProfile], output_dir: str) -> Path:
        """Save improved profiles to disk."""
        out = Path(output_dir) / "profiles"
        out.mkdir(parents=True, exist_ok=True)
        for p in profiles:
            safe_name = p.model_id.replace("/", "_")
            p.save(out / f"{safe_name}.json")
        logger.info(f"Saved {len(profiles)} profiles to {out}")
        return Path(output_dir)

    @property
    def history(self) -> list[AutoTrainResult]:
        """History of all auto-training cycles."""
        return self._history
