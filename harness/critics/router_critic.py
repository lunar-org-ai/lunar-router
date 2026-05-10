"""Router quality-gate critic.

Reads the candidate payload off the Proposal's first Mutation, builds
both the candidate and current routers, runs ``RouterEvaluator`` on
each, and applies the locked P15.3 quality gate:

  - delta_auroc           ≥ min_auroc_improvement   (default 0.0)
  - candidate_win_rate    ≥ min_win_rate            (default 0.5)
  - delta_avg_error       ≤ max_error_rate_increase (default 0.05)

Cold-start (no current config exists yet): the candidate is compared
against an implicit AUROC=0.5 / quality_random baseline. A candidate
that's no better than random gets rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from harness.critics.base import Critic, CriticStage, register_critic
from harness.types import CriticContext, CriticVerdict
from router.config_io import load_current_config
from router.core.clustering import KMeansClusterAssigner
from router.core.embeddings import PromptEmbedder, SentenceTransformerProvider
from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError
from router.evaluation.cache import ResponseCache
from router.evaluation.evaluator import (
    CacheGapError,
    EvaluationResult,
    RouterEvaluator,
)
from router.models.llm_profile import LLMProfile
from router.models.llm_registry import LLMRegistry
from router.uniroute import UniRouteRouter


logger = logging.getLogger("harness.critics.router_critic")


@dataclass
class RouterQualityGate:
    """Locked P15.3 thresholds. Override via critic params if needed."""

    min_auroc_improvement: float = 0.0
    min_win_rate: float = 0.5
    max_error_rate_increase: float = 0.05


@register_critic
class RouterCritic(Critic):
    """Quality-gate critic for ``kind="router_config"`` proposals.

    Params (all optional):
      cache_path: str — defaults to ``evals/_response_cache/cache.jsonl``
      eval_lambda_steps: int = 5
      min_auroc_improvement / min_win_rate / max_error_rate_increase
    """

    name = "router_quality_gate"
    stage = CriticStage.POST

    def verdict(self, ctx: CriticContext) -> CriticVerdict:
        gate = self._gate_from_params()

        # 1. Pull the candidate payload off the proposal.
        try:
            candidate_payload = self._extract_payload(ctx)
        except ValueError as e:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=str(e),
                severity="block",
            )

        # 2. Build the candidate router.
        try:
            candidate_router = self._build_router_from_payload(candidate_payload)
        except Exception as e:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=f"failed to build candidate router: {type(e).__name__}: {e}",
                severity="block",
            )

        # 3. Build the current router (cold-start tolerated).
        current_router: Optional[UniRouteRouter] = None
        try:
            current_assigner, current_registry, current_lambda = load_current_config()
            current_router = UniRouteRouter(
                embedder=candidate_router.embedder,
                cluster_assigner=current_assigner,
                registry=current_registry,
                cost_weight=current_lambda,
            )
        except (RouterConfigNotFoundError, RouterConfigInvalidError):
            logger.info("no current router_config — comparing against AUROC=0.5 baseline")

        # 4. Build a tiny eval dataset out of the proposer's traces metadata.
        #    The smoke + tests pass cache + dataset directly via params; in
        #    production the proposer's metadata.dataset_path is consulted.
        eval_dataset = self._params_dataset()
        cache = self._params_cache()
        if eval_dataset is None or cache is None:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=(
                    "router_critic needs params['dataset'] (PromptDataset) "
                    "and params['cache'] (ResponseCache) to run"
                ),
                severity="block",
            )

        # 5. Score candidate (always) + current (when present).
        eval_lambda_steps = int(self.params.get("eval_lambda_steps", 5))
        try:
            cand_eval = RouterEvaluator(
                router=candidate_router,
                cache=cache,
                profiles=list(candidate_router.registry),
                lambda_steps=eval_lambda_steps,
            ).evaluate(eval_dataset, dataset_name="candidate")
        except CacheGapError as e:
            return CriticVerdict(
                critic=self.name,
                approved=False,
                reason=f"cache gap: {e}",
                severity="block",
            )
        cur_eval: Optional[EvaluationResult] = None
        if current_router is not None:
            try:
                cur_eval = RouterEvaluator(
                    router=current_router,
                    cache=cache,
                    profiles=list(current_router.registry),
                    lambda_steps=eval_lambda_steps,
                ).evaluate(eval_dataset, dataset_name="current")
            except CacheGapError:
                # If the current router's models aren't in the cache anymore
                # (model upgrade, etc.) treat as cold-start.
                cur_eval = None

        # 6. Apply gate.
        delta_auroc = (
            cand_eval.metrics.auroc - cur_eval.metrics.auroc
            if cur_eval is not None
            else cand_eval.metrics.auroc - 0.5
        )
        delta_avg_err = (
            (1.0 - cand_eval.metrics.win_rate) - (1.0 - cur_eval.metrics.win_rate)
            if cur_eval is not None
            else (1.0 - cand_eval.metrics.win_rate) - 0.5
        )

        passed_auroc = delta_auroc >= gate.min_auroc_improvement
        passed_win_rate = cand_eval.metrics.win_rate >= gate.min_win_rate
        passed_err_bound = delta_avg_err <= gate.max_error_rate_increase

        all_passed = passed_auroc and passed_win_rate and passed_err_bound
        reason = (
            f"delta_auroc={delta_auroc:+.4f} (≥{gate.min_auroc_improvement:+.4f}: {passed_auroc}); "
            f"win_rate={cand_eval.metrics.win_rate:.3f} (≥{gate.min_win_rate}: {passed_win_rate}); "
            f"delta_avg_err={delta_avg_err:+.4f} (≤{gate.max_error_rate_increase:+.4f}: {passed_err_bound})"
        )
        return CriticVerdict(
            critic=self.name,
            approved=all_passed,
            reason=reason,
            severity="block" if not all_passed else "info",
        )

    # ------------------------------------------------------------------

    def _gate_from_params(self) -> RouterQualityGate:
        return RouterQualityGate(
            min_auroc_improvement=float(
                self.params.get("min_auroc_improvement", 0.0)
            ),
            min_win_rate=float(self.params.get("min_win_rate", 0.5)),
            max_error_rate_increase=float(
                self.params.get("max_error_rate_increase", 0.05)
            ),
        )

    def _extract_payload(self, ctx: CriticContext) -> dict:
        if not ctx.proposal.mutations:
            raise ValueError("proposal has no mutations")
        first = ctx.proposal.mutations[0]
        payload = first.value
        if not isinstance(payload, dict):
            raise ValueError(
                f"router_critic expected dict payload on Mutation.value, got "
                f"{type(payload).__name__}"
            )
        if "model_psi" not in payload or "k" not in payload:
            raise ValueError("payload missing required keys: 'k' / 'model_psi'")
        return payload

    def _build_router_from_payload(self, payload: dict) -> UniRouteRouter:
        # Centroids: prefer params['centroids'] (tests inject), else fail loud
        # since the executor is the only path that writes the .npz sidecar
        # and the critic runs *before* the executor.
        centroids = self.params.get("centroids")
        if centroids is None and "centroids" in payload and payload["centroids"]:
            centroids = payload["centroids"]
        if centroids is None:
            raise ValueError(
                "candidate centroids not found — pass params['centroids'] from the "
                "proposer's KMeansTrainResult.assigner.centroids until the executor "
                "lands the .npz sidecar (P15.3.7 T5)"
            )
        assigner = KMeansClusterAssigner(centroids=np.asarray(centroids))

        registry = LLMRegistry()
        for model_id, blob in payload["model_psi"].items():
            psi = np.asarray(blob["psi_vector"], dtype=float)
            counts = np.asarray(
                blob.get("cluster_sample_counts", [1] * len(psi)), dtype=int
            )
            registry.register(
                LLMProfile(
                    model_id=model_id,
                    psi_vector=psi,
                    cost_per_1k_tokens=float(blob.get("cost_per_1k_tokens", 0.0)),
                    num_validation_samples=int(counts.sum()) if counts.size else 0,
                    cluster_sample_counts=counts,
                    metadata=dict(blob.get("metadata", {})),
                )
            )

        embedder = self._params_embedder()
        return UniRouteRouter(
            embedder=embedder,
            cluster_assigner=assigner,
            registry=registry,
            cost_weight=float(payload.get("cost_weight", 0.0)),
        )

    def _params_embedder(self) -> PromptEmbedder:
        emb = self.params.get("embedder")
        if emb is not None:
            return emb
        return PromptEmbedder(SentenceTransformerProvider())

    def _params_cache(self) -> Optional[Any]:
        cache = self.params.get("cache")
        if cache is not None:
            return cache
        path = self.params.get("cache_path")
        if path is None:
            return None
        return ResponseCache(path=path)

    def _params_dataset(self) -> Optional[Any]:
        return self.params.get("dataset")
