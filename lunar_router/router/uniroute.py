"""
UniRoute Router: The main routing logic.

Implements the core routing decision:
    h* = argmin_h [γ(x, h) + λ·c(h)]

where γ(x, h) = Φ(x)ᵀ · Ψ(h)
"""

from dataclasses import dataclass, field
from typing import Optional
import logging
import numpy as np

from ..core.embeddings import PromptEmbedder
from ..core.clustering import ClusterAssigner, ClusterResult
from ..models.llm_profile import LLMProfile
from ..models.llm_registry import LLMRegistry
from ..models.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """
    Result of a routing decision.

    Attributes:
        selected_model: ID of the selected model.
        expected_error: Predicted error rate γ(x, h).
        cost_adjusted_score: Score including cost penalty.
        all_scores: Scores for all considered models.
        cluster_id: Dominant cluster for the prompt.
        cluster_probabilities: Full probability distribution.
        reasoning: Optional explanation of the decision.
    """
    selected_model: str
    expected_error: float
    cost_adjusted_score: float
    all_scores: dict[str, float]
    cluster_id: int
    cluster_probabilities: np.ndarray
    reasoning: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "selected_model": self.selected_model,
            "expected_error": self.expected_error,
            "cost_adjusted_score": self.cost_adjusted_score,
            "all_scores": self.all_scores,
            "cluster_id": self.cluster_id,
            "cluster_probabilities": self.cluster_probabilities.tolist(),
            "reasoning": self.reasoning,
        }


@dataclass
class RoutingStats:
    """Statistics about routing decisions."""
    total_requests: int = 0
    model_selections: dict[str, int] = field(default_factory=dict)
    cluster_distributions: dict[int, int] = field(default_factory=dict)
    avg_expected_error: float = 0.0
    avg_cost_score: float = 0.0

    def update(self, decision: RoutingDecision) -> None:
        """Update stats with a new decision."""
        self.total_requests += 1

        # Model selections
        model = decision.selected_model
        self.model_selections[model] = self.model_selections.get(model, 0) + 1

        # Cluster distribution
        cluster = decision.cluster_id
        self.cluster_distributions[cluster] = self.cluster_distributions.get(cluster, 0) + 1

        # Running averages
        n = self.total_requests
        self.avg_expected_error = (
            (self.avg_expected_error * (n - 1) + decision.expected_error) / n
        )
        self.avg_cost_score = (
            (self.avg_cost_score * (n - 1) + decision.cost_adjusted_score) / n
        )


class UniRouteRouter:
    """
    Main UniRoute router.

    Routes prompts to the best LLM based on predicted error
    and cost trade-off.

    Implements:
        h* = argmin_h [γ(x, h) + λ·c(h)]

    where:
        - γ(x, h) = Φ(x)ᵀ · Ψ(h) is the predicted error
        - λ is the cost weight (cost_weight)
        - c(h) is the cost of model h

    Attributes:
        embedder: Prompt embedder for φ(x).
        cluster_assigner: Cluster assigner for Φ(x).
        registry: Registry of available LLMs with profiles.
        cost_weight: Weight λ for cost penalty.
        use_soft_assignment: Use soft cluster probabilities.
    """

    def __init__(
        self,
        embedder: PromptEmbedder,
        cluster_assigner: ClusterAssigner,
        registry: LLMRegistry,
        cost_weight: float = 0.0,
        use_soft_assignment: bool = True,
        allowed_models: Optional[list[str]] = None,
    ):
        """
        Initialize the router.

        Args:
            embedder: PromptEmbedder for generating embeddings.
            cluster_assigner: ClusterAssigner for cluster mapping.
            registry: LLMRegistry with model profiles.
            cost_weight: Weight λ for cost in the objective (0 = ignore cost).
            use_soft_assignment: If True, use soft cluster probabilities.
                                 If False, use one-hot (hard) assignment.
            allowed_models: Optional list of model IDs to consider for routing.
                           If None, all models in the registry are available.
                           Example: ["gpt-4o", "gpt-4o-mini", "mistral-small-latest"]
        """
        self.embedder = embedder
        self.cluster_assigner = cluster_assigner
        self.registry = registry
        self.cost_weight = cost_weight
        self.use_soft_assignment = use_soft_assignment
        self.allowed_models = allowed_models
        self._stats = RoutingStats()

        # Validate allowed_models if provided
        if allowed_models:
            available = {p.model_id for p in registry.get_all()}
            invalid = set(allowed_models) - available
            if invalid:
                raise ValueError(
                    f"Models not found in registry: {invalid}. "
                    f"Available: {available}"
                )

    @property
    def stats(self) -> RoutingStats:
        """Return routing statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = RoutingStats()

    def route(
        self,
        prompt: str,
        available_models: Optional[list[str]] = None,
        cost_weight_override: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Route a prompt to the best LLM.

        Args:
            prompt: The input prompt text.
            available_models: Optional list of model IDs to consider.
                             If None, uses allowed_models from constructor.
                             If that's also None, considers all registered models.
            cost_weight_override: Override the default cost_weight for this request.

        Returns:
            RoutingDecision with the selected model and metrics.

        Raises:
            ValueError: If no models are available.
        """
        # Step 1: Compute prompt representation Φ(x)
        embedding = self.embedder.embed(prompt)
        cluster_result = self.cluster_assigner.assign(embedding)

        # Step 2: Get Φ vector (soft or hard)
        if self.use_soft_assignment:
            phi = cluster_result.probabilities
        else:
            phi = cluster_result.to_one_hot()

        # Step 3: Get available model profiles
        # Use provided list, or fall back to constructor's allowed_models
        models_to_use = available_models if available_models is not None else self.allowed_models
        profiles = self.registry.get_available_models(models_to_use)

        if not profiles:
            raise ValueError("No models available for routing")

        # Step 4: Score each model
        lambda_ = cost_weight_override if cost_weight_override is not None else self.cost_weight

        scores: dict[str, float] = {}
        best_model: Optional[str] = None
        best_score = float("inf")
        best_error = float("inf")

        for profile in profiles:
            # γ(x, h) = Φ(x)ᵀ · Ψ(h)
            expected_error = profile.get_expected_error(phi)

            # Final score with cost penalty
            score = expected_error + lambda_ * profile.cost_per_1k_tokens

            scores[profile.model_id] = score

            if score < best_score:
                best_score = score
                best_error = expected_error
                best_model = profile.model_id

        # Build reasoning
        reasoning = self._build_reasoning(
            cluster_result, phi, profiles, scores, best_model, lambda_
        )

        decision = RoutingDecision(
            selected_model=best_model,
            expected_error=best_error,
            cost_adjusted_score=best_score,
            all_scores=scores,
            cluster_id=cluster_result.cluster_id,
            cluster_probabilities=cluster_result.probabilities,
            reasoning=reasoning,
        )

        # Update stats
        self._stats.update(decision)

        logger.info(
            f"Routed to {best_model} "
            f"(cluster={cluster_result.cluster_id}, "
            f"error={best_error:.4f}, score={best_score:.4f})"
        )

        return decision

    def _build_reasoning(
        self,
        cluster_result: ClusterResult,
        phi: np.ndarray,
        profiles: list[LLMProfile],
        scores: dict[str, float],
        selected: str,
        lambda_: float,
    ) -> str:
        """Build human-readable reasoning for the decision."""
        lines = [
            f"Cluster: {cluster_result.cluster_id} (confidence: {phi[cluster_result.cluster_id]:.2%})",
            f"Cost weight (λ): {lambda_}",
            "Model scores:",
        ]

        for profile in sorted(profiles, key=lambda p: scores[p.model_id]):
            error = profile.get_expected_error(phi)
            cost_term = lambda_ * profile.cost_per_1k_tokens
            marker = " ← selected" if profile.model_id == selected else ""
            lines.append(
                f"  {profile.model_id}: "
                f"error={error:.4f} + cost={cost_term:.4f} = {scores[profile.model_id]:.4f}"
                f"{marker}"
            )

        return "\n".join(lines)

    def route_batch(
        self,
        prompts: list[str],
        available_models: Optional[list[str]] = None,
        cost_weight_override: Optional[float] = None,
    ) -> list[RoutingDecision]:
        """
        Route multiple prompts.

        Args:
            prompts: List of prompt texts.
            available_models: Optional list of model IDs to consider.
            cost_weight_override: Override the default cost_weight.

        Returns:
            List of RoutingDecision objects.
        """
        return [
            self.route(prompt, available_models, cost_weight_override)
            for prompt in prompts
        ]

    def route_and_execute(
        self,
        prompt: str,
        llm_clients: dict[str, LLMClient],
        available_models: Optional[list[str]] = None,
        cost_weight_override: Optional[float] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[RoutingDecision, LLMResponse]:
        """
        Route a prompt and execute it on the selected model.

        Args:
            prompt: The input prompt.
            llm_clients: Dictionary mapping model_id to LLMClient.
            available_models: Optional list of model IDs to consider.
            cost_weight_override: Override cost weight.
            max_tokens: Max tokens for generation.
            temperature: Temperature for generation.

        Returns:
            Tuple of (RoutingDecision, LLMResponse).

        Raises:
            ValueError: If selected model has no client.
        """
        decision = self.route(prompt, available_models, cost_weight_override)

        if decision.selected_model not in llm_clients:
            raise ValueError(
                f"No client for selected model '{decision.selected_model}'. "
                f"Available clients: {list(llm_clients.keys())}"
            )

        client = llm_clients[decision.selected_model]
        response = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        return decision, response

    def get_best_model_for_cluster(
        self,
        cluster_id: int,
        available_models: Optional[list[str]] = None,
        cost_weight_override: Optional[float] = None,
    ) -> Optional[str]:
        """
        Get the best model for a specific cluster.

        Args:
            cluster_id: The cluster index.
            available_models: Optional list of model IDs to consider.
            cost_weight_override: Override cost weight.

        Returns:
            Model ID of the best model, or None if no models available.
        """
        profiles = self.registry.get_available_models(available_models)

        if not profiles:
            return None

        lambda_ = cost_weight_override if cost_weight_override is not None else self.cost_weight

        best_model = None
        best_score = float("inf")

        for profile in profiles:
            error = profile.get_cluster_error(cluster_id)
            score = error + lambda_ * profile.cost_per_1k_tokens

            if score < best_score:
                best_score = score
                best_model = profile.model_id

        return best_model

    def analyze_routing_distribution(
        self,
        prompts: list[str],
        available_models: Optional[list[str]] = None,
    ) -> dict:
        """
        Analyze how prompts would be distributed across models.

        Args:
            prompts: List of prompts to analyze.
            available_models: Optional list of model IDs.

        Returns:
            Dictionary with distribution statistics.
        """
        model_counts: dict[str, int] = {}
        cluster_counts: dict[int, int] = {}
        total_error = 0.0

        for prompt in prompts:
            decision = self.route(prompt, available_models)

            model_counts[decision.selected_model] = (
                model_counts.get(decision.selected_model, 0) + 1
            )
            cluster_counts[decision.cluster_id] = (
                cluster_counts.get(decision.cluster_id, 0) + 1
            )
            total_error += decision.expected_error

        return {
            "num_prompts": len(prompts),
            "model_distribution": {
                k: v / len(prompts) for k, v in model_counts.items()
            },
            "model_counts": model_counts,
            "cluster_distribution": cluster_counts,
            "avg_expected_error": total_error / len(prompts) if prompts else 0,
        }
