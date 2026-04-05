"""
API Schemas: Pydantic models for API request/response validation.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Request to route a prompt to an LLM."""

    prompt: str = Field(..., description="The prompt text to route")
    available_models: Optional[list[str]] = Field(
        None,
        description="List of model IDs to consider. If None, considers all."
    )
    cost_weight: Optional[float] = Field(
        None,
        description="Override cost weight (λ) for this request",
        ge=0.0,
    )
    execute: bool = Field(
        False,
        description="If True, also execute the prompt on the selected model"
    )
    max_tokens: int = Field(
        256,
        description="Max tokens for generation (if execute=True)",
        gt=0,
    )
    temperature: float = Field(
        0.0,
        description="Temperature for generation (if execute=True)",
        ge=0.0,
        le=2.0,
    )


class RouteResponse(BaseModel):
    """Response from routing a prompt."""

    selected_model: str = Field(..., description="ID of the selected model")
    expected_error: float = Field(
        ...,
        description="Predicted error rate γ(x, h)",
        ge=0.0,
        le=1.0,
    )
    cost_adjusted_score: float = Field(
        ...,
        description="Score including cost penalty"
    )
    cluster_id: int = Field(..., description="Dominant cluster for the prompt")
    all_scores: Optional[dict[str, float]] = Field(
        None,
        description="Scores for all considered models"
    )
    response_text: Optional[str] = Field(
        None,
        description="Generated response (if execute=True)"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Explanation of the routing decision"
    )


class BatchRouteRequest(BaseModel):
    """Request to route multiple prompts."""

    prompts: list[str] = Field(..., description="List of prompts to route")
    available_models: Optional[list[str]] = Field(None)
    cost_weight: Optional[float] = Field(None, ge=0.0)


class BatchRouteResponse(BaseModel):
    """Response from batch routing."""

    decisions: list[RouteResponse] = Field(
        ...,
        description="Routing decisions for each prompt"
    )
    distribution: dict[str, float] = Field(
        ...,
        description="Fraction of prompts routed to each model"
    )
    avg_expected_error: float = Field(..., description="Average expected error")


class ModelInfo(BaseModel):
    """Information about a registered model."""

    model_id: str = Field(..., description="Model identifier")
    cost_per_1k_tokens: float = Field(..., description="Cost per 1k tokens")
    num_clusters: int = Field(..., description="Number of clusters")
    overall_accuracy: float = Field(
        ...,
        description="Overall accuracy across all clusters"
    )
    strongest_clusters: list[tuple[int, float]] = Field(
        ...,
        description="Top clusters where model performs best"
    )


class ModelListResponse(BaseModel):
    """Response listing all registered models."""

    models: list[ModelInfo] = Field(..., description="List of model info")
    default_model: Optional[str] = Field(
        None,
        description="ID of the default model"
    )


class RegisterModelRequest(BaseModel):
    """Request to register a new model profile."""

    model_id: str = Field(..., description="Model identifier")
    psi_vector: list[float] = Field(
        ...,
        description="Error rates per cluster (Ψ vector)"
    )
    cost_per_1k_tokens: float = Field(
        ...,
        description="Cost per 1k tokens",
        ge=0.0,
    )
    num_validation_samples: int = Field(
        ...,
        description="Number of samples used to compute Ψ",
        gt=0,
    )
    cluster_sample_counts: list[float] = Field(
        ...,
        description="Samples per cluster used in computation"
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )


class StatsResponse(BaseModel):
    """Router statistics response."""

    total_requests: int = Field(..., description="Total routing requests")
    model_selections: dict[str, int] = Field(
        ...,
        description="Count of selections per model"
    )
    cluster_distributions: dict[str, int] = Field(
        ...,
        description="Count of prompts per cluster"
    )
    avg_expected_error: float = Field(..., description="Average expected error")
    avg_cost_score: float = Field(..., description="Average cost-adjusted score")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    router_initialized: bool = Field(
        ...,
        description="Whether the router is initialized"
    )
    num_models: int = Field(..., description="Number of registered models")
    num_clusters: int = Field(..., description="Number of clusters")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional details")


# ── Router Intelligence ──────────────────────────────────────────────────────


class KpiValue(BaseModel):
    value: Any
    delta_pct: Optional[float] = None


class EfficiencyResponse(BaseModel):
    """Router efficiency metrics."""

    kpis: dict[str, KpiValue] = Field(default_factory=dict)
    model_distribution: list[dict[str, Any]] = Field(default_factory=list)
    cost_savings_trend: list[dict[str, Any]] = Field(default_factory=list)
    model_breakdown: list[dict[str, Any]] = Field(default_factory=list)


class ModelPerformanceResponse(BaseModel):
    """Model performance and comparison data."""

    kpis: dict[str, Any] = Field(default_factory=dict)
    cluster_accuracy: list[dict[str, Any]] = Field(default_factory=list)
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    teacher_student: Optional[dict[str, Any]] = None


class TrainingActivityResponse(BaseModel):
    """Auto-training activity and advisor decisions."""

    kpis: dict[str, Any] = Field(default_factory=dict)
    training_history: list[dict[str, Any]] = Field(default_factory=list)
    signal_trends: list[dict[str, Any]] = Field(default_factory=list)
    advisor_decisions: list[dict[str, Any]] = Field(default_factory=list)
    training_cycles: list[dict[str, Any]] = Field(default_factory=list)
