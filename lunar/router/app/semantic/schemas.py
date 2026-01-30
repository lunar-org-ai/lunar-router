"""
Pydantic schemas for the semantic routing endpoint.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the chat format."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class SemanticRouterRequest(BaseModel):
    """Request schema for the /v1/router endpoint."""

    messages: List[ChatMessage] = Field(
        ...,
        description="Messages in OpenAI format [{role, content}]",
        min_length=1,
    )

    models: Optional[List[str]] = Field(
        None,
        description="List of models to route among (min 2). If None, uses all available models for the tenant.",
        min_length=2,
    )

    cost_weight: float = Field(
        0.0,
        description="Cost penalty weight (lambda). 0=ignore cost, higher=prefer cheaper models",
        ge=0.0,
    )

    execute: bool = Field(
        True,
        description="If True, execute the prompt on the selected model. If False, only return routing decision.",
    )

    stream: bool = Field(
        False,
        description="If True and execute=True, stream the response using SSE.",
    )

    max_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens for generation (when execute=True)",
        gt=0,
    )

    temperature: Optional[float] = Field(
        None,
        description="Temperature for generation (when execute=True)",
        ge=0.0,
        le=2.0,
    )


class RoutingDecisionInfo(BaseModel):
    """Information about the semantic routing decision."""

    selected_model: str = Field(..., description="The model selected by semantic routing")
    selected_provider: str = Field(..., description="The provider for the selected model")
    expected_error: float = Field(..., description="Expected error rate gamma(x,h)")
    cost_adjusted_score: float = Field(..., description="Final score including cost penalty")
    cluster_id: int = Field(..., description="The semantic cluster assigned to this prompt")
    all_scores: Optional[Dict[str, float]] = Field(
        None,
        description="Scores for all candidate models",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Human-readable explanation of the routing decision",
    )


class SemanticRouterResponse(BaseModel):
    """Response schema for the /v1/router endpoint."""

    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion", description="Object type")
    model: str = Field(..., description="The model that was used")

    choices: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Completion choices (empty if execute=False)",
    )

    usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Token usage information",
    )

    routing: RoutingDecisionInfo = Field(
        ...,
        description="Semantic routing decision metadata",
    )


class SemanticRouterError(BaseModel):
    """Error response for semantic routing."""

    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    available_profiles: Optional[List[str]] = Field(
        None,
        description="List of models with available profiles",
    )
    missing_profiles: Optional[List[str]] = Field(
        None,
        description="List of requested models without profiles",
    )
