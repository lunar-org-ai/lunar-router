"""
router-core: Shared LLM routing logic for Lunar Router.

This package provides the core routing functionality that can be used by:
- SaaS Router: Multi-tenant router with DynamoDB/Secrets Manager
- Data Plane Router: Single-tenant router for enterprise deployments

Usage:
    from router_core import HealthFirstPlanner, ProviderAdapter
    from router_core.providers import StatsProvider, PricingProvider
    from router_core.models import ChatCompletionRequest, ChatCompletionResponse
"""

__version__ = "0.1.0"

# Core classes
from .planner import HealthFirstPlanner, RoundRobinPlanner
from .adapters.base import ProviderAdapter, _extract_prompt
from .error_classifier import ErrorCategory, classify_error, is_retryable, should_skip_provider

# Provider interfaces
from .providers.base import (
    StatsProvider,
    PricingProvider,
    SecretsProvider,
    TelemetryProvider,
    StatsSummary,
    PricingInfo,
    # Default implementations
    InMemoryStatsProvider,
    StaticPricingProvider,
    EnvSecretsProvider,
    NoOpTelemetryProvider,
)

# Cache
from .cache.ranking_cache import RankingCache, get_ranking_cache
from .cache.pricing_cache import PricingCache, get_pricing_cache

# Models/Schemas
from .models.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    Usage,
    ChatCompletionChunk,
    StreamChoice,
    DeltaContent,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ModelInfo,
    ModelsListResponse,
    ErrorResponse,
    ErrorDetail,
    InferRequest,
    InferResponse,
    HealthResponse,
)

__all__ = [
    # Version
    "__version__",
    # Planners
    "HealthFirstPlanner",
    "RoundRobinPlanner",
    # Adapters
    "ProviderAdapter",
    "_extract_prompt",
    # Error handling
    "ErrorCategory",
    "classify_error",
    "is_retryable",
    "should_skip_provider",
    # Provider interfaces
    "StatsProvider",
    "PricingProvider",
    "SecretsProvider",
    "TelemetryProvider",
    "StatsSummary",
    "PricingInfo",
    # Default implementations
    "InMemoryStatsProvider",
    "StaticPricingProvider",
    "EnvSecretsProvider",
    "NoOpTelemetryProvider",
    # Cache
    "RankingCache",
    "get_ranking_cache",
    "PricingCache",
    "get_pricing_cache",
    # Models
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatChoice",
    "Usage",
    "ChatCompletionChunk",
    "StreamChoice",
    "DeltaContent",
    "CompletionRequest",
    "CompletionResponse",
    "CompletionChoice",
    "ModelInfo",
    "ModelsListResponse",
    "ErrorResponse",
    "ErrorDetail",
    "InferRequest",
    "InferResponse",
    "HealthResponse",
]
