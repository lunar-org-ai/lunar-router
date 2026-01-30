"""
Semantic routing module for ECS Router.

Integrates UniRoute semantic routing with the existing ECS adapter infrastructure.
"""

from .schemas import (
    SemanticRouterRequest,
    SemanticRouterResponse,
    RoutingDecisionInfo,
)
from .router_service import SemanticRouterService

__all__ = [
    "SemanticRouterRequest",
    "SemanticRouterResponse",
    "RoutingDecisionInfo",
    "SemanticRouterService",
]
