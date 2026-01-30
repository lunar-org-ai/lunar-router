"""
Semantic Router API endpoint - Local Version.
Uses local JSON-based handlers.
"""

from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union

from ..semantic.schemas import SemanticRouterRequest, SemanticRouterResponse
from ..semantic.router_service_local import get_semantic_service

router = APIRouter(prefix="/v1", tags=["Semantic Router"])


@router.post(
    "/router",
    response_model=SemanticRouterResponse,
    summary="Semantic LLM Routing",
    description="""
Route prompts to the optimal LLM using semantic analysis.

## Routing Algorithm

Uses the UniRoute algorithm to select the best model:

    h* = argmin_h [gamma(x, h) + lambda * c(h)]

Where:
- gamma(x, h) = predicted error rate for model h on prompt x
- lambda = cost_weight parameter
- c(h) = cost per 1k tokens for model h

## Modes

1. **Automatic Routing**: Don't specify `models` - routes among all available models
2. **Restricted Routing**: Specify a `models` list (min 2) - routes only among those models

## Execution

- Set `execute=true` (default) to execute the prompt on the selected model
- Set `execute=false` to only get the routing decision without execution
- Set `stream=true` for streaming responses (when execute=true)
""",
)
async def semantic_route(
    req: SemanticRouterRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> Union[SemanticRouterResponse, StreamingResponse]:
    """
    Semantic routing endpoint.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    service = get_semantic_service()

    response = await service.route_and_execute(
        request=req,
        tenant_id=tenant_id,
        background_tasks=background_tasks,
    )

    return response


@router.get(
    "/router/profiles",
    summary="List Available Routing Profiles",
    description="Returns the list of models that have semantic routing profiles available.",
)
async def list_routing_profiles():
    """
    List all models that have semantic routing profiles.
    """
    service = get_semantic_service()
    await service.initialize()

    profiles = service.get_available_profiles()

    return {
        "profiles": profiles,
        "count": len(profiles),
    }
