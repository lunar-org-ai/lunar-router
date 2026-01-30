"""
Semantic Router API endpoint.

Provides the /v1/router endpoint for semantic-based LLM routing.
"""

from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union

from ..semantic.schemas import SemanticRouterRequest, SemanticRouterResponse
from ..semantic.router_service import get_semantic_service

router = APIRouter(prefix="/v1", tags=["Semantic Router"])


@router.post(
    "/router",
    response_model=SemanticRouterResponse,
    summary="Semantic LLM Routing",
    description="""
Route prompts to the optimal LLM using semantic analysis.

## Routing Algorithm

Uses the UniRoute algorithm to select the best model:

    h* = argmin_h [γ(x, h) + λ·c(h)]

Where:
- γ(x, h) = predicted error rate for model h on prompt x
- λ = cost_weight parameter
- c(h) = cost per 1k tokens for model h

## Modes

1. **Automatic Routing**: Don't specify `models` - routes among all available models for the tenant
2. **Restricted Routing**: Specify a `models` list (min 2) - routes only among those models

## Execution

- Set `execute=true` (default) to execute the prompt on the selected model
- Set `execute=false` to only get the routing decision without execution
- Set `stream=true` for streaming responses (when execute=true)
""",
    responses={
        200: {"description": "Successful routing and optional execution"},
        400: {
            "description": "Invalid request - insufficient models or missing profiles",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_profiles": {
                            "summary": "Models without profiles",
                            "value": {
                                "detail": {
                                    "error": "Models without profiles",
                                    "code": "MISSING_PROFILES",
                                    "missing_profiles": ["unknown-model"],
                                    "available_profiles": ["gpt-4o", "gpt-4o-mini", "mistral-small-latest"]
                                }
                            }
                        },
                        "insufficient_models": {
                            "summary": "Not enough models",
                            "value": {
                                "detail": {
                                    "error": "Semantic routing requires at least 2 models",
                                    "code": "INSUFFICIENT_MODELS",
                                    "provided": 1
                                }
                            }
                        }
                    }
                }
            }
        },
        401: {"description": "Authentication required"},
        502: {"description": "Model execution failed"},
    }
)
async def semantic_route(
    req: SemanticRouterRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> Union[SemanticRouterResponse, StreamingResponse]:
    """
    Semantic routing endpoint.

    Routes prompts to the best LLM based on semantic analysis of the prompt content.
    Optionally executes the prompt on the selected model.

    The routing uses pre-trained profiles that map prompt characteristics to expected
    model performance, selecting the model with the lowest expected error (optionally
    weighted by cost).

    Args:
        req: The semantic router request containing messages and options.
        request: FastAPI request object (for tenant_id extraction).
        background_tasks: FastAPI background tasks for async operations.

    Returns:
        SemanticRouterResponse with routing decision and optional completion.
        Or StreamingResponse if stream=True and execute=True.

    Raises:
        HTTPException: For validation errors or execution failures.
    """
    # Extract tenant_id from auth middleware
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    # Get service and execute
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
    responses={
        200: {
            "description": "List of model IDs with available profiles",
            "content": {
                "application/json": {
                    "example": {
                        "profiles": [
                            "gpt-4o",
                            "gpt-4o-mini",
                            "gpt-4-turbo",
                            "gpt-3.5-turbo",
                            "mistral-large-latest",
                            "mistral-small-latest"
                        ],
                        "count": 6
                    }
                }
            }
        }
    }
)
async def list_routing_profiles():
    """
    List all models that have semantic routing profiles.

    This endpoint can be used to discover which models support semantic routing
    before making a routing request.

    Returns:
        Dictionary with list of profile IDs and count.
    """
    service = get_semantic_service()
    await service.initialize()

    profiles = service.get_available_profiles()

    return {
        "profiles": profiles,
        "count": len(profiles),
    }
