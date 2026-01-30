"""
Deployment routes for EKS/Kubernetes deployments.

Provides endpoints to:
- List deployments for a tenant
- Get deployment details
- Get deployment costs (inference-time based pricing)
- Get cost overview across all deployments
"""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any, Optional

from ..database.DeploymentHandler import DeploymentHandler
from ..database.deployment_models import (
    DeploymentModel,
    DeploymentCostSummary,
    DeploymentsCostOverview,
    INSTANCE_COSTS_USD_PER_HOUR,
)


router = APIRouter(prefix="/v1/deployments", tags=["Deployments"])


@router.get("", response_model=List[DeploymentModel])
async def list_deployments(
    request: Request,
    status: Optional[str] = None
):
    """
    List all deployments for the current tenant.

    Query params:
        status: Filter by status (in_service, paused, resuming)

    Returns list of deployments with their current state and metrics.
    """
    tenant_id = getattr(request.state, "tenant_id", "default")

    deployments = await DeploymentHandler.get_deployments_by_tenant(
        tenant_id=tenant_id,
        status_filter=status
    )

    return deployments


@router.get("/costs", response_model=DeploymentsCostOverview)
async def get_costs_overview(request: Request):
    """
    Get cost overview for all deployments.

    Returns aggregated metrics including:
    - Total uptime vs inference time
    - Cost comparison (uptime-based vs inference-based)
    - Savings from inference-time pricing
    - Per-deployment breakdown

    Example response:
    ```json
    {
        "tenant_id": "tenant-123",
        "period": "all_time",
        "total_deployments": 3,
        "active_deployments": 2,
        "total_uptime_hours": 48.5,
        "total_inference_hours": 2.3,
        "overall_utilization_percent": 4.74,
        "total_uptime_cost_usd": 48.77,
        "total_inference_cost_usd": 2.31,
        "total_savings_usd": 46.46,
        "savings_percent": 95.26,
        "deployments": [...]
    }
    ```
    """
    tenant_id = getattr(request.state, "tenant_id", "default")

    overview = await DeploymentHandler.get_tenant_costs_overview(tenant_id)

    return overview


@router.get("/instance-types", response_model=Dict[str, float])
async def get_instance_types():
    """
    Get supported instance types and their costs per hour.

    Returns a dictionary mapping instance type to USD per hour.
    Useful for understanding pricing before creating deployments.

    Example response:
    ```json
    {
        "g5.xlarge": 1.006,
        "g5.2xlarge": 1.212,
        "g5.4xlarge": 1.624,
        "p4d.24xlarge": 32.77,
        ...
    }
    ```
    """
    return DeploymentHandler.get_supported_instance_types()


@router.get("/{deployment_id}", response_model=DeploymentModel)
async def get_deployment(deployment_id: str, request: Request):
    """
    Get details for a specific deployment.

    Returns deployment configuration, status, and accumulated metrics.
    """
    tenant_id = getattr(request.state, "tenant_id", "default")

    deployment = await DeploymentHandler.get_deployment(deployment_id)

    if not deployment:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{deployment_id}' not found"
        )

    # Verify tenant ownership
    if deployment.tenant_id != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deployment"
        )

    return deployment


@router.get("/{deployment_id}/costs", response_model=DeploymentCostSummary)
async def get_deployment_costs(deployment_id: str, request: Request):
    """
    Get detailed cost breakdown for a deployment.

    Returns:
    - Current and total uptime
    - Total inference time
    - Utilization percentage (inference_time / uptime)
    - Cost comparison (what you would pay by uptime vs inference time)
    - Savings amount

    Example response:
    ```json
    {
        "deployment_id": "deploy-llama-8b",
        "model_id": "llama-3.1-8b",
        "instance_type": "g5.xlarge",
        "status": "in_service",

        "current_uptime_seconds": 7200,
        "total_uptime_seconds": 86400,
        "total_inference_seconds": 3600,
        "utilization_percent": 4.17,

        "cost_per_hour": 1.006,
        "uptime_cost_usd": 24.144,
        "inference_cost_usd": 1.006,
        "savings_usd": 23.138,

        "started_at": "2025-01-06T10:00:00Z"
    }
    ```
    """
    tenant_id = getattr(request.state, "tenant_id", "default")

    # First verify access
    deployment = await DeploymentHandler.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{deployment_id}' not found"
        )

    if deployment.tenant_id != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this deployment"
        )

    # Get cost summary
    summary = await DeploymentHandler.get_deployment_cost_summary(deployment_id)

    if not summary:
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate cost summary"
        )

    return summary
