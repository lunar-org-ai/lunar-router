"""Deployment API routes — /v1/deployments/* endpoints.

Matches the contract expected by ui/src/features/production/api/deploymentService.ts
and ui/src/services/DeploymentService.ts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from . import storage
from .manager import (
    deploy_model,
    pause_deployment,
    resume_deployment,
    stop_deployment,
)
from .schemas import (
    CreateDeploymentRequest,
    DeploymentListResponse,
    DeploymentMetricsLatest,
    DeploymentMetricsResponse,
    DeploymentInferenceStats,
    DeploymentResponse,
)

logger = logging.getLogger(__name__)

deployment_router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_deployment(dep: dict) -> dict:
    """Format a storage dict into the DeploymentResponse shape."""
    return DeploymentResponse(
        deployment_id=dep.get("id", ""),
        endpoint_name=dep.get("endpoint_name", dep.get("model_id", "")),
        status=dep.get("status", ""),
        model_id=dep.get("model_id", ""),
        instance_type=dep.get("instance_type", "local-gpu"),
        updated_at=dep.get("updated_at", ""),
        tenant_id=dep.get("tenant_id", "local"),
        scaling=dep.get("scaling", {}),
        error_message=dep.get("error_message", ""),
        error_code=dep.get("error_code", ""),
        endpoint_url=dep.get("endpoint_url", ""),
    ).model_dump()


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@deployment_router.post("", response_model=DeploymentResponse)
async def create_deployment(body: CreateDeploymentRequest):
    """Create a new vLLM deployment."""
    model_path = body.model_path or body.model_id
    config = body.config.model_dump()
    config["instance_type"] = body.instance_type

    if body.scaling:
        config["scaling"] = body.scaling.model_dump()

    result = await deploy_model(
        model_id=body.model_id,
        model_path=model_path,
        config=config,
    )

    dep = storage.get_deployment(result["deployment_id"])
    if dep:
        return JSONResponse(content=_format_deployment(dep))
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


@deployment_router.get("", response_model=DeploymentListResponse)
async def list_deployments(statuses: str = Query("")):
    """List deployments, optionally filtered by comma-separated statuses."""
    status_list = (
        [s.strip() for s in statuses.split(",") if s.strip()]
        if statuses
        else None
    )
    deps = storage.list_deployments(statuses=status_list)
    return {"deployments": [_format_deployment(d) for d in deps]}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@deployment_router.get("/{deployment_id}/status", response_model=DeploymentResponse)
async def get_deployment_status(deployment_id: str):
    """Get deployment status — polled every 5 s by the UI."""
    dep = storage.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(404, f"Deployment {deployment_id} not found")
    return JSONResponse(content=_format_deployment(dep))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@deployment_router.get(
    "/{deployment_id}/metrics",
    response_model=DeploymentMetricsResponse,
)
async def get_deployment_metrics(
    deployment_id: str,
    type: str = Query(""),
    minutes: int = Query(60),
    period: int = Query(60),
):
    """Get deployment metrics aggregated from ClickHouse inference_metrics."""
    from . import inference_metrics as metrics_store

    dep = storage.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(404, f"Deployment {deployment_id} not found")

    stats = metrics_store.get_aggregate_stats(deployment_id, minutes=minutes)
    series = metrics_store.get_time_series(deployment_id, minutes=minutes, period_seconds=period)
    latest = metrics_store.get_latest(deployment_id)

    now = datetime.now(timezone.utc).isoformat()

    return DeploymentMetricsResponse(
        deployment_id=deployment_id,
        latest=DeploymentMetricsLatest(
            timestamp=latest.get("timestamp") or now,
            model_latency_ms=float(latest.get("model_latency_ms", 0)),
            invocations=stats.get("total_inferences", 0),
        ),
        inference_stats=DeploymentInferenceStats(
            total_inferences=stats.get("total_inferences", 0),
            successful=stats.get("successful", 0),
            failed=stats.get("failed", 0),
            success_rate=stats.get("success_rate", 100.0),
            avg_latency_ms=stats.get("avg_latency_ms", 0),
            total_tokens=stats.get("total_tokens", 0),
        ),
        time_series=series,
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@deployment_router.delete("/{deployment_id}", status_code=204)
async def delete_deployment(deployment_id: str):
    """Stop the vLLM process and remove the deployment."""
    dep = storage.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(404, f"Deployment {deployment_id} not found")

    await stop_deployment(deployment_id)
    storage.update_deployment(deployment_id, status="deleting")
    storage.delete_deployment(deployment_id)


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------


@deployment_router.patch("/{deployment_id}/pause", response_model=DeploymentResponse)
async def pause_deployment_route(deployment_id: str):
    """Pause a running deployment (SIGSTOP)."""
    dep = storage.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(404, f"Deployment {deployment_id} not found")
    if dep.get("status") not in ("in_service", "active"):
        raise HTTPException(400, "Can only pause an active deployment")

    await pause_deployment(deployment_id)
    dep = storage.get_deployment(deployment_id)
    return JSONResponse(content=_format_deployment(dep))


@deployment_router.patch("/{deployment_id}/resume", response_model=DeploymentResponse)
async def resume_deployment_route(deployment_id: str):
    """Resume a paused deployment (SIGCONT)."""
    dep = storage.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(404, f"Deployment {deployment_id} not found")
    if dep.get("status") != "paused":
        raise HTTPException(400, "Can only resume a paused deployment")

    await resume_deployment(deployment_id)
    dep = storage.get_deployment(deployment_id)
    return JSONResponse(content=_format_deployment(dep))
