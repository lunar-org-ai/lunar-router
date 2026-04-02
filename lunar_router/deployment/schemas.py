"""Pydantic schemas for deployment API — matches UI DeploymentService contract."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ScalingConfig(BaseModel):
    """Auto-scaling configuration."""

    min: int = Field(1, description="Minimum instances", ge=1)
    max: int = Field(1, description="Maximum instances", ge=1)


class VLLMConfig(BaseModel):
    """vLLM engine configuration."""

    vllm_args: str = Field(
        "",
        description="Space-separated vLLM CLI arguments "
        "(e.g. '--max-model-len 4096 --dtype bfloat16')",
    )


class CreateDeploymentRequest(BaseModel):
    """Request to create a new vLLM deployment.

    Matches the shape sent by ui/src/features/production/api/deploymentService.ts
    """

    model_id: str = Field(..., description="Model identifier or HuggingFace path")
    model_path: Optional[str] = Field(
        None,
        description="Local filesystem path to model weights. Defaults to model_id.",
    )
    instance_type: str = Field(
        "local-gpu",
        description="Instance type (local-gpu, gpu-xs, gpu-m, etc.)",
    )
    scaling: Optional[ScalingConfig] = None
    config: VLLMConfig = Field(default_factory=VLLMConfig)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DeploymentResponse(BaseModel):
    """Single deployment status — matches UI DeploymentResponse type."""

    deployment_id: str = Field(..., description="Unique deployment ID")
    endpoint_name: str = Field("", description="Human-readable endpoint name")
    status: str = Field(
        ...,
        description="Deployment status: creating | starting | in_service | "
        "paused | failed | stopped | deleting",
    )
    model_id: str = Field("", description="Model identifier")
    instance_type: str = Field("local-gpu", description="Instance type")
    updated_at: str = Field("", description="Last update ISO timestamp")
    tenant_id: str = Field("local", description="Tenant ID")
    scaling: Optional[dict[str, Any]] = Field(default_factory=dict)
    error_message: str = Field("", description="Error details if failed")
    error_code: str = Field("", description="Structured error code")
    endpoint_url: str = Field("", description="Inference URL when in_service")


class DeploymentListResponse(BaseModel):
    """Response for listing deployments."""

    deployments: list[DeploymentResponse]


class DeploymentMetricsLatest(BaseModel):
    """Latest metrics snapshot."""

    cpu_utilization: float = 0
    memory_utilization: float = 0
    gpu_utilization: float = 0
    gpu_memory_utilization: float = 0
    model_latency_ms: float = 0
    invocations: int = 0
    timestamp: str = ""


class DeploymentInferenceStats(BaseModel):
    """Aggregated inference statistics."""

    total_inferences: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 100
    avg_latency_ms: float = 0
    total_tokens: int = 0
    total_cost_usd: float = 0


class TimeSeriesPoint(BaseModel):
    """Single point in a time-series."""

    timestamp: str
    value: float


class DeploymentMetricsResponse(BaseModel):
    """Deployment metrics — matches UI deploymentMetricsService.ts NEW format."""

    deployment_id: str
    latest: DeploymentMetricsLatest = Field(default_factory=DeploymentMetricsLatest)
    inference_stats: DeploymentInferenceStats = Field(
        default_factory=DeploymentInferenceStats,
    )
    time_series: dict[str, list[TimeSeriesPoint]] = Field(default_factory=dict)
