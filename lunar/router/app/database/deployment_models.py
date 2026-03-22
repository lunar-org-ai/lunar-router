"""
Deployment models for EKS/Kubernetes deployments with inference-time pricing.

These models support calculating costs based on actual GPU inference time
rather than per-token pricing, which is more accurate for self-hosted models.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict
from decimal import Decimal


class DeploymentModel(BaseModel):
    """Model representing an EKS deployment."""

    deployment_id: str
    tenant_id: str
    model_id: str
    status: str  # in_service, paused, resuming, creating, failed
    service_url: Optional[str] = None

    # Kubernetes info
    k8s_deployment_name: Optional[str] = None
    k8s_namespace: Optional[str] = "default"

    # Instance and cost configuration
    instance_type: Optional[str] = None  # e.g., "g5.xlarge"
    cost_per_hour: Optional[float] = None  # USD per hour

    # Timestamps
    created_at: Optional[str] = None
    started_at: Optional[str] = None  # When deployment became in_service
    paused_at: Optional[str] = None   # When deployment was paused

    # Accumulated metrics
    total_uptime_seconds: float = 0.0
    total_inference_seconds: float = 0.0
    total_requests: int = 0


class DeploymentCostSummary(BaseModel):
    """Cost summary for a single deployment."""

    deployment_id: str
    model_id: str
    instance_type: Optional[str] = None
    status: str

    # Time metrics
    current_uptime_seconds: float = 0.0   # Time in current session
    total_uptime_seconds: float = 0.0     # All-time uptime
    total_inference_seconds: float = 0.0  # All-time inference time
    utilization_percent: float = 0.0      # inference_time / uptime_time * 100

    # Cost metrics
    cost_per_hour: float = 0.0
    uptime_cost_usd: float = 0.0          # Cost if charged by uptime
    inference_cost_usd: float = 0.0       # Cost if charged by inference time
    savings_usd: float = 0.0              # uptime_cost - inference_cost

    # Timestamps
    started_at: Optional[str] = None


class DeploymentsCostOverview(BaseModel):
    """Overview of costs across all deployments for a tenant."""

    tenant_id: str
    period: str  # e.g., "all_time", "last_24h", "last_7d"

    # Totals
    total_deployments: int = 0
    active_deployments: int = 0

    # Time totals
    total_uptime_hours: float = 0.0
    total_inference_hours: float = 0.0
    overall_utilization_percent: float = 0.0

    # Cost totals
    total_uptime_cost_usd: float = 0.0
    total_inference_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    savings_percent: float = 0.0

    # Per-deployment breakdown
    deployments: List[DeploymentCostSummary] = []


# Instance cost configuration (Spot prices)
# Supports both Lunar Router tier names and AWS instance types
INSTANCE_COSTS_USD_PER_HOUR: Dict[str, float] = {
    # ============================================
    # GPU Tiers (stored in deployments table)
    # ============================================

    # GPU XS - NVIDIA L4 (7B-13B models)
    "gpu-xs": 0.20,          # 1x L4, 24GB VRAM
    "gpu-xs-2x": 0.35,       # 1x L4, 24GB VRAM (more vCPU/RAM)

    # GPU S - NVIDIA L40S (13B-34B models)
    "gpu-s": 0.60,           # 1x L40S, 48GB VRAM
    "gpu-s-2x": 1.00,        # 1x L40S, 48GB VRAM (more vCPU/RAM)

    # GPU M - NVIDIA A10G (30B-70B INT4 models)
    "gpu-m": 1.80,           # 4x A10G, 96GB VRAM
    "gpu-m-2x": 3.00,        # 4x A10G, 96GB VRAM (more vCPU/RAM)
    "gpu-m-4x": 5.00,        # 8x A10G, 192GB VRAM

    # GPU L - NVIDIA L40S (70B FP16 models)
    "gpu-l": 3.50,           # 4x L40S, 192GB VRAM
    "gpu-l-2x": 6.00,        # 4x L40S, 192GB VRAM (more vCPU/RAM)
    "gpu-l-4x": 10.00,       # 8x L40S, 384GB VRAM

    # GPU XL - NVIDIA A100 (70B-180B models)
    "gpu-xl": 12.00,         # 8x A100 40GB, 320GB VRAM
    "gpu-xl-80gb": 18.00,    # 8x A100 80GB, 640GB VRAM

    # GPU XXL - NVIDIA H100/H200 (405B models)
    "gpu-xxl": 20.00,        # 8x H100 80GB, 640GB VRAM
    "gpu-xxl-h200": 30.00,   # 8x H200 141GB, 1128GB VRAM

    # ============================================
    # AWS Instance Types (alternative format)
    # ============================================

    # G6 - NVIDIA L4
    "g6.xlarge": 0.20,
    "g6.2xlarge": 0.35,

    # G6e - NVIDIA L40S
    "g6e.xlarge": 0.60,
    "g6e.2xlarge": 1.00,
    "g6e.12xlarge": 3.50,
    "g6e.24xlarge": 6.00,
    "g6e.48xlarge": 10.00,

    # G5 - NVIDIA A10G
    "g5.xlarge": 0.30,
    "g5.2xlarge": 0.50,
    "g5.4xlarge": 0.80,
    "g5.8xlarge": 1.20,
    "g5.12xlarge": 1.80,
    "g5.24xlarge": 3.00,
    "g5.48xlarge": 5.00,

    # P4d/P4de - NVIDIA A100
    "p4d.24xlarge": 12.00,
    "p4de.24xlarge": 18.00,

    # P5/P5e - NVIDIA H100/H200
    "p5.48xlarge": 20.00,
    "p5e.48xlarge": 30.00,

    # SageMaker format (ml. prefix)
    "ml.g5.2xlarge": 0.50,
    "ml.g5.xlarge": 0.30,
    "ml.g6.xlarge": 0.20,
    "ml.g6.2xlarge": 0.35,

    # Default fallback
    "default": 0.35,
}


def get_instance_cost(instance_type: Optional[str]) -> float:
    """
    Get the cost per hour for an instance type.
    Returns default cost if instance type is unknown.
    """
    if not instance_type:
        return INSTANCE_COSTS_USD_PER_HOUR["default"]
    return INSTANCE_COSTS_USD_PER_HOUR.get(
        instance_type,
        INSTANCE_COSTS_USD_PER_HOUR["default"]
    )
