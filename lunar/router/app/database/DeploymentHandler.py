"""
DeploymentHandler - Manages EKS deployment data and cost calculations.

This handler provides:
- Deployment queries by tenant/id
- Inference time tracking
- Cost calculations based on inference time

Backwards-compatible: Works with deployments that don't have the new
cost fields (instance_type, cost_per_hour, etc.) by using defaults.
"""

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import aioboto3

from .deployment_models import (
    DeploymentModel,
    DeploymentCostSummary,
    DeploymentsCostOverview,
    get_instance_cost,
    INSTANCE_COSTS_USD_PER_HOUR,
)


# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEPLOYMENTS_TABLE = os.environ.get("DEPLOYMENTS_TABLE_NAME", "")


class DeploymentHandler:
    """Handler for EKS deployment operations and cost tracking."""

    region: str = AWS_REGION
    table_name: str = DEPLOYMENTS_TABLE

    @staticmethod
    def _parse_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DynamoDB item to Python dict."""
        result = {}
        for k, v in item.items():
            if "S" in v:
                result[k] = v["S"]
            elif "N" in v:
                result[k] = float(v["N"])
            elif "BOOL" in v:
                result[k] = v["BOOL"]
            elif "M" in v:
                result[k] = DeploymentHandler._parse_item(v["M"])
            elif "NULL" in v:
                result[k] = None
        return result

    @staticmethod
    def _item_to_model(item: Dict[str, Any]) -> DeploymentModel:
        """Convert parsed item to DeploymentModel with defaults for missing fields."""
        # Get instance type and cost, with backwards-compatible defaults
        instance_type = item.get("instance_type")
        cost_per_hour = item.get("cost_per_hour")

        # If cost_per_hour not set, derive from instance_type or use default
        if cost_per_hour is None:
            cost_per_hour = get_instance_cost(instance_type)

        return DeploymentModel(
            deployment_id=item.get("deployment_id", ""),
            tenant_id=item.get("tenant_id", ""),
            model_id=item.get("model_id", ""),
            status=item.get("status", "unknown"),
            service_url=item.get("service_url"),
            k8s_deployment_name=item.get("k8s_deployment_name") or item.get("endpoint_name"),
            k8s_namespace=item.get("k8s_namespace", "default"),
            instance_type=instance_type,
            cost_per_hour=cost_per_hour,
            created_at=item.get("created_at"),
            started_at=item.get("started_at"),
            paused_at=item.get("paused_at"),
            total_uptime_seconds=float(item.get("total_uptime_seconds", 0)),
            total_inference_seconds=float(item.get("total_inference_seconds", 0)),
            total_requests=int(item.get("total_requests", 0)),
        )

    @staticmethod
    async def get_deployment(deployment_id: str) -> Optional[DeploymentModel]:
        """
        Get a deployment by ID.

        Args:
            deployment_id: The deployment identifier

        Returns:
            DeploymentModel if found, None otherwise
        """
        if not DEPLOYMENTS_TABLE:
            return None

        session = aioboto3.Session()
        try:
            async with session.client("dynamodb", region_name=AWS_REGION) as client:
                response = await client.get_item(
                    TableName=DEPLOYMENTS_TABLE,
                    Key={"deployment_id": {"S": deployment_id}},
                )
                item = response.get("Item")
                if not item:
                    return None

                parsed = DeploymentHandler._parse_item(item)
                return DeploymentHandler._item_to_model(parsed)
        except Exception as e:
            print(f"[DeploymentHandler] Error getting deployment: {e}", flush=True)
            return None

    @staticmethod
    async def get_deployments_by_tenant(
        tenant_id: str,
        status_filter: Optional[str] = None
    ) -> List[DeploymentModel]:
        """
        Get all deployments for a tenant.

        Args:
            tenant_id: The tenant identifier
            status_filter: Optional status to filter by (in_service, paused, etc.)

        Returns:
            List of DeploymentModel
        """
        if not DEPLOYMENTS_TABLE:
            return []

        session = aioboto3.Session()
        deployments = []

        try:
            async with session.client("dynamodb", region_name=AWS_REGION) as client:
                # Query using tenant-status-index GSI
                query_kwargs = {
                    "TableName": DEPLOYMENTS_TABLE,
                    "IndexName": "tenant-status-index",
                }

                if status_filter:
                    query_kwargs["KeyConditionExpression"] = "tenant_id = :tid AND #st = :status"
                    query_kwargs["ExpressionAttributeNames"] = {"#st": "status"}
                    query_kwargs["ExpressionAttributeValues"] = {
                        ":tid": {"S": tenant_id},
                        ":status": {"S": status_filter},
                    }
                else:
                    # Get all statuses - need to query each status separately
                    # or use scan with filter (less efficient but simpler)
                    all_items = []
                    for status in ["in_service", "paused", "resuming", "creating"]:
                        resp = await client.query(
                            TableName=DEPLOYMENTS_TABLE,
                            IndexName="tenant-status-index",
                            KeyConditionExpression="tenant_id = :tid AND #st = :status",
                            ExpressionAttributeNames={"#st": "status"},
                            ExpressionAttributeValues={
                                ":tid": {"S": tenant_id},
                                ":status": {"S": status},
                            },
                        )
                        all_items.extend(resp.get("Items", []))

                    for item in all_items:
                        parsed = DeploymentHandler._parse_item(item)
                        deployments.append(DeploymentHandler._item_to_model(parsed))
                    return deployments

                response = await client.query(**query_kwargs)
                items = response.get("Items", [])

                for item in items:
                    parsed = DeploymentHandler._parse_item(item)
                    deployments.append(DeploymentHandler._item_to_model(parsed))

        except Exception as e:
            print(f"[DeploymentHandler] Error listing deployments: {e}", flush=True)

        return deployments

    @staticmethod
    async def update_inference_time(
        deployment_id: str,
        inference_seconds: float
    ) -> bool:
        """
        Atomically increment the inference time for a deployment.

        This is called after each request to track total inference time.
        Uses DynamoDB ADD operation for atomic updates.

        Args:
            deployment_id: The deployment identifier
            inference_seconds: Seconds of inference time to add

        Returns:
            True if update succeeded, False otherwise
        """
        if not DEPLOYMENTS_TABLE or inference_seconds <= 0:
            return False

        session = aioboto3.Session()
        try:
            async with session.client("dynamodb", region_name=AWS_REGION) as client:
                await client.update_item(
                    TableName=DEPLOYMENTS_TABLE,
                    Key={"deployment_id": {"S": deployment_id}},
                    UpdateExpression="ADD total_inference_seconds :inf, total_requests :one",
                    ExpressionAttributeValues={
                        ":inf": {"N": str(inference_seconds)},
                        ":one": {"N": "1"},
                    },
                )
                return True
        except Exception as e:
            print(f"[DeploymentHandler] Error updating inference time: {e}", flush=True)
            return False

    @staticmethod
    def calculate_request_cost(
        latency_ms: float,
        cost_per_hour: Optional[float] = None,
        instance_type: Optional[str] = None
    ) -> float:
        """
        Calculate the cost of a single request based on inference time.

        Formula: cost = (latency_seconds) × (cost_per_hour / 3600)

        Args:
            latency_ms: Request latency in milliseconds
            cost_per_hour: Cost per hour in USD (if known)
            instance_type: Instance type to look up cost (fallback)

        Returns:
            Cost in USD
        """
        if latency_ms <= 0:
            return 0.0

        # Determine cost per hour
        if cost_per_hour is None:
            cost_per_hour = get_instance_cost(instance_type)

        # Calculate cost
        inference_seconds = latency_ms / 1000.0
        cost_per_second = cost_per_hour / 3600.0
        return inference_seconds * cost_per_second

    @staticmethod
    async def get_deployment_cost_summary(
        deployment_id: str
    ) -> Optional[DeploymentCostSummary]:
        """
        Calculate detailed cost summary for a deployment.

        Returns:
            DeploymentCostSummary with all cost metrics
        """
        deployment = await DeploymentHandler.get_deployment(deployment_id)
        if not deployment:
            return None

        return DeploymentHandler._calculate_cost_summary(deployment)

    @staticmethod
    def _calculate_cost_summary(deployment: DeploymentModel) -> DeploymentCostSummary:
        """Calculate cost summary for a deployment model."""
        now = datetime.now(timezone.utc)

        # Calculate current session uptime if in_service
        current_uptime = 0.0
        if deployment.status == "in_service" and deployment.started_at:
            try:
                started = datetime.fromisoformat(deployment.started_at.replace("Z", "+00:00"))
                current_uptime = (now - started).total_seconds()
            except Exception:
                pass

        # Total uptime = stored + current session
        total_uptime = deployment.total_uptime_seconds + current_uptime

        # Get cost per hour
        cost_per_hour = deployment.cost_per_hour or get_instance_cost(deployment.instance_type)

        # Calculate costs
        cost_per_second = cost_per_hour / 3600.0
        uptime_cost = total_uptime * cost_per_second
        inference_cost = deployment.total_inference_seconds * cost_per_second
        savings = uptime_cost - inference_cost

        # Utilization percentage
        utilization = 0.0
        if total_uptime > 0:
            utilization = (deployment.total_inference_seconds / total_uptime) * 100

        return DeploymentCostSummary(
            deployment_id=deployment.deployment_id,
            model_id=deployment.model_id,
            instance_type=deployment.instance_type,
            status=deployment.status,
            current_uptime_seconds=current_uptime,
            total_uptime_seconds=total_uptime,
            total_inference_seconds=deployment.total_inference_seconds,
            utilization_percent=round(utilization, 2),
            cost_per_hour=cost_per_hour,
            uptime_cost_usd=round(uptime_cost, 6),
            inference_cost_usd=round(inference_cost, 6),
            savings_usd=round(savings, 6),
            started_at=deployment.started_at,
        )

    @staticmethod
    async def get_tenant_costs_overview(
        tenant_id: str
    ) -> DeploymentsCostOverview:
        """
        Get cost overview for all deployments of a tenant.

        Returns:
            DeploymentsCostOverview with aggregated metrics
        """
        deployments = await DeploymentHandler.get_deployments_by_tenant(tenant_id)

        summaries = []
        total_uptime = 0.0
        total_inference = 0.0
        total_uptime_cost = 0.0
        total_inference_cost = 0.0
        active_count = 0

        for deployment in deployments:
            summary = DeploymentHandler._calculate_cost_summary(deployment)
            summaries.append(summary)

            total_uptime += summary.total_uptime_seconds
            total_inference += summary.total_inference_seconds
            total_uptime_cost += summary.uptime_cost_usd
            total_inference_cost += summary.inference_cost_usd

            if deployment.status == "in_service":
                active_count += 1

        # Calculate overall utilization
        overall_utilization = 0.0
        if total_uptime > 0:
            overall_utilization = (total_inference / total_uptime) * 100

        # Calculate savings
        total_savings = total_uptime_cost - total_inference_cost
        savings_percent = 0.0
        if total_uptime_cost > 0:
            savings_percent = (total_savings / total_uptime_cost) * 100

        return DeploymentsCostOverview(
            tenant_id=tenant_id,
            period="all_time",
            total_deployments=len(deployments),
            active_deployments=active_count,
            total_uptime_hours=round(total_uptime / 3600, 2),
            total_inference_hours=round(total_inference / 3600, 2),
            overall_utilization_percent=round(overall_utilization, 2),
            total_uptime_cost_usd=round(total_uptime_cost, 6),
            total_inference_cost_usd=round(total_inference_cost, 6),
            total_savings_usd=round(total_savings, 6),
            savings_percent=round(savings_percent, 2),
            deployments=summaries,
        )

    @staticmethod
    def get_supported_instance_types() -> Dict[str, float]:
        """Return the dictionary of supported instance types and their costs."""
        return INSTANCE_COSTS_USD_PER_HOUR.copy()
