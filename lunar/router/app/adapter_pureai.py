# app/adapter_pureai.py
"""
PureAI Adapter - EKS/Kubernetes vLLM Adapter.

Simple passthrough to vLLM deployments on EKS.
Supports both /v1/chat/completions and /v1/completions.

Handles paused deployments:
- If deployment is paused, triggers auto-resume and returns 503 + Retry-After
- If deployment is resuming, returns 503 + Retry-After
"""

import asyncio
import os
import json
import time
from typing import Tuple, Dict, Any, Optional

import aioboto3
import aiohttp

from .adapters import ProviderAdapter
from .helpers.error_classifier import classify_error
from .models.error_types import ErrorCategory
from .database.deployment_models import get_instance_cost
from .database.DeploymentHandler import DeploymentHandler


# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEPLOYMENTS_TABLE = os.environ.get("DEPLOYMENTS_TABLE_NAME", "")
RESUME_STATE_MACHINE_ARN = os.environ.get("RESUME_STATE_MACHINE_ARN", "")

# HTTP timeout for vLLM requests
VLLM_TIMEOUT = aiohttp.ClientTimeout(total=120, connect=10)


class PureAIAdapter(ProviderAdapter):
    """
    Adapter for PureAI self-hosted models on EKS (vLLM).
    Simple passthrough - forwards requests directly to vLLM.
    """

    def __init__(self, name: str, logical_model: str, tenant_id: str):
        super().__init__(name=name, model=logical_model)
        self.logical_model = logical_model
        self.tenant_id = tenant_id
        self._session = aioboto3.Session()

    async def _get_deployment(self, include_paused: bool = False) -> Optional[Dict[str, Any]]:
        """
        Look up EKS deployment from DynamoDB.

        Args:
            include_paused: If True, also looks for paused/resuming deployments
        """
        if not DEPLOYMENTS_TABLE:
            return None

        try:
            async with self._session.client("dynamodb", region_name=AWS_REGION) as ddb:
                # First try to find an in_service deployment
                resp = await ddb.query(
                    TableName=DEPLOYMENTS_TABLE,
                    IndexName="tenant-status-index",
                    KeyConditionExpression="tenant_id = :tid AND #st = :status",
                    FilterExpression="contains(model_id, :model)",
                    ExpressionAttributeNames={"#st": "status"},
                    ExpressionAttributeValues={
                        ":tid": {"S": self.tenant_id},
                        ":status": {"S": "in_service"},
                        ":model": {"S": self.logical_model},
                    },
                )
                items = resp.get("Items", [])
                if items:
                    return self._parse_item(items[0])

                # If no in_service found and include_paused is True, check for paused/resuming
                if include_paused:
                    for status in ["paused", "resuming"]:
                        resp = await ddb.query(
                            TableName=DEPLOYMENTS_TABLE,
                            IndexName="tenant-status-index",
                            KeyConditionExpression="tenant_id = :tid AND #st = :status",
                            FilterExpression="contains(model_id, :model)",
                            ExpressionAttributeNames={"#st": "status"},
                            ExpressionAttributeValues={
                                ":tid": {"S": self.tenant_id},
                                ":status": {"S": status},
                                ":model": {"S": self.logical_model},
                            },
                        )
                        items = resp.get("Items", [])
                        if items:
                            return self._parse_item(items[0])

                return None
        except Exception as e:
            print(f"[PureAI] DynamoDB error: {e}", flush=True)
            return None

    async def _trigger_resume(self, deployment: Dict[str, Any]) -> bool:
        """
        Trigger the resume Step Function for a paused deployment.

        Returns True if successfully triggered, False otherwise.
        """
        if not RESUME_STATE_MACHINE_ARN:
            print("[PureAI] RESUME_STATE_MACHINE_ARN not configured", flush=True)
            return False

        deployment_id = deployment.get("deployment_id")

        sfn_input = {
            "deployment_id": deployment_id,
            "endpoint_name": deployment.get("endpoint_name") or deployment.get("k8s_deployment_name"),
            "tenant_id": str(self.tenant_id),
            "k8s_namespace": deployment.get("k8s_namespace", "default"),
            "replicas": 1,
        }

        try:
            async with self._session.client("stepfunctions", region_name=AWS_REGION) as sfn:
                resp = await sfn.start_execution(
                    stateMachineArn=RESUME_STATE_MACHINE_ARN,
                    input=json.dumps(sfn_input),
                )
                print(f"[PureAI] Auto-resume triggered for {deployment_id}: {resp.get('executionArn')}", flush=True)
                return True
        except Exception as e:
            # Check if already running
            if "ExecutionAlreadyExists" in str(e):
                print(f"[PureAI] Resume already in progress for {deployment_id}", flush=True)
                return True
            print(f"[PureAI] Failed to trigger resume: {e}", flush=True)
            return False

    def _parse_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DynamoDB item."""
        result = {}
        for k, v in item.items():
            if "S" in v:
                result[k] = v["S"]
            elif "N" in v:
                result[k] = float(v["N"])
            elif "M" in v:
                result[k] = self._parse_item(v["M"])
            elif "BOOL" in v:
                result[k] = v["BOOL"]
        return result

    async def send(self, req: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Send inference request to vLLM on EKS. Supports streaming if 'stream' in req."""
        start = time.time()

        # Get deployment - first try in_service only
        deployment = await self._get_deployment(include_paused=False)
        if not deployment:
            # No in_service deployment found, check for paused/resuming
            deployment = await self._get_deployment(include_paused=True)

            if deployment:
                status = deployment.get("status", "").lower()

                if status == "paused":
                    # Trigger auto-resume
                    await self._trigger_resume(deployment)
                    return (
                        {
                            "error": "Deployment is paused. Auto-resuming, please retry in 2 minutes.",
                            "error_category": ErrorCategory.DEPLOYMENT_ERROR.value,
                            "error_details": {
                                "exception_type": "DeploymentPaused",
                                "provider": self.name,
                                "deployment_status": "paused",
                            },
                            "retry_after": 120,
                            "status": "resuming"
                        },
                        {"error": 1.0, "latency_ms": (time.time() - start) * 1000},
                    )

                elif status == "resuming":
                    return (
                        {
                            "error": "Deployment is resuming. Please retry in 1 minute.",
                            "error_category": ErrorCategory.DEPLOYMENT_ERROR.value,
                            "error_details": {
                                "exception_type": "DeploymentResuming",
                                "provider": self.name,
                                "deployment_status": "resuming",
                            },
                            "retry_after": 60,
                            "status": "resuming"
                        },
                        {"error": 1.0, "latency_ms": (time.time() - start) * 1000},
                    )

            # No deployment at all
            return (
                {
                    "error": f"No active deployment for {self.logical_model}",
                    "error_category": ErrorCategory.DEPLOYMENT_ERROR.value,
                    "error_details": {
                        "exception_type": "NoDeployment",
                        "provider": self.name,
                        "model": self.logical_model,
                    },
                },
                {"error": 1.0, "latency_ms": (time.time() - start) * 1000},
            )

        service_url = deployment.get("service_url")
        if not service_url:
            return (
                {
                    "error": "Deployment missing service_url",
                    "error_category": ErrorCategory.DEPLOYMENT_ERROR.value,
                    "error_details": {
                        "exception_type": "MissingServiceUrl",
                        "provider": self.name,
                    },
                },
                {"error": 1.0, "latency_ms": (time.time() - start) * 1000},
            )

        # Build vLLM request - passthrough with model override
        model_id = deployment.get("model_id", self.logical_model)
        base_url = service_url.rstrip('/')

        # Determine request type and endpoint
        is_chat_request = "messages" in req
        stream = req.get("stream", False)

        if is_chat_request:
            endpoint = f"{base_url}/v1/chat/completions"
            payload = {
                "model": model_id,
                "messages": req.get("messages", []),
                "max_tokens": req.get("max_tokens", 1024),
                "temperature": req.get("temperature", 0.7),
            }
        else:
            endpoint = f"{base_url}/v1/completions"
            payload = {
                "model": model_id,
                "prompt": req.get("prompt", ""),
                "max_tokens": req.get("max_tokens", 1024),
                "temperature": req.get("temperature", 0.7),
            }
            # Add stop sequences if provided
            if req.get("stop"):
                payload["stop"] = req["stop"]

        if stream:
            # Streaming mode: yield chunks as they arrive from vLLM
            async def stream_generator():
                first_chunk = True
                collected_text = ""
                ttft_ms = 0.0
                try:
                    async with aiohttp.ClientSession(timeout=VLLM_TIMEOUT) as session:
                        async with session.post(endpoint, json={**payload, "stream": True}) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                yield {
                                    "error": f"vLLM {resp.status}: {error_text}",
                                    "error_category": ErrorCategory.SERVER_ERROR.value,
                                    "error_details": {
                                        "exception_type": "HTTPError",
                                        "provider": self.name,
                                        "http_status": resp.status,
                                    },
                                }
                                return
                            
                            # Read streaming response line by line
                            buffer = ""
                            try:
                                async for chunk_bytes in resp.content.iter_any():
                                    buffer += chunk_bytes.decode('utf-8', errors='ignore')
                                    lines = buffer.split('\n')
                                    # Keep the last incomplete line in buffer
                                    buffer = lines[-1]
                                    
                                    for line in lines[:-1]:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        
                                        # vLLM streams lines as 'data: {...}' or '[DONE]'
                                        if line.startswith("data: "):
                                            data = line[len("data: "):]
                                            if data == "[DONE]":
                                                break
                                            try:
                                                chunk = json.loads(data)
                                            except Exception as e:
                                                print(f"[PureAI] Failed to parse JSON: {e}, data: {data}", flush=True)
                                                continue
                                            
                                            if first_chunk:
                                                ttft_ms = (time.time() - start) * 1000.0
                                                first_chunk = False
                                            
                                            # Extract text from chunk - handle both chat and completion formats
                                            delta = ""
                                            if is_chat_request:
                                                # Chat format: choices[0].delta.content
                                                try:
                                                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                                                except Exception:
                                                    delta = ""
                                            else:
                                                # Completion format: choices[0].text
                                                try:
                                                    delta = chunk.get("choices", [{}])[0].get("text", "") or ""
                                                except Exception:
                                                    delta = ""
                                            
                                            # Debug: log chunk structure
                                            print(f"[PureAI] Chunk (is_chat={is_chat_request}): {json.dumps(chunk)[:200]}", flush=True)
                                            print(f"[PureAI] Extracted delta: '{delta}'", flush=True)
                                            
                                            if delta:
                                                collected_text += delta
                                                yield chunk
                            
                            except Exception as stream_error:
                                print(f"[PureAI] Stream reading error: {stream_error}", flush=True)
                                raise
                            
                            # Process any remaining data in buffer
                            if buffer.strip():
                                line = buffer.strip()
                                if line.startswith("data: "):
                                    data = line[len("data: "):]
                                    if data != "[DONE]":
                                        try:
                                            chunk = json.loads(data)
                                            delta = ""
                                            if is_chat_request:
                                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                                            else:
                                                delta = chunk.get("choices", [{}])[0].get("text", "") or ""
                                            if delta:
                                                collected_text += delta
                                                yield chunk
                                        except Exception:
                                            pass
                
                except Exception as e:
                    print(f"[PureAI] Streaming error: {e}", flush=True)
                    yield {
                        "error": f"pureai streaming error: {e}",
                        "error_category": classify_error(e),
                        "error_details": {
                            "exception_type": type(e).__name__,
                            "provider": self.name,
                        },
                    }

            # Get cost info for streaming (will be used after stream ends)
            cost_per_hour = deployment.get("cost_per_hour")
            instance_type = deployment.get("instance_type")
            if cost_per_hour is None:
                cost_per_hour = get_instance_cost(instance_type)
            deployment_id = deployment.get("deployment_id")

            # Return the generator and metrics placeholder
            return (
                {
                    "stream": stream_generator(),
                    "model": model_id,
                },
                {
                    "ttft_ms": 0.0,  # Will be set on first chunk
                    "latency_ms": 0.0,  # Will be set after stream ends
                    "tokens_in": 0,  # Not available here
                    "tokens_out": 0,  # Not available here
                    "error": 0.0,
                    # Inference-time pricing fields (for post-stream cost calculation)
                    "cost_per_hour": cost_per_hour,
                    "instance_type": instance_type,
                    "deployment_id": deployment_id,
                },
            )

        # Non-streaming mode (original logic)
        try:
            async with aiohttp.ClientSession(timeout=VLLM_TIMEOUT) as session:
                async with session.post(endpoint, json=payload) as resp:
                    latency_ms = (time.time() - start) * 1000

                    if resp.status != 200:
                        error_text = await resp.text()

                        # Check for chat template error and provide clear message
                        if resp.status == 400 and "chat template" in error_text.lower():
                            return (
                                {
                                    "error": f"Model '{self.logical_model}' does not support chat; use /v1/completions or choose a chat-capable model.",
                                    "error_category": ErrorCategory.INVALID_REQUEST.value,
                                    "error_details": {
                                        "exception_type": "ChatTemplateError",
                                        "provider": self.name,
                                        "http_status": resp.status,
                                    },
                                },
                                {"error": 1.0, "latency_ms": latency_ms},
                            )

                        # Classify error based on HTTP status
                        error_category = ErrorCategory.SERVER_ERROR.value
                        if resp.status == 401:
                            error_category = ErrorCategory.AUTH_ERROR.value
                        elif resp.status == 429:
                            error_category = ErrorCategory.RATE_LIMIT.value
                        elif resp.status == 400:
                            error_category = ErrorCategory.INVALID_REQUEST.value

                        return (
                            {
                                "error": f"vLLM {resp.status}: {error_text}",
                                "error_category": error_category,
                                "error_details": {
                                    "exception_type": "HTTPError",
                                    "provider": self.name,
                                    "http_status": resp.status,
                                },
                            },
                            {"error": 1.0, "latency_ms": latency_ms},
                        )

                    result = await resp.json()

                    # Extract text from response
                    choices = result.get("choices", [])
                    if not choices:
                        return (
                            {
                                "error": "Empty response from vLLM",
                                "error_category": ErrorCategory.MODEL_ERROR.value,
                                "error_details": {
                                    "exception_type": "EmptyResponse",
                                    "provider": self.name,
                                },
                            },
                            {"error": 1.0, "latency_ms": latency_ms},
                        )

                    # Handle both chat and completions response formats
                    choice = choices[0]
                    if "message" in choice:
                        text = choice["message"].get("content", "")
                    else:
                        text = choice.get("text", "")

                    usage = result.get("usage", {})

                    # Calculate inference cost based on latency and instance cost
                    cost_per_hour = deployment.get("cost_per_hour")
                    instance_type = deployment.get("instance_type")
                    if cost_per_hour is None:
                        cost_per_hour = get_instance_cost(instance_type)

                    inference_cost = DeploymentHandler.calculate_request_cost(
                        latency_ms=latency_ms,
                        cost_per_hour=cost_per_hour,
                        instance_type=instance_type
                    )

                    # Update deployment inference time in background (non-blocking)
                    deployment_id = deployment.get("deployment_id")
                    if deployment_id:
                        asyncio.create_task(
                            DeploymentHandler.update_inference_time(
                                deployment_id=deployment_id,
                                inference_seconds=latency_ms / 1000.0
                            )
                        )

                    return (
                        {"text": text},
                        {
                            "ttft_ms": 0.0,
                            "latency_ms": latency_ms,
                            "tokens_in": usage.get("prompt_tokens", 0),
                            "tokens_out": usage.get("completion_tokens", 0),
                            "error": 0.0,
                            # Inference-time pricing fields
                            "inference_cost_usd": inference_cost,
                            "cost_per_hour": cost_per_hour,
                            "instance_type": instance_type,
                            "deployment_id": deployment_id,
                        },
                    )

        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            print(f"[PureAI] Timeout error (is_chat={is_chat_request}): {e}", flush=True)
            latency_ms = (time.time() - start) * 1000

            # If timeout on chat request, it might be because model doesn't support chat
            if is_chat_request:
                return (
                    {
                        "error": f"Model '{self.logical_model}' does not support chat; use /v1/completions or choose a chat-capable model.",
                        "error_category": ErrorCategory.INVALID_REQUEST.value,
                        "error_details": {
                            "exception_type": "ChatTimeoutError",
                            "provider": self.name,
                            "hint": "Model may not have a chat template configured",
                        },
                    },
                    {"error": 1.0, "latency_ms": latency_ms},
                )

            return (
                {
                    "error": f"pureai timeout: {e}",
                    "error_category": ErrorCategory.SERVER_ERROR.value,
                    "error_details": {
                        "exception_type": type(e).__name__,
                        "provider": self.name,
                    },
                },
                {"error": 1.0, "latency_ms": latency_ms},
            )

        except Exception as e:
            print(f"[PureAI] Error: {e}", flush=True)
            latency_ms = (time.time() - start) * 1000
            error_category = classify_error(e)
            return (
                {
                    "error": f"pureai error: {e}",
                    "error_category": error_category,
                    "error_details": {
                        "exception_type": type(e).__name__,
                        "provider": self.name,
                    },
                },
                {"error": 1.0, "latency_ms": latency_ms},
            )

    def healthy(self) -> Dict[str, Any]:
        """Health check."""
        ok = bool(DEPLOYMENTS_TABLE)
        return {"ok": ok, "err_rate": 0.0 if ok else 1.0, "headroom": 1.0}
