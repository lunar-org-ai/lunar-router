"""
Semantic Router Service.

Integrates UniRoute semantic routing with ECS Router adapters.
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from decimal import Decimal

# Lunar Router imports
from lunar_router import (
    UniRouteRouter,
    RoutingDecision,
    PromptEmbedder,
    SentenceTransformerProvider,
    load_cluster_assigner,
    LLMRegistry,
)

# ECS Router imports
from ..helpers.utils import adapters_for, publish_usage_event, generate_stream_response
from ..database.PricingHandler import PricingHandler
from ..database.TenantStatsHandler import TenantStatsHandler
from ..database.DeploymentHandler import DeploymentHandler
from ..database.models import TenantStatsModel, ProviderAttempt

from .schemas import (
    SemanticRouterRequest,
    SemanticRouterResponse,
    RoutingDecisionInfo,
    ChatMessage,
)

logger = logging.getLogger(__name__)


# Global singleton for the semantic router service
_semantic_service: Optional["SemanticRouterService"] = None


def get_semantic_service() -> "SemanticRouterService":
    """Get or create the global semantic router service."""
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticRouterService()
    return _semantic_service


class SemanticRouterService:
    """
    Service that bridges UniRoute's semantic routing with ECS adapters.

    This service:
    1. Loads pre-trained UniRoute components (embedder, clusters, profiles)
    2. Routes prompts using semantic analysis
    3. Executes on selected model using existing ECS adapters
    """

    def __init__(self):
        self._embedder: Optional[PromptEmbedder] = None
        self._cluster_assigner = None
        self._registry: Optional[LLMRegistry] = None
        self._router: Optional[UniRouteRouter] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Lazy initialization of UniRoute components.

        Called on first request to avoid cold-start delays on app startup.
        """
        if self._initialized:
            return

        logger.info("Initializing SemanticRouterService...")

        # Get configuration from environment
        state_path = Path(os.getenv(
            "UNIROUTE_STATE_PATH",
            Path(__file__).parent.parent / "data" / "uniroute_state"
        ))
        embedding_model = os.getenv("UNIROUTE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        default_cost_weight = float(os.getenv("UNIROUTE_DEFAULT_COST_WEIGHT", "0.0"))

        logger.info(f"Loading UniRoute state from: {state_path}")
        logger.info(f"Using embedding model: {embedding_model}")

        # 1. Initialize SentenceTransformers embedder (384 dims)
        # CRITICAL: Must use same model as training (all-MiniLM-L6-v2)
        provider = SentenceTransformerProvider(model_name=embedding_model)
        self._embedder = PromptEmbedder(provider, cache_enabled=True)
        logger.info(f"Embedder initialized with dimension: {self._embedder.dimension}")

        # 2. Load pre-trained cluster assigner (mmlu_full has 100 clusters to match profiles)
        cluster_path = state_path / "clusters" / "mmlu_full.npz"
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

        self._cluster_assigner = load_cluster_assigner(cluster_path)
        logger.info(f"Cluster assigner loaded with {self._cluster_assigner.num_clusters} clusters")

        # 3. Load all pre-trained profiles
        profiles_path = state_path / "profiles"
        if not profiles_path.exists():
            raise FileNotFoundError(f"Profiles directory not found: {profiles_path}")

        self._registry = LLMRegistry.load(profiles_path)
        logger.info(f"Registry loaded with {len(self._registry)} model profiles")

        # Log available profiles
        available_profiles = [p.model_id for p in self._registry.get_all()]
        logger.info(f"Available profiles: {available_profiles}")

        # 4. Create router (without allowed_models filter - we filter per-request)
        self._router = UniRouteRouter(
            embedder=self._embedder,
            cluster_assigner=self._cluster_assigner,
            registry=self._registry,
            cost_weight=default_cost_weight,
            use_soft_assignment=True,
        )

        self._initialized = True
        logger.info("SemanticRouterService initialization complete")

    def get_available_profiles(self) -> List[str]:
        """Return list of model IDs that have profiles."""
        if not self._registry:
            return []
        return [p.model_id for p in self._registry.get_all()]

    async def get_tenant_models_with_profiles(self, tenant_id: str) -> List[str]:
        """
        Get list of models available for the tenant that have profiles.

        Returns only models that:
        1. Are available in PricingTable for this tenant
        2. Have a pre-trained profile in the registry
        """
        # Get all models from pricing table
        try:
            all_models = await PricingHandler.get_all_models()
        except Exception as e:
            logger.error(f"Failed to get models from PricingTable: {e}")
            all_models = []

        # Filter to only models that have profiles
        available_profiles = set(self.get_available_profiles())
        tenant_models = []

        for model_info in all_models:
            # model_info is dict with "id" field from PricingHandler.get_all_models()
            model_id = model_info.get("id", "") if isinstance(model_info, dict) else getattr(model_info, "id", "")
            if model_id in available_profiles:
                tenant_models.append(model_id)

        return tenant_models

    def extract_prompt_text(self, messages: List[ChatMessage]) -> str:
        """
        Extract prompt text from messages for embedding.

        Uses the last user message for routing decision.
        """
        # Find last user message
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content

        # Fallback: concatenate all messages
        return "\n".join(m.content for m in messages)

    async def route(
        self,
        messages: List[ChatMessage],
        tenant_id: str,
        models: Optional[List[str]] = None,
        cost_weight: float = 0.0,
    ) -> RoutingDecision:
        """
        Make a routing decision without execution.

        Args:
            messages: Chat messages in OpenAI format.
            tenant_id: Tenant identifier for model lookup.
            models: Optional list of models to route among (min 2).
                   If None, uses all tenant models with profiles.
            cost_weight: Cost penalty weight (lambda).

        Returns:
            RoutingDecision from UniRoute.

        Raises:
            HTTPException: If validation fails or insufficient models.
        """
        await self.initialize()

        available_profiles = set(self.get_available_profiles())

        if models:
            # User-specified models - validate they ALL have profiles
            missing_profiles = [m for m in models if m not in available_profiles]

            if missing_profiles:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Models without profiles",
                        "code": "MISSING_PROFILES",
                        "missing_profiles": missing_profiles,
                        "available_profiles": list(available_profiles),
                    }
                )

            if len(models) < 2:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Semantic routing requires at least 2 models",
                        "code": "INSUFFICIENT_MODELS",
                        "provided": len(models),
                    }
                )

            available = models
        else:
            # Automatic mode - get all tenant models with profiles
            available = await self.get_tenant_models_with_profiles(tenant_id)

            if len(available) < 2:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "No models with semantic profiles available for this tenant",
                        "code": "NO_PROFILES_AVAILABLE",
                        "available_profiles": list(available_profiles),
                        "tenant_models_with_profiles": available,
                    }
                )

        # Extract prompt and route
        prompt = self.extract_prompt_text(messages)
        decision = self._router.route(
            prompt=prompt,
            available_models=available,
            cost_weight_override=cost_weight,
        )

        logger.info(
            f"Semantic routing: tenant={tenant_id}, "
            f"selected={decision.selected_model}, "
            f"cluster={decision.cluster_id}, "
            f"error={decision.expected_error:.4f}, "
            f"score={decision.cost_adjusted_score:.4f}"
        )

        return decision

    async def route_and_execute(
        self,
        request: SemanticRouterRequest,
        tenant_id: str,
        background_tasks: BackgroundTasks,
    ) -> SemanticRouterResponse:
        """
        Route a request and optionally execute it.

        Args:
            request: The semantic router request.
            tenant_id: Tenant identifier.
            background_tasks: FastAPI background tasks for async operations.

        Returns:
            SemanticRouterResponse with routing info and optional completion.
        """
        # Make routing decision
        decision = await self.route(
            messages=request.messages,
            tenant_id=tenant_id,
            models=request.models,
            cost_weight=request.cost_weight,
        )

        selected_model = decision.selected_model
        req_id = f"chatcmpl_{uuid.uuid4().hex}"

        # Build routing info
        routing_info = RoutingDecisionInfo(
            selected_model=selected_model,
            selected_provider="",  # Will be filled after adapter selection
            expected_error=decision.expected_error,
            cost_adjusted_score=decision.cost_adjusted_score,
            cluster_id=decision.cluster_id,
            all_scores=decision.all_scores,
            reasoning=decision.reasoning,
        )

        if not request.execute:
            # Return only routing decision
            return SemanticRouterResponse(
                id=req_id,
                model=selected_model,
                choices=[],
                usage=None,
                routing=routing_info,
            )

        # Execute using ECS adapters
        adapters = await adapters_for(selected_model, tenant_id=tenant_id)

        if not adapters:
            raise HTTPException(
                status_code=404,
                detail=f"No adapter found for model '{selected_model}'"
            )

        adapter = adapters[0]
        routing_info.selected_provider = adapter.name

        # Build request payload
        messages_dict = [m.model_dump() for m in request.messages]
        payload = {
            "tenant": tenant_id,
            "model": selected_model,
            "messages": messages_dict,
            "stream": request.stream,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        start_time = time.time()

        # Execute request
        resp, metrics = await adapter.send(payload)

        if "error" in resp:
            raise HTTPException(
                status_code=502,
                detail=f"Model execution failed: {resp.get('error')}"
            )

        # Handle streaming response
        if request.stream and "stream" in resp:
            tenant_stats = TenantStatsModel(
                TenantId=tenant_id,
                CreationDate=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                Provider=adapter.name,
                Model=selected_model,
                TTFT=Decimal("0.0"),
                Latency=Decimal("0.0"),
                Success=True,
                InputText=self.extract_prompt_text(request.messages),
                OutputText="",
                TotalTokens=0,
                RoutingInfo=json.dumps({
                    "type": "semantic",
                    "cluster_id": decision.cluster_id,
                    "expected_error": decision.expected_error,
                    "cost_adjusted_score": decision.cost_adjusted_score,
                }),
            )

            return StreamingResponse(
                generate_stream_response(
                    resp["stream"],
                    req_id,
                    selected_model,
                    tenant_id,
                    selected_model,
                    adapter.name,
                    start_time,
                    tenant_stats,
                    background_tasks,
                    metrics,
                ),
                media_type="text/plain",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Routing-Model": selected_model,
                    "X-Routing-Cluster": str(decision.cluster_id),
                }
            )

        # Non-streaming response
        resp_text = resp.get("text", "")
        latency_ms = (time.time() - start_time) * 1000

        # Calculate tokens and cost
        prompt_tokens = int(metrics.get("tokens_in", 0))
        completion_tokens = int(metrics.get("tokens_out", 0))
        total_tokens = prompt_tokens + completion_tokens

        # Get pricing
        price_info = None
        total_cost_usd = None
        try:
            if adapter.name == "pureai":
                # Check if adapter provided inference cost (new behavior)
                if "inference_cost_usd" in metrics:
                    # Use inference-time pricing
                    inference_cost = metrics.get("inference_cost_usd", 0.0)
                    total_cost_usd = round(inference_cost, 10)

                    price_info = {
                        "input_cost_usd": 0.0,
                        "output_cost_usd": 0.0,
                        "cache_input_cost_usd": 0.0,
                        "total_cost_usd": total_cost_usd,
                        "pricing_model": "inference_time",
                        "cost_per_hour": metrics.get("cost_per_hour"),
                        "inference_seconds": latency_ms / 1000.0,
                    }
                else:
                    # Fallback to token-based pricing (backwards compatibility)
                    price = {
                        "input_per_million": 0.10,
                        "output_per_million": 0.10,
                        "cache_input_per_million": 0.0,
                    }
                    cached_prompt_tokens = int(metrics.get("cached_prompt_tokens", 0))

                    bd = PricingHandler.breakdown_usd(
                        price={
                            "input_per_million": float(price.get("input_per_million", 0)),
                            "output_per_million": float(price.get("output_per_million", 0)),
                            "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                        },
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_prompt_tokens=cached_prompt_tokens,
                    )

                    price_info = {
                        "input_cost_usd": round(bd["input_cost_usd"], 10),
                        "output_cost_usd": round(bd["output_cost_usd"], 10),
                        "cache_input_cost_usd": round(bd["cache_input_cost_usd"], 10),
                        "total_cost_usd": round(bd["total_cost_usd"], 10),
                    }
                    total_cost_usd = bd["total_cost_usd"]
            else:
                price_response = await PricingHandler.get_price(adapter.name, selected_model)
                price = price_response.dict() if price_response else await PricingHandler.get_avg_price_by_provider(adapter.name)

                cached_prompt_tokens = int(metrics.get("cached_prompt_tokens", 0))

                bd = PricingHandler.breakdown_usd(
                    price={
                        "input_per_million": float(price.get("input_per_million", 0)),
                        "output_per_million": float(price.get("output_per_million", 0)),
                        "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                    },
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_prompt_tokens=cached_prompt_tokens,
                )

                price_info = {
                    "input_cost_usd": round(bd["input_cost_usd"], 10),
                    "output_cost_usd": round(bd["output_cost_usd"], 10),
                    "cache_input_cost_usd": round(bd["cache_input_cost_usd"], 10),
                    "total_cost_usd": round(bd["total_cost_usd"], 10),
                }
                total_cost_usd = bd["total_cost_usd"]
        except Exception as e:
            logger.warning(f"Failed to calculate pricing: {e}")

        # Record stats
        tenant_stats = TenantStatsModel(
            TenantId=tenant_id,
            CreationDate=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            Provider=adapter.name,
            Model=selected_model,
            TTFT=Decimal(str(metrics.get("ttft_ms", 0.0))),
            Latency=Decimal(str(latency_ms)),
            Success=True,
            InputText=self.extract_prompt_text(request.messages),
            OutputText=resp_text,
            TotalTokens=total_tokens,
            Cost=Decimal(str(total_cost_usd)) if total_cost_usd else None,
            RoutingInfo=json.dumps({
                "type": "semantic",
                "cluster_id": decision.cluster_id,
                "expected_error": decision.expected_error,
                "cost_adjusted_score": decision.cost_adjusted_score,
            }),
        )
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)

        # Publish usage event
        if price_info:
            background_tasks.add_task(
                publish_usage_event,
                request_id=req_id,
                tenant=tenant_id,
                logical_model=selected_model,
                provider=adapter.name,
                metrics=metrics,
                price_info=price_info,
            )

        # Build response
        return SemanticRouterResponse(
            id=req_id,
            model=selected_model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp_text,
                },
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                **(price_info or {}),
                "latency_ms": latency_ms,
                "ttft_ms": metrics.get("ttft_ms", 0),
            },
            routing=routing_info,
        )
