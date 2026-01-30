"""
Semantic Router Service - Local Version.

Uses local JSON-based handlers instead of DynamoDB.
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

# Use local handlers
from ..database.local import LocalPricingHandler as PricingHandler
from ..database.local import LocalStatsHandler as TenantStatsHandler
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
    Service that bridges UniRoute's semantic routing with local adapters.
    """

    def __init__(self):
        self._embedder: Optional[PromptEmbedder] = None
        self._cluster_assigner = None
        self._registry: Optional[LLMRegistry] = None
        self._router: Optional[UniRouteRouter] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Lazy initialization of UniRoute components."""
        if self._initialized:
            return

        logger.info("Initializing SemanticRouterService...")

        state_path = Path(os.getenv(
            "UNIROUTE_STATE_PATH",
            Path(__file__).parent.parent / "data" / "uniroute_state"
        ))
        embedding_model = os.getenv("UNIROUTE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        default_cost_weight = float(os.getenv("UNIROUTE_DEFAULT_COST_WEIGHT", "0.0"))

        logger.info(f"Loading UniRoute state from: {state_path}")
        logger.info(f"Using embedding model: {embedding_model}")

        # 1. Initialize SentenceTransformers embedder
        provider = SentenceTransformerProvider(model_name=embedding_model)
        self._embedder = PromptEmbedder(provider, cache_enabled=True)
        logger.info(f"Embedder initialized with dimension: {self._embedder.dimension}")

        # 2. Load pre-trained cluster assigner
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

        available_profiles = [p.model_id for p in self._registry.get_all()]
        logger.info(f"Available profiles: {available_profiles}")

        # 4. Create router
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
        """Get list of models available that have profiles."""
        try:
            all_models = await PricingHandler.get_all_models()
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            all_models = []

        available_profiles = set(self.get_available_profiles())
        tenant_models = []

        for model_info in all_models:
            model_id = model_info.get("id", "") if isinstance(model_info, dict) else getattr(model_info, "id", "")
            if model_id in available_profiles:
                tenant_models.append(model_id)

        return tenant_models

    def extract_prompt_text(self, messages: List[ChatMessage]) -> str:
        """Extract prompt text from messages for embedding."""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content

        return "\n".join(m.content for m in messages)

    async def route(
        self,
        messages: List[ChatMessage],
        tenant_id: str,
        models: Optional[List[str]] = None,
        cost_weight: float = 0.0,
    ) -> RoutingDecision:
        """Make a routing decision without execution."""
        await self.initialize()

        available_profiles = set(self.get_available_profiles())

        if models:
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
            available = await self.get_tenant_models_with_profiles(tenant_id)

            if len(available) < 2:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "No models with semantic profiles available",
                        "code": "NO_PROFILES_AVAILABLE",
                        "available_profiles": list(available_profiles),
                        "tenant_models_with_profiles": available,
                    }
                )

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
        """Route a request and optionally execute it."""
        from ..adapter_openai_litellm import OpenAILiteLLMAdapter
        from ..adapter_anthropic import AnthropicAdapter
        from ..adapter_deepseek import DeepSeekAdapter
        from ..adapter_gemini import GeminiAdapter
        from ..adapter_mistral import MistralAdapter
        from ..adapter_groq import GroqAdapter
        from ..adapter_cohere import CohereAdapter

        decision = await self.route(
            messages=request.messages,
            tenant_id=tenant_id,
            models=request.models,
            cost_weight=request.cost_weight,
        )

        selected_model = decision.selected_model
        req_id = f"chatcmpl_{uuid.uuid4().hex}"

        routing_info = RoutingDecisionInfo(
            selected_model=selected_model,
            selected_provider="",
            expected_error=decision.expected_error,
            cost_adjusted_score=decision.cost_adjusted_score,
            cluster_id=decision.cluster_id,
            all_scores=decision.all_scores,
            reasoning=decision.reasoning,
        )

        if not request.execute:
            return SemanticRouterResponse(
                id=req_id,
                model=selected_model,
                choices=[],
                usage=None,
                routing=routing_info,
            )

        # Get providers for the selected model
        providers = await PricingHandler.get_providers_by_model(selected_model)
        if not providers:
            # Default to OpenAI
            adapter = OpenAILiteLLMAdapter(name="openai", logical_model=selected_model, model_name=selected_model)
        else:
            provider = providers[0]["provider"]
            modelid = providers[0].get("modelid")

            if provider == "openai":
                adapter = OpenAILiteLLMAdapter(name=provider, logical_model=selected_model, model_name=selected_model)
            elif provider == "anthropic":
                adapter = AnthropicAdapter(name=provider, logical_model=selected_model, model_name=f"anthropic/{modelid or selected_model}")
            elif provider == "deepseek":
                adapter = DeepSeekAdapter(name=provider, logical_model=selected_model, model_name=f"deepseek/{modelid or selected_model}")
            elif provider == "gemini":
                adapter = GeminiAdapter(name=provider, logical_model=selected_model, model_name=f"gemini/{modelid or selected_model}")
            elif provider == "mistral":
                adapter = MistralAdapter(name=provider, logical_model=selected_model, model_name=f"mistral/{modelid or selected_model}")
            elif provider == "groq":
                adapter = GroqAdapter(name=provider, logical_model=selected_model, model_name=f"groq/{modelid or selected_model}")
            elif provider == "cohere":
                adapter = CohereAdapter(name=provider, logical_model=selected_model, model_name=f"cohere_chat/{modelid or selected_model}")
            else:
                adapter = OpenAILiteLLMAdapter(name="openai", logical_model=selected_model, model_name=selected_model)

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
            async def generate_stream():
                async for chunk in resp["stream"]:
                    if isinstance(chunk, dict):
                        delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                    else:
                        try:
                            delta_content = chunk.choices[0].delta.get("content", "") or ""
                        except Exception:
                            delta_content = ""

                    if delta_content:
                        chunk_data = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "model": selected_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": delta_content},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "model": selected_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
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

        prompt_tokens = int(metrics.get("tokens_in", 0))
        completion_tokens = int(metrics.get("tokens_out", 0))
        total_tokens = prompt_tokens + completion_tokens

        # Get pricing
        price_info = None
        total_cost_usd = None
        try:
            price_response = await PricingHandler.get_price(adapter.name, selected_model)
            price = price_response.dict() if price_response else await PricingHandler.get_avg_price_by_provider(adapter.name)

            if price:
                bd = PricingHandler.breakdown_usd(
                    price={
                        "input_per_million": float(price.get("input_per_million", 0)),
                        "output_per_million": float(price.get("output_per_million", 0)),
                        "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                    },
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
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
        )
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)

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
