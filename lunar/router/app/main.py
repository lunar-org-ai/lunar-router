import json
import time
import uuid
import json
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .middleware.auth_middleware import AuthMiddleware

from .schemas import (
    ProviderInfo,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    Usage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
)
from .router import HealthFirstPlanner
from .database.models import TenantStatsModel, ProviderAttempt
from .database.TenantStatsHandler import TenantStatsHandler
from datetime import datetime
from decimal import Decimal
from .database.PricingHandler import PricingHandler
from .routes import stats, pricing, infer, conversations, semantic_router, deployments, rag, data_plane
from .helpers.utils import _parse_model_string, adapters_for, publish_usage_event, generate_stream_response

# Setup

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")  # optional

MODELS_DIR = BASE / "configs" / "models"

app = FastAPI(
    title="Lunar Router - Health-First LLM Routing (Async + Global Metrics)",
    version="0.5.0",
)

# Add CORS middleware
# Note: When allow_credentials=True, allow_origins cannot be ["*"]
# Must specify exact origins for CORS to work with credentials
ALLOWED_ORIGINS = [
    "https://dev.dqt4cqxof77im.amplifyapp.com",
    "https://temp-rebuild.d13gur8sx7lqmc.amplifyapp.com",
    "https://dev.d13gur8sx7lqmc.amplifyapp.com",
    "https://main.d13gur8sx7lqmc.amplifyapp.com",
    "https://lunar.pureai-console.com",
    "https://pureai-console.com",
    "http://localhost:3000",
]

# Add authentication middleware BEFORE CORS (middlewares execute in reverse order)
# So CORS will process the response AFTER auth, adding headers to 401 responses
app.add_middleware(AuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats.router)
app.include_router(pricing.router)
app.include_router(infer.router)
app.include_router(conversations.router)
app.include_router(semantic_router.router)
app.include_router(deployments.router)
app.include_router(rag.router)
app.include_router(data_plane.router)

@app.get("/health")
async def health_check():
    """
    Health check for ALB / automations.
    """
    return {"status": "ok"}

planner = HealthFirstPlanner()


# Discovery & Metrics APIs
# Note: Model listing moved to /v1/pricing/models (reads from PricingTable DynamoDB)
@app.get("/v1/providers", response_model=List[ProviderInfo])
async def providers(
    model: str = Query(None, description="Logical model name to list providers for"),
):
    """
   /providers?model=gpt-4o-mini => lists providers for this logical model.
    If you don't pass the model, try "gpt-4o-mini" as an example.
    """
    import yaml

    target_model = model or "gpt-4o-mini"
    cfg_path = MODELS_DIR / f"{target_model}.yaml"
    if not cfg_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{target_model}' not found.",
        )

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    out: List[ProviderInfo] = []
    for p in cfg.get("providers", []):
        out.append(
            ProviderInfo(
                id=p["id"],
                type=p["type"],
                enabled=p.get("enabled", True),
                params=p.get("params", {}),
            )
        )
    return out

# OpenAI-compatible /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    OpenAI-style endpoint with router:

    - model="gpt-4o-mini"

    - routes between providers for this logical model

    - model="openai/gpt-4o-mini"

    - forces provider "openai" for this logical model
    """
    logical_model, forced_provider = _parse_model_string(req.model)
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    # Get adapters - pass forced_provider and tenant_id for pureai routing
    adapters = await adapters_for(logical_model, forced_provider=forced_provider, tenant_id=tenant_id)

    # Decide provider set: forced ou ranked
    if forced_provider:
        candidate = next((a for a in adapters if a.name == forced_provider), None)
        if not candidate:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{forced_provider}' not found for model '{logical_model}'",
            )
        ordered = [candidate]
    else:
        # Only rank if there are multiple adapters
        ordered = await planner.rank(logical_model, adapters) if len(adapters) > 1 else adapters

    last_error = None
    chosen_adapter_name: Optional[str] = None
    resp_text: Optional[str] = None
    metrics_used: Dict[str, float] = {}
    tenant_stats: TenantStatsModel = None
    provider_attempts: List[ProviderAttempt] = []

    # Fallback loop
    for chosen in ordered:
        attempt_start = datetime.utcnow()
        resp, metrics = await chosen.send(
            {
                "tenant": tenant_id,
                "model": logical_model,
                "messages": [m.model_dump() for m in req.messages],
                "stream": req.stream,
            }
        )

        # Track this provider attempt
        attempt = ProviderAttempt(
            provider=chosen.name,
            success="error" not in resp,
            error_category=resp.get("error_category") if "error" in resp else None,
            error_message=resp.get("error") if "error" in resp else None,
            latency_ms=metrics.get("latency_ms", 0),
            timestamp=attempt_start.isoformat(),
        )
        provider_attempts.append(attempt)

        tenant_stats = TenantStatsModel(TenantId=tenant_id,
                                    CreationDate=datetime.utcnow().isoformat(),
                                    Provider=chosen.name,
                                    Model=logical_model,
                                    TTFT=Decimal(str(metrics.get("ttft_ms", 0.0))),
                                    Latency=Decimal(str(metrics.get("latency_ms", 0.0))),
                                    Success=not bool(metrics.get("error", 0.0)),
                                    # Should we add all the context or only the latest user message?
                                    InputText="".join([m.content for m in req.messages]),
                                    OutputText=resp.get("text", ""),
                                    TotalTokens=int(metrics.get("tokens_in", 0)) + int(metrics.get("tokens_out", 0)),
                                    # New fields for enhanced error tracking
                                    ErrorCategory=resp.get("error_category") if "error" in resp else None,
                                    ErrorMessage=resp.get("error") if "error" in resp else None,
                                    ProviderAttempts=json.dumps([a.model_dump() for a in provider_attempts]),
                                    FallbackCount=len(provider_attempts),
                                    FinalProvider=chosen.name)

        if "error" not in resp:
            chosen_adapter_name = chosen.name
            metrics_used = metrics
            
            # Handle streaming response
            if req.stream and "stream" in resp:
                req_id = f"chatcmpl_{uuid.uuid4().hex}"
                start_time = time.time()
                
                return StreamingResponse(
                    generate_stream_response(
                        resp["stream"], 
                        req_id, 
                        req.model, 
                        tenant_id, 
                        logical_model, 
                        chosen_adapter_name, 
                        start_time,
                        tenant_stats,
                        background_tasks,
                        metrics
                    ),
                    media_type="text/plain",
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            
            resp_text = resp.get("text", "")
            break

        last_error = resp.get("error")

    if resp_text is None:
        tenant_stats.ErrorType = last_error
        await TenantStatsHandler.insert(tenant_stats)

        # Check if the error is about chat not being supported
        error_str = str(last_error).lower() if last_error else ""
        if "does not support chat" in error_str or "chat template" in error_str:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{logical_model}' does not support chat completions. Use /v1/completions instead.",
            )

        raise HTTPException(
            status_code=502,
            detail=f"All providers failed for model '{logical_model}': {last_error}",
        )

    # Token usage
    prompt_tokens = int(metrics_used.get("tokens_in", 0))
    completion_tokens = int(metrics_used.get("tokens_out", 0))
    total_tokens = prompt_tokens + completion_tokens

    # Cost breakdown
    input_cost = output_cost = cache_input_cost = total_cost_usd = None
    price_info: Optional[Dict[str, float]] = None

    if chosen_adapter_name:
        provider = chosen_adapter_name
        model = logical_model

        # PureAI uses inference-time pricing for self-hosted models
        if provider == "pureai":
            # Check if adapter provided inference cost (new behavior)
            if "inference_cost_usd" in metrics_used:
                # Use inference-time pricing
                inference_cost = metrics_used.get("inference_cost_usd", 0.0)
                price = None  # Not used when inference cost is provided
            else:
                # Fallback to token-based pricing (backwards compatibility)
                price = {
                    "input_per_million": 0.10,
                    "output_per_million": 0.10,
                    "cache_input_per_million": 0.0,
                }
        else:
            price_response = await PricingHandler.get_price(provider, model)
            price = price_response.dict() if price_response else await PricingHandler.get_avg_price_by_provider(provider)

        def r(x: float) -> float:
            return round(x, 10)

        # Use inference-time pricing for pureai if available
        if provider == "pureai" and "inference_cost_usd" in metrics_used:
            # Inference-time pricing (cost based on GPU time, not tokens)
            inference_cost = metrics_used.get("inference_cost_usd", 0.0)
            total_cost_usd = r(inference_cost)

            price_info = {
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "cache_input_cost_usd": 0.0,
                "total_cost_usd": total_cost_usd,
                "pricing_model": "inference_time",
                "cost_per_hour": metrics_used.get("cost_per_hour"),
                "inference_seconds": (metrics_used.get("latency_ms", 0) / 1000.0),
            }
        else:
            # Token-based pricing (API providers and fallback)
            cached_prompt_tokens = int(metrics_used.get("cached_prompt_tokens", 0))

            bd = PricingHandler.breakdown_usd(
                price={
                    "input_per_million": float(price.get("input_per_million", 0)),
                    "output_per_million": float(price.get("output_per_million", 0)),
                    "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                },
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_prompt_tokens)

            input_cost = r(bd["input_cost_usd"])
            cache_input_cost = r(bd["cache_input_cost_usd"])
            output_cost = r(bd["output_cost_usd"])
            total_cost_usd = r(bd["total_cost_usd"])

            price_info = {
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "cache_input_cost_usd": cache_input_cost,
                "total_cost_usd": total_cost_usd,
            }

        tenant_stats.Cost = Decimal(str(total_cost_usd))
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)

    req_id = f"chatcmpl_{uuid.uuid4().hex}"

    if (chosen_adapter_name):
        background_tasks.add_task(
            publish_usage_event,
            request_id=req_id,
            tenant=tenant_id,
            logical_model=logical_model,
            provider=chosen_adapter_name,
            metrics=metrics_used,
            price_info=price_info,
        )

    # Extract latency metrics
    latency_ms = float(metrics_used.get("latency_ms", 0)) if metrics_used else None
    ttft_ms = float(metrics_used.get("ttft_ms", 0)) if metrics_used else None

    return ChatCompletionResponse(
        id=req_id,
        model=req.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=resp_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cache_input_cost_usd=cache_input_cost,
            total_cost_usd=total_cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
        ),
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    req: CompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    OpenAI-style text completions endpoint (legacy).

    - model="gpt-4o-mini" or "pureai/Llama-3.2-1B"
    - prompt="Hello, how are you?"
    """
    logical_model, forced_provider = _parse_model_string(req.model)
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    # Get adapters
    adapters = await adapters_for(logical_model, forced_provider=forced_provider, tenant_id=tenant_id)

    # Decide provider set: forced or ranked
    if forced_provider:
        candidate = next((a for a in adapters if a.name == forced_provider), None)
        if not candidate:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{forced_provider}' not found for model '{logical_model}'",
            )
        ordered = [candidate]
    else:
        # Only rank if there are multiple adapters
        ordered = await planner.rank(logical_model, adapters) if len(adapters) > 1 else adapters

    last_error = None
    chosen_adapter_name: Optional[str] = None
    resp_text: Optional[str] = None
    metrics_used: Dict[str, float] = {}
    tenant_stats: TenantStatsModel = None
    provider_attempts: List[ProviderAttempt] = []

    # Fallback loop
    for chosen in ordered:
        attempt_start = datetime.utcnow()
        resp, metrics = await chosen.send(
            {
                "tenant": tenant_id,
                "model": logical_model,
                "prompt": req.prompt,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stop": req.stop,
                "stream": getattr(req, "stream", False),
            }
        )

        # Track this provider attempt
        attempt = ProviderAttempt(
            provider=chosen.name,
            success="error" not in resp,
            error_category=resp.get("error_category") if "error" in resp else None,
            error_message=resp.get("error") if "error" in resp else None,
            latency_ms=metrics.get("latency_ms", 0),
            timestamp=attempt_start.isoformat(),
        )
        provider_attempts.append(attempt)

        tenant_stats = TenantStatsModel(
            TenantId=tenant_id,
            CreationDate=datetime.utcnow().isoformat(),
            Provider=chosen.name,
            Model=logical_model,
            TTFT=Decimal(str(metrics.get("ttft_ms", 0.0))),
            Latency=Decimal(str(metrics.get("latency_ms", 0.0))),
            Success=not bool(metrics.get("error", 0.0)),
            InputText=req.prompt,
            OutputText=resp.get("text", ""),
            TotalTokens=int(metrics.get("tokens_in", 0)) + int(metrics.get("tokens_out", 0)),
            # New fields for enhanced error tracking
            ErrorCategory=resp.get("error_category") if "error" in resp else None,
            ErrorMessage=resp.get("error") if "error" in resp else None,
            ProviderAttempts=json.dumps([a.model_dump() for a in provider_attempts]),
            FallbackCount=len(provider_attempts),
            FinalProvider=chosen.name
        )

        if "error" not in resp:
            chosen_adapter_name = chosen.name
            metrics_used = metrics
            # Handle streaming response
            if getattr(req, "stream", False) and "stream" in resp:
                req_id = f"cmpl_{uuid.uuid4().hex}"
                start_time = time.time()
                from .helpers.utils import generate_stream_response
                return StreamingResponse(
                    generate_stream_response(
                        resp["stream"],
                        req_id,
                        req.model,
                        tenant_id,
                        logical_model,
                        chosen_adapter_name,
                        start_time,
                        tenant_stats,
                        background_tasks,
                        metrics,
                        is_chat=False
                    ),
                    media_type="text/plain",
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            resp_text = resp.get("text", "")
            break

        last_error = resp.get("error")

    if resp_text is None:
        tenant_stats.ErrorType = last_error
        await TenantStatsHandler.insert(tenant_stats)
        raise HTTPException(
            status_code=502,
            detail=f"All providers failed for model '{logical_model}': {last_error}",
        )

    # Token usage
    prompt_tokens = int(metrics_used.get("tokens_in", 0))
    completion_tokens = int(metrics_used.get("tokens_out", 0))
    total_tokens = prompt_tokens + completion_tokens

    # Cost breakdown
    input_cost = output_cost = cache_input_cost = total_cost_usd = None
    price_info: Optional[Dict[str, float]] = None

    if chosen_adapter_name:
        provider = chosen_adapter_name
        model = logical_model

        # PureAI uses inference-time pricing for self-hosted models
        if provider == "pureai":
            # Check if adapter provided inference cost (new behavior)
            if "inference_cost_usd" in metrics_used:
                price = None  # Not used when inference cost is provided
            else:
                # Fallback to token-based pricing (backwards compatibility)
                price = {
                    "input_per_million": 0.10,
                    "output_per_million": 0.10,
                    "cache_input_per_million": 0.0,
                }
        else:
            price_response = await PricingHandler.get_price(provider, model)
            price = price_response.dict() if price_response else await PricingHandler.get_avg_price_by_provider(provider)

        def r(x: float) -> float:
            return round(x, 10)

        # Use inference-time pricing for pureai if available
        if provider == "pureai" and "inference_cost_usd" in metrics_used:
            # Inference-time pricing (cost based on GPU time, not tokens)
            inference_cost = metrics_used.get("inference_cost_usd", 0.0)
            total_cost_usd = r(inference_cost)

            price_info = {
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "cache_input_cost_usd": 0.0,
                "total_cost_usd": total_cost_usd,
                "pricing_model": "inference_time",
                "cost_per_hour": metrics_used.get("cost_per_hour"),
                "inference_seconds": (metrics_used.get("latency_ms", 0) / 1000.0),
            }
        else:
            # Token-based pricing (API providers and fallback)
            cached_prompt_tokens = int(metrics_used.get("cached_prompt_tokens", 0))

            bd = PricingHandler.breakdown_usd(
                price={
                    "input_per_million": float(price.get("input_per_million", 0)),
                    "output_per_million": float(price.get("output_per_million", 0)),
                    "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                },
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_prompt_tokens
            )

            input_cost = r(bd["input_cost_usd"])
            cache_input_cost = r(bd["cache_input_cost_usd"])
            output_cost = r(bd["output_cost_usd"])
            total_cost_usd = r(bd["total_cost_usd"])

            price_info = {
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "cache_input_cost_usd": cache_input_cost,
                "total_cost_usd": total_cost_usd,
            }

        tenant_stats.Cost = Decimal(str(total_cost_usd))
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)

    req_id = f"cmpl_{uuid.uuid4().hex}"

    if chosen_adapter_name:
        background_tasks.add_task(
            publish_usage_event,
            request_id=req_id,
            tenant=tenant_id,
            logical_model=logical_model,
            provider=chosen_adapter_name,
            metrics=metrics_used,
            price_info=price_info,
        )

    latency_ms = float(metrics_used.get("latency_ms", 0)) if metrics_used else None
    ttft_ms = float(metrics_used.get("ttft_ms", 0)) if metrics_used else None

    return CompletionResponse(
        id=req_id,
        model=req.model,
        choices=[
            CompletionChoice(
                index=0,
                text=resp_text,
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cache_input_cost_usd=cache_input_cost,
            total_cost_usd=total_cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
        ),
    )
