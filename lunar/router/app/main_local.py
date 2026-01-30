"""
Lunar Router - Local Development Version
Open-source LLM routing with semantic routing (UniRoute)

This is the local development version that uses:
- JSON files instead of DynamoDB
- Simple API key auth instead of AWS Cognito
- No AWS dependencies required
"""
import json
import time
import uuid
import os
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Use local auth middleware instead of AWS-based
from .middleware.local_auth import LocalAuthMiddleware

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
from .router_local import HealthFirstPlanner
from .database.models import TenantStatsModel, ProviderAttempt

# Use local handlers instead of DynamoDB
from .database.local import LocalPricingHandler as PricingHandler
from .database.local import LocalStatsHandler as TenantStatsHandler

from datetime import datetime
from decimal import Decimal

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")

MODELS_DIR = BASE / "configs" / "models"

app = FastAPI(
    title="Lunar Router - Open Source LLM Routing",
    description="Health-first LLM routing with semantic routing capabilities",
    version="1.0.0",
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv("LUNAR_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

app.add_middleware(LocalAuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import and include routes (local versions)
from .routes import pricing_local as pricing_routes
from .routes import semantic_router_local as semantic_router

app.include_router(pricing_routes.router)
app.include_router(semantic_router.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "lunar-router"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Lunar Router",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


planner = HealthFirstPlanner()


def _parse_model_string(model_str: str):
    """Parse model string: 'openai/gpt-4' -> ('gpt-4', 'openai')"""
    if "/" in model_str:
        provider_id, logical = model_str.split("/", 1)
        return logical, provider_id
    return model_str, None


async def adapters_for_local(
    model_name: str,
    forced_provider: Optional[str] = None,
) -> List:
    """
    Get adapters for a model using local pricing data.
    """
    from .adapter_openai_litellm import OpenAILiteLLMAdapter
    from .adapter_anthropic import AnthropicAdapter
    from .adapter_deepseek import DeepSeekAdapter
    from .adapter_gemini import GeminiAdapter
    from .adapter_mistral import MistralAdapter
    from .adapter_groq import GroqAdapter
    from .adapter_cohere import CohereAdapter

    # Handle forced providers
    if forced_provider == "openai":
        return [OpenAILiteLLMAdapter(name="openai", logical_model=model_name, model_name=model_name)]

    if forced_provider == "anthropic":
        return [AnthropicAdapter(name="anthropic", logical_model=model_name, model_name=f"anthropic/{model_name}")]

    if forced_provider == "deepseek":
        return [DeepSeekAdapter(name="deepseek", logical_model=model_name, model_name=f"deepseek/{model_name}")]

    if forced_provider == "gemini":
        return [GeminiAdapter(name="gemini", logical_model=model_name, model_name=f"gemini/{model_name}")]

    if forced_provider == "mistral":
        return [MistralAdapter(name="mistral", logical_model=model_name, model_name=f"mistral/{model_name}")]

    if forced_provider == "groq":
        return [GroqAdapter(name="groq", logical_model=model_name, model_name=f"groq/{model_name}")]

    if forced_provider == "cohere":
        return [CohereAdapter(name="cohere", logical_model=model_name, model_name=f"cohere_chat/{model_name}")]

    # Get providers from local pricing data
    providers = await PricingHandler.get_providers_by_model(model_name)

    if not providers:
        # Default to OpenAI if no providers configured
        return [OpenAILiteLLMAdapter(name="openai", logical_model=model_name, model_name=model_name)]

    adapters = []
    for provider_info in providers:
        provider = provider_info["provider"]
        modelid = provider_info.get("modelid")

        if provider == "openai":
            adapters.append(OpenAILiteLLMAdapter(name=provider, logical_model=model_name, model_name=model_name))
        elif provider == "anthropic":
            adapters.append(AnthropicAdapter(name=provider, logical_model=model_name, model_name=f"anthropic/{modelid or model_name}"))
        elif provider == "deepseek":
            adapters.append(DeepSeekAdapter(name=provider, logical_model=model_name, model_name=f"deepseek/{modelid or model_name}"))
        elif provider == "gemini":
            adapters.append(GeminiAdapter(name=provider, logical_model=model_name, model_name=f"gemini/{modelid or model_name}"))
        elif provider == "mistral":
            adapters.append(MistralAdapter(name=provider, logical_model=model_name, model_name=f"mistral/{modelid or model_name}"))
        elif provider == "groq":
            adapters.append(GroqAdapter(name=provider, logical_model=model_name, model_name=f"groq/{modelid or model_name}"))
        elif provider == "cohere":
            adapters.append(CohereAdapter(name=provider, logical_model=model_name, model_name=f"cohere_chat/{modelid or model_name}"))

    return adapters if adapters else [OpenAILiteLLMAdapter(name="openai", logical_model=model_name, model_name=model_name)]


@app.get("/v1/providers", response_model=List[ProviderInfo])
async def providers(
    model: str = Query(None, description="Logical model name to list providers for"),
):
    """List providers for a model."""
    target_model = model or "gpt-4o-mini"

    providers_list = await PricingHandler.get_providers_by_model(target_model)

    out: List[ProviderInfo] = []
    for p in providers_list:
        out.append(
            ProviderInfo(
                id=p["provider"],
                type=p["provider"],
                enabled=True,
                params={},
            )
        )
    return out


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """OpenAI-compatible chat completions endpoint with routing."""
    logical_model, forced_provider = _parse_model_string(req.model)
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    adapters = await adapters_for_local(logical_model, forced_provider=forced_provider)

    if forced_provider:
        candidate = next((a for a in adapters if a.name == forced_provider), None)
        if not candidate:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{forced_provider}' not found for model '{logical_model}'",
            )
        ordered = [candidate]
    else:
        ordered = await planner.rank(logical_model, adapters) if len(adapters) > 1 else adapters

    last_error = None
    chosen_adapter_name: Optional[str] = None
    resp_text: Optional[str] = None
    metrics_used: Dict[str, float] = {}
    provider_attempts: List[ProviderAttempt] = []

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
            InputText="".join([m.content for m in req.messages if m.content]),
            OutputText=resp.get("text", ""),
            TotalTokens=int(metrics.get("tokens_in", 0)) + int(metrics.get("tokens_out", 0)),
            ErrorCategory=resp.get("error_category") if "error" in resp else None,
            ErrorMessage=resp.get("error") if "error" in resp else None,
            ProviderAttempts=json.dumps([a.model_dump() for a in provider_attempts]),
            FallbackCount=len(provider_attempts),
            FinalProvider=chosen.name
        )

        if "error" not in resp:
            chosen_adapter_name = chosen.name
            metrics_used = metrics

            if req.stream and "stream" in resp:
                req_id = f"chatcmpl_{uuid.uuid4().hex}"
                start_time = time.time()

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
                                "model": req.model,
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
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(),
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

    # Calculate cost
    prompt_tokens = int(metrics_used.get("tokens_in", 0))
    completion_tokens = int(metrics_used.get("tokens_out", 0))
    total_tokens = prompt_tokens + completion_tokens

    input_cost = output_cost = cache_input_cost = total_cost_usd = None

    if chosen_adapter_name:
        price_response = await PricingHandler.get_price(chosen_adapter_name, logical_model)
        price = price_response.dict() if price_response else await PricingHandler.get_avg_price_by_provider(chosen_adapter_name)

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

            input_cost = round(bd["input_cost_usd"], 10)
            output_cost = round(bd["output_cost_usd"], 10)
            cache_input_cost = round(bd["cache_input_cost_usd"], 10)
            total_cost_usd = round(bd["total_cost_usd"], 10)

            tenant_stats.Cost = Decimal(str(total_cost_usd))

        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)

    req_id = f"chatcmpl_{uuid.uuid4().hex}"

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
            latency_ms=float(metrics_used.get("latency_ms", 0)) if metrics_used else None,
            ttft_ms=float(metrics_used.get("ttft_ms", 0)) if metrics_used else None,
        ),
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    req: CompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """OpenAI-compatible text completions endpoint (legacy)."""
    logical_model, forced_provider = _parse_model_string(req.model)
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    adapters = await adapters_for_local(logical_model, forced_provider=forced_provider)

    if forced_provider:
        candidate = next((a for a in adapters if a.name == forced_provider), None)
        if not candidate:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{forced_provider}' not found for model '{logical_model}'",
            )
        ordered = [candidate]
    else:
        ordered = await planner.rank(logical_model, adapters) if len(adapters) > 1 else adapters

    last_error = None
    chosen_adapter_name: Optional[str] = None
    resp_text: Optional[str] = None
    metrics_used: Dict[str, float] = {}

    for chosen in ordered:
        resp, metrics = await chosen.send(
            {
                "tenant": tenant_id,
                "model": logical_model,
                "prompt": req.prompt,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stop": req.stop,
            }
        )

        if "error" not in resp:
            chosen_adapter_name = chosen.name
            metrics_used = metrics
            resp_text = resp.get("text", "")
            break

        last_error = resp.get("error")

    if resp_text is None:
        raise HTTPException(
            status_code=502,
            detail=f"All providers failed for model '{logical_model}': {last_error}",
        )

    prompt_tokens = int(metrics_used.get("tokens_in", 0))
    completion_tokens = int(metrics_used.get("tokens_out", 0))
    total_tokens = prompt_tokens + completion_tokens

    req_id = f"cmpl_{uuid.uuid4().hex}"

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
        ),
    )
