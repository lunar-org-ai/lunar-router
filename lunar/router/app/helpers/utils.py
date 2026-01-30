from fastapi import HTTPException
from pathlib import Path
from typing import Dict, Optional, List, Any
import asyncio
import os
import json
import time
from ..database.PricingHandler import PricingHandler
from ..adapters import ProviderAdapter
from ..adapter_openai_litellm import OpenAILiteLLMAdapter
from ..adapter_anthropic import AnthropicAdapter
from ..adapter_deepseek import DeepSeekAdapter
from ..adapter_gemini import GeminiAdapter
from ..adapter_mistral import MistralAdapter
from ..adapter_perplexity import PerplexityAdapter
from ..adapter_groq import GroqAdapter
from ..adapter_cerebras import CerebrasAdapter
from ..adapter_cohere import CohereAdapter
from ..adapter_sambanova import SambaNovaAdapter
from ..adapter_bedrock import BedrockAdapter
from ..adapter_pureai import PureAIAdapter
from ..adapters import ProviderAdapter, MockProvider
from ..database.TenantStatsHandler import TenantStatsHandler
from ..database.PricingHandler import PricingHandler
from ..cache import get_adapters_cache
from decimal import Decimal
from litellm import token_counter
from dotenv import load_dotenv
import boto3
import json
import time
import time, yaml

BASE = Path(__file__).resolve().parents[2]
load_dotenv(BASE / ".env")  # optional


MODELS_DIR = BASE / "configs" / "models"
_adapter_cache: Dict[str, list] = {} # Simple cache: logical_model_name -> list[ProviderAdapter]

def load_model_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_adapters_for_model(model_cfg_path: str) -> List[ProviderAdapter]:
    cfg = load_model_config(model_cfg_path)
    model = cfg["model"]
    adapters: List[ProviderAdapter] = []
    for p in cfg.get("providers", []):
        if not p.get("enabled", True):
            continue
        typ = p["type"]; pid = p["id"]; params = p.get("params", {})
        if typ == "mock":
            adapters.append(
                MockProvider(
                    name=pid, model=model,
                    base_latency_ms=int(params.get("base_latency_ms", 300)),
                    jitter_ms=int(params.get("jitter_ms", 100)),
                    ttft_ms=int(params.get("ttft_ms", 120)),
                    error_rate=float(params.get("error_rate", 0.02)),
                )
            )
        elif typ == "openai_litellm":
            adapters.append(
                OpenAILiteLLMAdapter(
                    name=pid, logical_model=model,
                    model_name=str(params.get("model_name", model)),
                )
            )
        elif typ == "anthropic_litellm":
            adapters.append(
                AnthropicAdapter(
                    name=pid, logical_model=model,
                    model_name=f"anthropic/{params.get('model_name', model)}",
                )
            )
        elif typ == "deepseek_litellm":
            adapters.append(
                DeepSeekAdapter(
                    name=pid, logical_model=model,
                    model_name=f"deepseek/{params.get('model_name', model)}",
                )
            )
        elif typ == "gemini_litellm":
            adapters.append(
                GeminiAdapter(
                    name=pid, logical_model=model,
                    model_name=f"gemini/{params.get('model_name', model)}",
                )
            )
        elif typ == "mistral_litellm":
            adapters.append(
                MistralAdapter(
                    name=pid, logical_model=model,
                    model_name=f"mistral/{params.get('model_name', model)}",
                )
            )
        elif typ == "perplexity_litellm":
            adapters.append(
                PerplexityAdapter(
                    name=pid, logical_model=model,
                    model_name=f"perplexity/{params.get('model_name', model)}",
                )
            )
        elif typ == "groq_litellm":
            adapters.append(
                GroqAdapter(
                    name=pid, logical_model=model,
                    model_name=f"groq/{params.get('model_name', model)}",
                )
            )
        elif typ == "cerebras_litellm":
            adapters.append(
                CerebrasAdapter(
                    name=pid, logical_model=model,
                    model_name=f"cerebras/{params.get('model_name', model)}",
                )
            )
        elif typ == "cohere_litellm":
            adapters.append(
                CohereAdapter(
                    name=pid, logical_model=model,
                    model_name=f"cohere_chat/{params.get('model_name', model)}",
                )
            )
        elif typ == "sambanova_litellm":
            adapters.append(
                SambaNovaAdapter(
                    name=pid, logical_model=model,
                    model_name=str(params.get('model_name', model)),
                )
            )
        elif typ == "bedrock":
            adapters.append(
                BedrockAdapter(
                    name=pid, logical_model=model,
                    model_name=str(params.get("model_name", model)),
                    region=str(params.get("region", "us-east-1")),
                )
            )
    return adapters

def adapters_for_yaml(model_name: str):
    """Original YAML-based adapters_for method, renamed for clarity."""
    cfg_path = MODELS_DIR / f"{model_name}.yaml"
    if not cfg_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found (expected {cfg_path.name}).",
        )
    if model_name not in _adapter_cache:
        _adapter_cache[model_name] = build_adapters_for_model(str(cfg_path))
    return _adapter_cache[model_name]

async def adapters_for(
    model_name: str,
    forced_provider: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> List[ProviderAdapter]:
    """
    Get adapters for a model.

    When forced_provider is specified (e.g., openai/gpt-4):
    - pureai: creates PureAIAdapter with tenant_id for SageMaker lookup
    - openai: creates OpenAILiteLLMAdapter directly
    - bedrock: creates BedrockAdapter directly

    Without forced_provider: uses database, then falls back to YAML.
    """
    # Handle forced providers - create adapter directly without YAML lookup
    if forced_provider == "pureai":
        if not tenant_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for PureAI models",
            )
        return [PureAIAdapter(name="pureai", logical_model=model_name, tenant_id=tenant_id)]

    if forced_provider == "openai":
        # Direct OpenAI call - model_name is sent to OpenAI API
        return [OpenAILiteLLMAdapter(name="openai", logical_model=model_name, model_name=model_name)]

    if forced_provider == "bedrock":
        # Direct Bedrock call
        return [BedrockAdapter(name="bedrock", logical_model=model_name, model_name=model_name)]

    if forced_provider == "anthropic":
        # Anthropic via LiteLLM with dedicated adapter
        return [AnthropicAdapter(name="anthropic", logical_model=model_name, model_name=f"anthropic/{model_name}")]

    if forced_provider == "deepseek":
        # DeepSeek via LiteLLM with dedicated adapter
        return [DeepSeekAdapter(name="deepseek", logical_model=model_name, model_name=f"deepseek/{model_name}")]

    if forced_provider == "gemini":
        # Gemini via LiteLLM with dedicated adapter
        return [GeminiAdapter(name="gemini", logical_model=model_name, model_name=f"gemini/{model_name}")]

    if forced_provider == "mistral":
        # Mistral via LiteLLM with dedicated adapter
        return [MistralAdapter(name="mistral", logical_model=model_name, model_name=f"mistral/{model_name}")]

    if forced_provider == "perplexity":
        # Perplexity via LiteLLM with dedicated adapter
        return [PerplexityAdapter(name="perplexity", logical_model=model_name, model_name=f"perplexity/{model_name}")]

    if forced_provider == "groq":
        # Groq via LiteLLM with dedicated adapter
        return [GroqAdapter(name="groq", logical_model=model_name, model_name=f"groq/{model_name}")]

    if forced_provider == "cerebras":
        # Cerebras via LiteLLM with dedicated adapter
        return [CerebrasAdapter(name="cerebras", logical_model=model_name, model_name=f"cerebras/{model_name}")]

    if forced_provider == "cohere":
        # Cohere via LiteLLM with dedicated adapter
        return [CohereAdapter(name="cohere", logical_model=model_name, model_name=f"cohere_chat/{model_name}")]

    if forced_provider == "sambanova":
        # SambaNova via OpenAI-compatible API with dedicated adapter
        return [SambaNovaAdapter(name="sambanova", logical_model=model_name, model_name=model_name)]

    # No forced provider - use dynamic lookup (database then YAML)
    return await adapters_for_dynamic(model_name)


def _parse_model_string(model_str: str):
    """
    Parse model string:

      - "gpt-4o-mini"        -> (model="gpt-4o-mini", forced_provider=None)
      - "openai/gpt-4o-mini" -> (model="gpt-4o-mini", forced_provider="openai")
    """
    if "/" in model_str:
        provider_id, logical = model_str.split("/", 1)
        return logical, provider_id
    return model_str, None


# Billing / Usage Events - SQS (FIFO)
BILLING_QUEUE_URL = os.getenv("BILLING_QUEUE_URL", "").strip()
sqs_client = boto3.client("sqs") if BILLING_QUEUE_URL else None
def publish_usage_event(
    *,
    request_id: str,
    tenant: str,
    logical_model: str,
    provider: str,
    metrics: Dict[str, float],
    price_info: Optional[Dict[str, float]] = None,
) -> None:
    """
    Publishes usage/billing events in the SQS queue (FIFO).
    Runs via FastAPI BackgroundTasks (does not block responses).
    """
    if not sqs_client or not BILLING_QUEUE_URL:
        return

    try:
        event = {
            "version": "1",
            "request_id": request_id,
            "timestamp": time.time(),
            "tenant_id": tenant or "default",
            "logical_model": logical_model,
            "provider": provider,
            "metrics": {
                "ttft_ms": float(metrics.get("ttft_ms", 0.0)),
                "latency_ms": float(metrics.get("latency_ms", 0.0)),
                "tokens_in": int(metrics.get("tokens_in", 0)),
                "tokens_out": int(metrics.get("tokens_out", 0)),
                "error": float(metrics.get("error", 0.0)),
            },
            "pricing": {
                "input_cost_usd": float((price_info or {}).get("input_cost_usd", 0.0)),
                "output_cost_usd": float((price_info or {}).get("output_cost_usd", 0.0)),
                "cache_input_cost_usd": float((price_info or {}).get("cache_input_cost_usd", 0.0)),
                "total_cost_usd": float((price_info or {}).get("total_cost_usd", 0.0)),
            },
        }

        tenant_for_group = event["tenant_id"] or "global"

        sqs_client.send_message(
            QueueUrl=BILLING_QUEUE_URL,
            MessageBody=json.dumps(event),
            MessageGroupId=tenant_for_group,
            MessageDeduplicationId=request_id,
        )
    except Exception as e:
        print(f"[billing] Failed to enqueue usage event: {e}", flush=True)


async def adapters_for_dynamic(model_name: str) -> List[ProviderAdapter]:
    """
    Dynamically create adapters based on providers found in the PricingTable for the given model.
    Falls back to YAML configuration if no providers are found in the database.
    
    OPTIMIZATION: Uses in-memory adapters cache with 5-minute TTL to avoid repeated DB calls and adapter recreation.
    """
    # OPTIMIZATION: Check cache first
    cache = get_adapters_cache()
    cached_adapters = cache.get(model_name)
    if cached_adapters is not None:
        return cached_adapters
    
    try:
        # Get providers from the database
        providers = await PricingHandler.get_providers_by_model(model_name)
        
        if not providers:
            # Fallback to original YAML-based approach
            return adapters_for_yaml(model_name)
        
        adapters: List[ProviderAdapter] = []
        
        for provider_info in providers:
            provider = provider_info["provider"]
            modelid = provider_info["modelid"]
            print(f"Provider from DB for model {model_name}: {provider} (modelid: {modelid})", flush=True)

            if provider == "openai":
                # Create OpenAI LiteLLM adapter for OpenAI provider
                adapter = OpenAILiteLLMAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=model_name,
                )
                adapters.append(adapter)
            elif provider == "anthropic":
                # Create Anthropic adapter for Anthropic provider
                adapter = AnthropicAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"anthropic/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "deepseek":
                # Create DeepSeek adapter for DeepSeek provider
                adapter = DeepSeekAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"deepseek/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "gemini":
                # Create Gemini adapter for Gemini provider
                adapter = GeminiAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"gemini/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "mistral":
                # Create Mistral adapter for Mistral provider
                adapter = MistralAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"mistral/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "perplexity":
                # Create Perplexity adapter for Perplexity provider
                adapter = PerplexityAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"perplexity/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "groq":
                # Create Groq adapter for Groq provider
                adapter = GroqAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"groq/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "cerebras":
                # Create Cerebras adapter for Cerebras provider
                adapter = CerebrasAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"cerebras/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "cohere":
                # Create Cohere adapter for Cohere provider
                adapter = CohereAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=f"cohere_chat/{modelid or model_name}",
                )
                adapters.append(adapter)
            elif provider == "sambanova":
                # Create SambaNova adapter for SambaNova provider
                adapter = SambaNovaAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=modelid or model_name,
                )
                adapters.append(adapter)
            elif provider == "bedrock":
                # Create Bedrock adapter for Bedrock provider
                adapter = BedrockAdapter(
                    name=provider,
                    logical_model=model_name,
                    model_name=model_name,
                    modelid=modelid)
                adapters.append(adapter)
        
        # OPTIMIZATION: Cache the adapters asynchronously (non-blocking)
        asyncio.create_task(cache.set_async(model_name, adapters))
        return adapters
        
    except Exception as e:
        print(f"[adapters_for_dynamic] Error getting providers from database: {e}, falling back to YAML", flush=True)
        # Fallback to original YAML-based approach
        return adapters_for_yaml(model_name)
    
async def generate_stream_response(stream, req_id: str, req_model: str, tenant_id: str, logical_model: str, chosen_adapter_name: str, start_time: float, tenant_stats=None, background_tasks=None, metrics=None, is_chat=True):
    """Generate streaming response for chat or text completion
    
    Args:
        is_chat: True for /v1/chat/completions, False for /v1/completions
    """
    first_chunk = True
    ttft_ms = None
    collected_text = ""
    final_latency_ms = 0.0
    
    try:
        async for chunk in stream:
            try:
                if first_chunk:
                    first_chunk = False
                    ttft_ms = (time.time() - start_time) * 1000.0
                    
                chunk_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk" if is_chat else "text_completion.chunk",
                    "model": req_model,
                    "choices": []
                }
                
                delta_content = ""
                # Handle both dict (vLLM, raw JSON) and object (litellm) formats
                if isinstance(chunk, dict):
                    # Dict format from vLLM or other providers
                    try:
                        if is_chat:
                            # Chat format: choices[0].delta.content
                            delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                        else:
                            # Completion format: choices[0].text
                            delta_content = chunk.get("choices", [{}])[0].get("text", "") or ""
                    except Exception as e:
                        print(f"[stream_response] Error extracting dict content (is_chat={is_chat}): {e}, chunk: {chunk}", flush=True)
                        delta_content = ""
                else:
                    # Object format from litellm
                    try:
                        if is_chat:
                            delta_content = chunk.choices[0].delta.get("content", "") or ""
                        else:
                            delta_content = chunk.choices[0].text or ""
                    except Exception:
                        try:
                            delta_content = getattr(
                                getattr(chunk.choices[0], "delta", None),
                                "content",
                                "",
                            ) or ""
                        except Exception:
                            delta_content = ""
                
                # Only process and yield chunks with actual content
                if delta_content:
                    collected_text += delta_content
                    if is_chat:
                        chunk_data["choices"] = [{
                            "index": 0,
                            "delta": {
                                "content": delta_content
                            },
                            "finish_reason": None
                        }]
                    else:
                        chunk_data["choices"] = [{
                            "index": 0,
                            "text": delta_content,
                            "finish_reason": None
                        }]
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            except Exception as e:
                print(f"[stream_response] Error processing chunk: {e}", flush=True)
                error_chunk = {
                    "id": req_id,
                    "object": "error",
                    "error": {
                        "message": f"Error processing chunk: {str(e)}",
                        "type": "error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        
        final_latency_ms = (time.time() - start_time) * 1000.0
        
        # Send final chunk with finish_reason
        final_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "model": req_model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"[stream_response] Fatal error: {e}", flush=True)
        error_chunk = {
            "id": req_id,
            "object": "error",
            "error": {
                "message": str(e),
                "type": "error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    # Update tenant stats asynchronously after streaming ends
    if tenant_stats and background_tasks:
        try:
            # Calculate tokens for the collected text
            try:
                tokens_out = token_counter(model=req_model, text=collected_text) or 0
            except Exception:
                # Fallback token estimation
                tokens_out = max(5, int(len(collected_text.split()) * 1.3))
            
            # Get tokens_in from metrics if available, otherwise keep current
            tokens_in = int(metrics.get("tokens_in", 0)) if metrics else 0
            
            # Calculate cost breakdown for streaming response
            if chosen_adapter_name:
                provider = chosen_adapter_name
                model = logical_model

                def r(x: float) -> float:
                    return round(x, 10)

                # PureAI uses inference-time pricing for self-hosted models
                if provider == "pureai":
                    # Check if metrics contain cost info (from adapter)
                    cost_per_hour = metrics.get("cost_per_hour") if metrics else None
                    instance_type = metrics.get("instance_type") if metrics else None
                    deployment_id = metrics.get("deployment_id") if metrics else None

                    if cost_per_hour is not None:
                        # Inference-time pricing (cost based on GPU time)
                        inference_cost = DeploymentHandler.calculate_request_cost(
                            latency_ms=final_latency_ms,
                            cost_per_hour=cost_per_hour,
                            instance_type=instance_type
                        )
                        total_cost_usd = r(inference_cost)

                        # Update deployment inference time
                        if deployment_id:
                            import asyncio
                            asyncio.create_task(
                                DeploymentHandler.update_inference_time(
                                    deployment_id=deployment_id,
                                    inference_seconds=final_latency_ms / 1000.0
                                )
                            )

                        price_info = {
                            "input_cost_usd": 0.0,
                            "output_cost_usd": 0.0,
                            "cache_input_cost_usd": 0.0,
                            "total_cost_usd": total_cost_usd,
                            "pricing_model": "inference_time",
                            "cost_per_hour": cost_per_hour,
                            "inference_seconds": final_latency_ms / 1000.0,
                        }
                        tenant_stats.Cost = Decimal(str(total_cost_usd))
                    else:
                        # Fallback to token-based pricing (backwards compatibility)
                        price = {
                            "input_per_million": 0.10,
                            "output_per_million": 0.10,
                            "cache_input_per_million": 0.0,
                        }
                        cached_prompt_tokens = int(metrics.get("cached_prompt_tokens", 0)) if metrics else 0

                        bd = PricingHandler.breakdown_usd(
                            price={
                                "input_per_million": float(price.get("input_per_million", 0)),
                                "output_per_million": float(price.get("output_per_million", 0)),
                                "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                            },
                            prompt_tokens=tokens_in,
                            completion_tokens=tokens_out,
                            cached_prompt_tokens=cached_prompt_tokens)

                        total_cost_usd = r(bd["total_cost_usd"])
                        tenant_stats.Cost = Decimal(str(total_cost_usd))

                        price_info = {
                            "input_cost_usd": r(bd["input_cost_usd"]),
                            "output_cost_usd": r(bd["output_cost_usd"]),
                            "cache_input_cost_usd": r(bd["cache_input_cost_usd"]),
                            "total_cost_usd": total_cost_usd,
                        }
                else:
                    try:
                        price_response = await PricingHandler.get_price(provider, model)
                        price = price_response.dict() if price_response else await PricingHandler.get_avg_price_by_provider(provider)
                    except Exception:
                        price = {"input_per_million": 0, "output_per_million": 0, "cache_input_per_million": 0}

                    cached_prompt_tokens = int(metrics.get("cached_prompt_tokens", 0)) if metrics else 0

                    bd = PricingHandler.breakdown_usd(
                        price={
                            "input_per_million": float(price.get("input_per_million", 0)),
                            "output_per_million": float(price.get("output_per_million", 0)),
                            "cache_input_per_million": float(price.get("cache_input_per_million", 0)),
                        },
                        prompt_tokens=tokens_in,
                        completion_tokens=tokens_out,
                        cached_prompt_tokens=cached_prompt_tokens)

                    total_cost_usd = r(bd["total_cost_usd"])
                    tenant_stats.Cost = Decimal(str(total_cost_usd))

                    # Prepare price_info for usage event
                    input_cost = r(bd["input_cost_usd"])
                    cache_input_cost = r(bd["cache_input_cost_usd"])
                    output_cost = r(bd["output_cost_usd"])

                    price_info = {
                        "input_cost_usd": input_cost,
                        "output_cost_usd": output_cost,
                        "cache_input_cost_usd": cache_input_cost,
                        "total_cost_usd": total_cost_usd,
                    }

                # Publish usage event for streaming response
                streaming_metrics = {
                    "ttft_ms": float(ttft_ms or 0.0),
                    "latency_ms": float(final_latency_ms),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "error": 0.0,
                }
                
                background_tasks.add_task(
                    publish_usage_event,
                    request_id=req_id,
                    tenant=tenant_id,
                    logical_model=logical_model,
                    provider=chosen_adapter_name,
                    metrics=streaming_metrics,
                    price_info=price_info,
                )

            # Update tenant_stats with correct streaming metrics
            tenant_stats.TTFT = Decimal(str(ttft_ms or 0.0))
            tenant_stats.Latency = Decimal(str(final_latency_ms))
            tenant_stats.OutputText = collected_text
            tenant_stats.TotalTokens = tokens_in + tokens_out
            tenant_stats.Success = True
            
            background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
        
        except Exception as e:
            error_chunk = {
                "id": req_id,
                "object": "error",
                "error": {
                    "message": str(e),
                    "type": "error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
