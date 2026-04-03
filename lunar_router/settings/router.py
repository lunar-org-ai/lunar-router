"""
Settings + Models API Router

Mounts settings under /v1/evaluations/settings and models under /v1/models.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from . import repository as repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])


def _tenant(request: Request) -> str:
    return request.headers.get("x-tenant-id", "default")


# ── Settings ────────────────────────────────────────────────────

@router.get("/v1/evaluations/settings")
async def get_settings(request: Request):
    tenant = _tenant(request)
    return repo.get(tenant)


@router.put("/v1/evaluations/settings")
async def update_settings(request: Request):
    tenant = _tenant(request)
    body = await request.json()

    allowed_fields = ["default_judge_model", "default_temperature", "max_parallel_requests", "python_script_timeout", "config"]
    updates = {}
    for field in allowed_fields:
        if field not in body:
            continue
        value = body[field]
        if field == "default_temperature":
            if not isinstance(value, (int, float)) or value < 0 or value > 2:
                raise HTTPException(400, "default_temperature must be between 0 and 2")
        if field == "max_parallel_requests":
            if not isinstance(value, int) or value < 1 or value > 20:
                raise HTTPException(400, "max_parallel_requests must be between 1 and 20")
        if field == "python_script_timeout":
            if not isinstance(value, int) or value < 1 or value > 300:
                raise HTTPException(400, "python_script_timeout must be between 1 and 300 seconds")
        updates[field] = value

    if not updates:
        raise HTTPException(400, "No valid fields to update")
    return repo.update(tenant, updates)


# ── Models available ────────────────────────────────────────────

@router.get("/v1/models/available")
async def list_available_models(request: Request):
    """List models available for evaluation.

    Queries Go engine for configured providers, then maps each provider
    to its known models from ``model_prices.MODEL_INFO``.
    """
    try:
        import httpx
        from ..evals_common.model_invoker import ModelInvoker
        from ..model_prices import MODEL_INFO

        PROVIDER_PREFIXES: dict[str, list[str]] = {
            "openai": ["gpt-", "o1", "o3", "o4"],
            "anthropic": ["claude-"],
            "google": ["gemini-"],
            "mistral": ["mistral-", "ministral-", "codestral-", "pixtral-", "open-mistral-", "open-mixtral-"],
            "deepseek": ["deepseek-"],
            "groq": ["llama-", "mixtral-8x7b-32768", "gemma2-"],
            "perplexity": ["sonar"],
            "cerebras": ["llama3.1-"],
            "sambanova": ["Meta-Llama-"],
            "together": ["meta-llama/", "mistralai/", "Qwen/"],
            "fireworks": ["accounts/fireworks/"],
            "cohere": ["command", "c4ai-"],
        }

        invoker = ModelInvoker()
        url = f"{invoker.base_url}/v1/config/keys"
        resp = httpx.get(url, timeout=5)
        configured: list[str] = []
        if resp.status_code == 200:
            configured = resp.json().get("configured_providers", [])

        if not configured:
            configured = ["openai"]

        models: list[dict] = []
        grouped: dict[str, list] = {}
        for provider in configured:
            prefixes = PROVIDER_PREFIXES.get(provider, [])
            for model_key in MODEL_INFO:
                if any(model_key.startswith(p) for p in prefixes):
                    entry = {
                        "id": f"{provider}/{model_key}",
                        "name": model_key,
                        "provider": provider,
                        "type": "external",
                        "available": True,
                    }
                    models.append(entry)
                    grouped.setdefault(provider, []).append(entry)

        return {"models": models, "count": len(models), "by_provider": grouped}
    except Exception as e:
        logger.exception("Error fetching available models")
        raise HTTPException(500, f"Failed to fetch models: {e}")
