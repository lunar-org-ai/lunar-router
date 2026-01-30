"""
Local JSON-based Pricing Handler for development/open-source use.
Replaces DynamoDB with local JSON file storage.
"""
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from statistics import mean

from ..models import PricingItem, PricingResponse
from ...cache import get_pricing_cache


# Default data directory
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PRICING_FILE = DATA_DIR / "pricing.json"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PRICING_FILE.exists():
        PRICING_FILE.write_text("[]")


def _load_pricing_data() -> List[dict]:
    _ensure_data_dir()
    try:
        return json.loads(PRICING_FILE.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_pricing_data(data: List[dict]):
    _ensure_data_dir()
    PRICING_FILE.write_text(json.dumps(data, indent=2))


class LocalPricingHandler:
    """Local JSON-based pricing handler that mimics DynamoDB PricingHandler."""

    @staticmethod
    async def upsert_price(
        provider: str,
        model: str,
        input_per_million: float,
        output_per_million: float,
        cache_input_per_million: float,
        modelid: Optional[str] = None
    ) -> PricingItem:
        """Insert or update a price for (provider, model)."""
        data = _load_pricing_data()
        updated_day = datetime.utcnow().strftime("%Y-%m-%d")

        # Find existing item
        existing_idx = None
        for i, item in enumerate(data):
            if item["Provider"] == provider and item["Model"] == model:
                existing_idx = i
                break

        new_item = {
            "Provider": provider,
            "Model": model,
            "ModelId": modelid,
            "UpdatedAt": updated_day,
            "input_per_million": input_per_million,
            "output_per_million": output_per_million,
            "cache_input_per_million": cache_input_per_million
        }

        if existing_idx is not None:
            data[existing_idx] = new_item
        else:
            data.append(new_item)

        _save_pricing_data(data)

        # Invalidate cache
        cache = get_pricing_cache()
        cache.invalidate(provider, model)

        return PricingItem(**new_item)

    @staticmethod
    async def get_by_provider(provider: str) -> List[PricingItem]:
        """Given a provider, return all models with their prices."""
        data = _load_pricing_data()
        items = [item for item in data if item["Provider"] == provider]
        return [PricingItem(**i) for i in items]

    @staticmethod
    async def get_avg_price_by_provider(provider: str) -> Optional[dict]:
        """Get average price for a provider across all their models."""
        cache = get_pricing_cache()
        cached = cache.get_avg_provider(provider)
        if cached is not None:
            return cached

        items = await LocalPricingHandler.get_by_provider(provider)
        if not items:
            result = {
                "input_per_million": 0.0,
                "output_per_million": 0.0,
                "cache_input_per_million": 0.0,
            }
            cache.set_avg_provider(provider, result)
            return result

        avg_input = mean(float(i.input_per_million) for i in items if i.input_per_million is not None)
        avg_output = mean(float(i.output_per_million) for i in items if i.output_per_million is not None)
        avg_cache_input = mean(float(i.cache_input_per_million) for i in items if i.cache_input_per_million is not None)

        result = {
            "input_per_million": avg_input,
            "output_per_million": avg_output,
            "cache_input_per_million": avg_cache_input,
        }
        cache.set_avg_provider(provider, result)
        return result

    @staticmethod
    async def get_by_model(model: str) -> List[PricingItem]:
        """Given a model, return all providers with their prices."""
        data = _load_pricing_data()
        items = [item for item in data if item["Model"] == model]
        return [PricingItem(**i) for i in items]

    @staticmethod
    async def get_price(provider: str, model: str) -> Optional[PricingResponse]:
        """Given a provider and model, return the price."""
        cache = get_pricing_cache()
        cached = cache.get(provider, model)
        if cached is not None:
            return PricingResponse(**cached)

        data = _load_pricing_data()
        for item in data:
            if item["Provider"] == provider and item["Model"] == model:
                pricing_response = PricingResponse(
                    ModelId=item.get("ModelId"),
                    UpdatedAt=item.get("UpdatedAt"),
                    input_per_million=item.get("input_per_million", 0.20),
                    output_per_million=item.get("output_per_million", 0.40),
                    cache_input_per_million=item.get("cache_input_per_million", 0.00)
                )
                asyncio.create_task(cache.set_async(provider, model, pricing_response.dict()))
                return pricing_response
        return None

    @staticmethod
    def breakdown_usd(
        price: Dict[str, float],
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,
    ) -> Dict[str, float]:
        """Calculate cost breakdown."""
        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        cached_prompt_tokens = max(0, int(cached_prompt_tokens))

        if cached_prompt_tokens > prompt_tokens:
            cached_prompt_tokens = prompt_tokens

        normal_prompt_tokens = prompt_tokens - cached_prompt_tokens

        input_cost = (normal_prompt_tokens / 1_000_000.0) * price["input_per_million"]
        cache_input_cost = (cached_prompt_tokens / 1_000_000.0) * price["cache_input_per_million"]
        output_cost = (completion_tokens / 1_000_000.0) * price["output_per_million"]
        total_cost = input_cost + cache_input_cost + output_cost

        return {
            "input_cost_usd": input_cost,
            "cache_input_cost_usd": cache_input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
        }

    @staticmethod
    def cost_usd(price: Dict[str, float], prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0) -> float:
        bd = LocalPricingHandler.breakdown_usd(price, prompt_tokens, completion_tokens, cached_prompt_tokens)
        return bd["total_cost_usd"]

    @staticmethod
    def expected_cost_per_1k(price: Dict[str, float], exp_ti: int, exp_to: int, cached_prompt_tokens: int = 0) -> float:
        exp_ti = max(0, int(exp_ti))
        exp_to = max(0, int(exp_to))
        cached_prompt_tokens = max(0, int(cached_prompt_tokens))

        total = exp_ti + exp_to
        if total <= 0:
            return 0.0

        cost = LocalPricingHandler.cost_usd(price, exp_ti, exp_to, cached_prompt_tokens)
        return cost * (1000.0 / float(total))

    @staticmethod
    async def get_providers_by_model(model: str) -> List[Dict[str, Optional[str]]]:
        """Given a model name, return all providers that support this model."""
        data = _load_pricing_data()
        providers = []
        for item in data:
            if item["Model"] == model:
                providers.append({
                    "provider": item.get("Provider"),
                    "modelid": item.get("ModelId")
                })
        return providers

    @staticmethod
    async def get_all_models() -> List[Dict[str, Any]]:
        """Get all unique models from the pricing data with their providers."""
        data = _load_pricing_data()

        models_map: Dict[str, Dict[str, Any]] = {}
        for item in data:
            model_name = item.get("Model", "")
            provider = item.get("Provider", "")
            model_id = item.get("ModelId")
            input_per_million = item.get("input_per_million", 0)
            output_per_million = item.get("output_per_million", 0)
            cache_input_per_million = item.get("cache_input_per_million", 0)

            if not model_name:
                continue

            if model_name not in models_map:
                models_map[model_name] = {
                    "id": model_name,
                    "owned_by": provider,
                    "providers": [],
                    "input_per_million": input_per_million,
                    "output_per_million": output_per_million,
                    "cache_input_per_million": cache_input_per_million,
                }

            models_map[model_name]["providers"].append({
                "id": provider,
                "model_id": model_id,
                "input_per_million": input_per_million,
                "output_per_million": output_per_million,
                "cache_input_per_million": cache_input_per_million,
            })

        return list(models_map.values())
