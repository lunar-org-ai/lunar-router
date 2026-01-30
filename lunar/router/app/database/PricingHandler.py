from datetime import datetime
import asyncio
import aioboto3
from boto3.dynamodb.conditions import Key, Attr
from boto3.dynamodb.types import TypeDeserializer
from typing import List, Optional, Dict, Any
from statistics import mean
from decimal import Decimal
from .models import PricingItem, PricingResponse
from typing import Dict
from ..cache import get_pricing_cache


class PricingHandler:
    region: str = "us-east-1"
    table_name: str = "PricingTable"
    deserializer = TypeDeserializer()

    def deserialize_item(item: dict) -> dict:
        return {k: PricingHandler.deserializer.deserialize(v) for k, v in item.items()}

    @staticmethod
    async def upsert_price(
        provider: str,
        model: str,
        input_per_million: float,
        output_per_million: float,
        cache_input_per_million: float,
        modelid: Optional[str] = None) -> PricingItem:
        """
        Insert or update a price for (provider, model).
        Returns the updated PricingItem.
        """
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=PricingHandler.region) as client:
            updated_day = datetime.utcnow().strftime("%Y-%m-%d")

            item = PricingItem(
                Provider=provider,
                Model=model,
                ModelId=modelid,
                UpdatedAt=updated_day,
                input_per_million=input_per_million,
                output_per_million=output_per_million,
                cache_input_per_million=cache_input_per_million
            )
            
            # Prepare item for DynamoDB
            dynamo_item = {
                "Provider": {"S": provider},
                "Model": {"S": model},
                "UpdatedAt": {"S": updated_day},
                "input_per_million": {"N": str(input_per_million)},
                "output_per_million": {"N": str(output_per_million)},
                "cache_input_per_million": {"N": str(cache_input_per_million)}
            }
            
            if modelid:
                dynamo_item["ModelId"] = {"S": modelid}

            await client.put_item(
                TableName=PricingHandler.table_name,
                Item=dynamo_item
            )
            return item

    @staticmethod
    async def get_by_provider(provider: str) -> List[PricingItem]:
        """
        Given a provider, return all models with their prices and update dates.
        """
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=PricingHandler.region) as client:
            response = await client.query(
                TableName=PricingHandler.table_name,
                KeyConditionExpression="Provider = :provider",
                ExpressionAttributeValues={
                    ":provider": {"S": provider}
                },
            )
            raw_items = response.get("Items", [])
            items = [PricingHandler.deserialize_item(i) for i in raw_items]

            return [PricingItem(**i) for i in items]
        
    @staticmethod
    async def get_avg_price_by_provider(provider: str) -> Optional[dict]:
        """
        Get average price for a provider across all their models.
        Returns keys compatible with breakdown_usd() function.
        """
        # Check cache first
        cache = get_pricing_cache()
        cached = cache.get_avg_provider(provider)
        if cached is not None:
            return cached
        
        items = await PricingHandler.get_by_provider(provider)
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
        
        # Cache the result asynchronously (non-blocking)
        asyncio.create_task(cache.set_avg_provider_async(provider, result))
        return result

    @staticmethod
    async def get_by_model(model: str) -> List[PricingItem]:
        """
        Given a model, return all providers with their prices and update dates.
        """
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=PricingHandler.region) as client:
            response = await client.scan(
                TableName=PricingHandler.table_name,
                FilterExpression="Model = :model",
                ExpressionAttributeValues={
                    ":model": {"S": model}
                },
            )
            raw_items = response.get("Items", [])
            items = [PricingHandler.deserialize_item(i) for i in raw_items]

            return [PricingItem(**i) for i in items]

    @staticmethod
    async def get_price(provider: str, model: str) -> Optional[PricingResponse]:
        """
        Given a provider and model, return the price and update date.
        
        OPTIMIZATION: Uses 5-minute cache to avoid repeated database queries.
        """
        # Check cache first
        cache = get_pricing_cache()
        cached = cache.get(provider, model)
        if cached is not None:
            return PricingResponse(**cached)
        
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=PricingHandler.region) as client:
            response = await client.get_item(
                TableName=PricingHandler.table_name,
                Key={
                    "Provider": {"S": provider},
                    "Model": {"S": model}
                }
            )
            raw_item = response.get("Item")
            if not raw_item:
                return None

            item = PricingHandler.deserialize_item(raw_item)

            pricing_response = PricingResponse(
                ModelId=item.get("ModelId"),
                UpdatedAt=item.get("UpdatedAt"),
                input_per_million=item.get("input_per_million", 0.20),
                output_per_million=item.get("output_per_million", 0.40),
                cache_input_per_million=item.get("cache_input_per_million", 0.00)
            )
            
            # Cache the result asynchronously (non-blocking)
            asyncio.create_task(cache.set_async(provider, model, pricing_response.dict()))
            return pricing_response

    @staticmethod
    def breakdown_usd(price: Dict[str, float], prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0,
    ) -> Dict[str, float]:
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
        bd = PricingHandler.breakdown_usd(
            price, prompt_tokens, completion_tokens, cached_prompt_tokens
        )
        return bd["total_cost_usd"]

    @staticmethod
    def expected_cost_per_1k(price: Dict[str, float], exp_ti: int, exp_to: int, cached_prompt_tokens: int = 0) -> float:
        exp_ti = max(0, int(exp_ti))
        exp_to = max(0, int(exp_to))
        cached_prompt_tokens = max(0, int(cached_prompt_tokens))

        total = exp_ti + exp_to
        if total <= 0:
            return 0.0

        cost = PricingHandler.cost_usd(price, exp_ti, exp_to, cached_prompt_tokens)
        return cost * (1000.0 / float(total))

    @staticmethod
    async def get_providers_by_model(model: str) -> List[Dict[str, Optional[str]]]:
        """
        Given a model name, return all providers that support this model.
        Returns a list of dicts with provider and modelid.
        """
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=PricingHandler.region) as client:
            response = await client.scan(
                TableName=PricingHandler.table_name,
                FilterExpression="Model = :model",
                ExpressionAttributeValues={
                    ":model": {"S": model}
                },
            )
            raw_items = response.get("Items", [])

            items = [PricingHandler.deserialize_item(i) for i in raw_items]

            # Extract provider info with modelid
            providers = []
            for item in items:
                if item.get("Provider"):
                    providers.append({
                        "provider": item.get("Provider"),
                        "modelid": item.get("ModelId")
                    })
            return providers

    @staticmethod
    async def get_all_models() -> List[Dict[str, Any]]:
        """
        Get all unique models from the PricingTable with their providers.
        Returns a list of dicts with model info and available providers.
        """
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=PricingHandler.region) as client:
            response = await client.scan(
                TableName=PricingHandler.table_name,
            )
            raw_items = response.get("Items", [])
            items = [PricingHandler.deserialize_item(i) for i in raw_items]

            # Group by model
            models_map: Dict[str, Dict[str, Any]] = {}
            for item in items:
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
                        "owned_by": provider,  # First provider becomes owner
                        "providers": [],
                        # Use first provider's pricing as default
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