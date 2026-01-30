"""
Pricing routes for local development.
Uses local JSON-based storage instead of DynamoDB.
"""
from fastapi import APIRouter
from fastapi import HTTPException
from typing import List, Dict, Any
from ..database.local import LocalPricingHandler as PricingHandler
from ..database.models import PricingItem, PricingResponse, PriceRequest

router = APIRouter(prefix="/v1/pricing", tags=["Pricing"])


@router.get("/models", response_model=List[Dict[str, Any]])
async def get_all_models():
    """
    List all models from pricing data with their providers.
    This is the source of truth for available models.
    """
    return await PricingHandler.get_all_models()


@router.post("/", response_model=PricingItem)
async def upsert_price(request: PriceRequest):
    """Create or update a price entry."""
    item = await PricingHandler.upsert_price(
        provider=request.provider,
        model=request.model,
        modelid=request.modelid,
        input_per_million=request.input_per_million,
        output_per_million=request.output_per_million,
        cache_input_per_million=request.cache_input_per_million
    )
    return item


@router.get("/provider/{provider}", response_model=List[PricingItem])
async def get_by_provider(provider: str):
    """Get all models from a provider."""
    items = await PricingHandler.get_by_provider(provider)
    if not items:
        raise HTTPException(status_code=404, detail="Provider not found")
    return items


@router.get("/model/{model}", response_model=List[PricingItem])
async def get_by_model(model: str):
    """Get all providers offering a specific model."""
    items = await PricingHandler.get_by_model(model)
    if not items:
        raise HTTPException(status_code=404, detail="Model not found")
    return items


@router.get("/{provider}/{model}", response_model=PricingResponse)
async def get_price(provider: str, model: str):
    """Get specific price for a provider/model combination."""
    item = await PricingHandler.get_price(provider, model)
    if not item:
        raise HTTPException(status_code=404, detail="Price not found")
    return item
