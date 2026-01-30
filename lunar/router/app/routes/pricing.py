from fastapi import APIRouter
from fastapi import HTTPException
from typing import List, Dict, Any
from ..database.PricingHandler import PricingHandler
from ..database.models import PricingItem, PricingResponse, PriceRequest

router = APIRouter(prefix="/v1/pricing", tags=["Pricing"])


# Get all models with their providers (from PricingTable)
@router.get("/models", response_model=List[Dict[str, Any]])
async def get_all_models():
    """
    List all models from PricingTable with their providers.
    This is the source of truth for available models.
    """
    return await PricingHandler.get_all_models()


# Upsert price (create or update)
@router.post("/", response_model=PricingItem)
async def upsert_price(request: PriceRequest):
    item = await PricingHandler.upsert_price(
        provider=request.provider,
        model=request.model,
        modelid=request.modelid,
        input_per_million=request.input_per_million,
        output_per_million=request.output_per_million,
        cache_input_per_million=request.cache_input_per_million
    )
    return item

# get all models from a provider
@router.get("/provider/{provider}", response_model=List[PricingItem])
async def get_by_provider(provider: str):
    items = await PricingHandler.get_by_provider(provider)
    if not items:
        raise HTTPException(status_code=404, detail="Provider not found")
    return items

# Get all providers offering a specific model
@router.get("/model/{model}", response_model=List[PricingItem])
async def get_by_model(model: str):
    items = await PricingHandler.get_by_model(model)
    if not items:
        raise HTTPException(status_code=404, detail="Model not found")
    return items

# Get specific price (provider + model)
@router.get("/{provider}/{model}", response_model=PricingResponse)
async def get_price(provider: str, model: str):
    item = await PricingHandler.get_price(provider, model)
    if not item:
        raise HTTPException(status_code=404, detail="Price not found")
    return item
