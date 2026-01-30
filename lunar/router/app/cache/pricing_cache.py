import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PricingData:
    input_per_million: float
    output_per_million: float
    cache_input_per_million: float

class PricingCache:
    """
    In-memory cache for pricing data.
    Pricing rarely changes, so we use a longer TTL (5 minutes).
    This avoids repeated database calls for pricing lookups.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[float, Dict]] = {}
    
    def get(self, provider: str, model: str) -> Optional[Dict]:
        """Get cached pricing, returns None if expired or missing"""
        key = f"{provider}:{model}"
        if key not in self._cache:
            return None
        
        cached_time, cached_result = self._cache[key]
        if time.time() - cached_time > self.ttl_seconds:
            del self._cache[key]
            return None
        
        print(f"[CACHE HIT] Pricing cache hit for {provider}/{model}", flush=True)
        return cached_result
    
    def set(self, provider: str, model: str, pricing: Dict) -> None:
        """Store pricing result with timestamp (synchronous)"""
        key = f"{provider}:{model}"
        self._cache[key] = (time.time(), pricing)
    
    async def set_async(self, provider: str, model: str, pricing: Dict) -> None:
        """Store pricing result asynchronously (non-blocking)"""
        print(f"[CACHE SET] Storing pricing for {provider}/{model} (TTL: {self.ttl_seconds}s)", flush=True)
        self.set(provider, model, pricing)
    
    def get_avg_provider(self, provider: str) -> Optional[Dict]:
        """Get average pricing for a provider across all models"""
        key = f"{provider}:*avg*"
        if key not in self._cache:
            return None
        
        cached_time, cached_result = self._cache[key]
        if time.time() - cached_time > self.ttl_seconds:
            del self._cache[key]
            return None
        
        print(f"[CACHE HIT] Average pricing cache hit for provider '{provider}'", flush=True)
        return cached_result
    
    def set_avg_provider(self, provider: str, pricing: Dict) -> None:
        """Store average provider pricing (synchronous)"""
        key = f"{provider}:*avg*"
        self._cache[key] = (time.time(), pricing)
    
    async def set_avg_provider_async(self, provider: str, pricing: Dict) -> None:
        """Store average provider pricing asynchronously (non-blocking)"""
        print(f"[CACHE SET] Storing average pricing for provider '{provider}' (TTL: {self.ttl_seconds}s)", flush=True)
        self.set_avg_provider(provider, pricing)
    
    def invalidate(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Clear cache for specific provider/model or all"""
        if provider is None:
            self._cache.clear()
        else:
            keys_to_delete = []
            for key in self._cache.keys():
                if model is None:
                    if key.startswith(f"{provider}:"):
                        keys_to_delete.append(key)
                else:
                    if key == f"{provider}:{model}":
                        keys_to_delete.append(key)
            for key in keys_to_delete:
                del self._cache[key]
    
    def size(self) -> int:
        """Return number of cached entries"""
        return len(self._cache)

# Global singleton instance
_pricing_cache = PricingCache(ttl_seconds=300)

def get_pricing_cache() -> PricingCache:
    """Get global pricing cache instance"""
    return _pricing_cache
