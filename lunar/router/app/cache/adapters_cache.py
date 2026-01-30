import time
from typing import List, Dict, Tuple, Optional
from ..adapters import ProviderAdapter

class AdaptersCache:
    """
    In-memory cache for provider adapters.
    Reduces database calls and adapter creation overhead.
    
    Caches the list of adapters built from database providers.
    TTL: 5 minutes (longer than ranking cache since adapters rarely change).
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[float, List[ProviderAdapter]]] = {}
    
    def get(self, model: str) -> Optional[List[ProviderAdapter]]:
        """Get cached adapters, returns None if expired or missing"""
        if model not in self._cache:
            return None
        
        cached_time, cached_result = self._cache[model]
        if time.time() - cached_time > self.ttl_seconds:
            del self._cache[model]
            return None
        
        print(f"[CACHE HIT] Adapters cache hit for model '{model}' - returned {len(cached_result)} adapters", flush=True)
        return cached_result
    
    def set(self, model: str, adapters: List[ProviderAdapter]) -> None:
        """Store adapters result with timestamp (synchronous)"""
        self._cache[model] = (time.time(), adapters)
    
    async def set_async(self, model: str, adapters: List[ProviderAdapter]) -> None:
        """Store adapters result asynchronously (non-blocking)"""
        print(f"[CACHE SET] Storing adapters for model '{model}' with {len(adapters)} adapters (TTL: {self.ttl_seconds}s)", flush=True)
        self.set(model, adapters)
    
    def invalidate(self, model: Optional[str] = None) -> None:
        """Clear cache for a specific model or all models"""
        if model is None:
            self._cache.clear()
            print(f"[CACHE INVALIDATE] Cleared all adapters cache", flush=True)
        elif model in self._cache:
            del self._cache[model]
            print(f"[CACHE INVALIDATE] Cleared adapters cache for model '{model}'", flush=True)
    
    def size(self) -> int:
        """Return number of cached entries"""
        return len(self._cache)

# Global singleton instance
_adapters_cache = AdaptersCache(ttl_seconds=300)

def get_adapters_cache() -> AdaptersCache:
    """Get global adapters cache instance"""
    return _adapters_cache
