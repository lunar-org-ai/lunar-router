"""
Pricing cache for cost optimization.

Caches pricing information to avoid repeated database queries.
TTL: 5 minutes (pricing changes infrequently).
"""

import time
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.base import PricingInfo

# Default TTL: 5 minutes
DEFAULT_TTL_SECONDS = 300


class PricingCache:
    """
    Cache for pricing information.

    Reduces DynamoDB queries for pricing by caching results
    with a 5-minute TTL.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, provider: str, model: str) -> Optional["PricingInfo"]:
        """
        Get cached pricing info.

        Args:
            provider: Provider name
            model: Model identifier

        Returns:
            PricingInfo if cached and valid, None otherwise
        """
        key = f"{provider}:{model}"
        entry = self._cache.get(key)

        if entry is None:
            return None

        if time.time() - entry["timestamp"] > self._ttl:
            return None

        return entry["pricing"]

    def set(self, provider: str, model: str, pricing: "PricingInfo") -> None:
        """
        Cache pricing info.

        Args:
            provider: Provider name
            model: Model identifier
            pricing: PricingInfo to cache
        """
        key = f"{provider}:{model}"
        self._cache[key] = {
            "pricing": pricing,
            "timestamp": time.time(),
        }

    def invalidate(self, provider: str, model: str) -> None:
        """Invalidate cache entry."""
        key = f"{provider}:{model}"
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached pricing."""
        self._cache.clear()


# Global instance
_pricing_cache: Optional[PricingCache] = None


def get_pricing_cache(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> PricingCache:
    """Get the global pricing cache instance."""
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = PricingCache(ttl_seconds)
    return _pricing_cache
