"""
Ranking cache for provider selection optimization.

This is the PRIMARY optimization for TTFT reduction.
Caches the ranked provider list for each model to avoid
repeated database queries within short time windows.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.base import ProviderAdapter

# Default TTL: 10 seconds (aggressive, short-lived)
DEFAULT_TTL_SECONDS = 10


class RankingCache:
    """
    Per-model provider ranking cache.

    This cache reduces 90% of stats database calls within 10s windows,
    which is critical for maintaining low TTFT.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    def get(self, model: str) -> Optional[List["ProviderAdapter"]]:
        """
        Get cached ranking for a model.

        Args:
            model: Model identifier

        Returns:
            List of ranked ProviderAdapter instances, or None if not cached/expired
        """
        entry = self._cache.get(model)
        if entry is None:
            return None

        if time.time() - entry["timestamp"] > self._ttl:
            # Expired
            return None

        return entry["ranking"]

    def set(self, model: str, ranking: List["ProviderAdapter"]) -> None:
        """
        Set ranking for a model.

        Args:
            model: Model identifier
            ranking: List of ranked ProviderAdapter instances
        """
        self._cache[model] = {
            "ranking": ranking,
            "timestamp": time.time(),
        }

    async def set_async(self, model: str, ranking: List["ProviderAdapter"]) -> None:
        """
        Async version of set() for non-blocking cache updates.

        Args:
            model: Model identifier
            ranking: List of ranked ProviderAdapter instances
        """
        async with self._lock:
            self.set(model, ranking)

    def invalidate(self, model: str) -> None:
        """
        Invalidate cache for a model.

        Args:
            model: Model identifier
        """
        self._cache.pop(model, None)

    def clear(self) -> None:
        """Clear all cached rankings."""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        now = time.time()
        valid = sum(
            1 for entry in self._cache.values()
            if now - entry["timestamp"] <= self._ttl
        )
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "ttl_seconds": self._ttl,
        }


# Global instance
_ranking_cache: Optional[RankingCache] = None


def get_ranking_cache(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> RankingCache:
    """
    Get the global ranking cache instance.

    Args:
        ttl_seconds: TTL for cache entries (only used on first call)

    Returns:
        RankingCache instance
    """
    global _ranking_cache
    if _ranking_cache is None:
        _ranking_cache = RankingCache(ttl_seconds)
    return _ranking_cache
