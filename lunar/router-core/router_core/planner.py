"""
HealthFirstPlanner - Provider selection based on health metrics.

Selects the best provider for a given model based on:
1. Error rate (lower is better)
2. TTFT p50 (lower is better)
3. Price (lower is better)
4. Latency p50 (lower is better)

Also includes exploration for unknown providers to gather metrics.
"""

import asyncio
import random
import time
from typing import List, Tuple, Optional

from .adapters.base import ProviderAdapter
from .providers.base import StatsProvider, PricingProvider
from .cache.ranking_cache import get_ranking_cache

# Configuration
MAX_ERR_RATE = 0.20  # Skip providers with >20% error rate
MAX_STATS_AGE_SEC = 600  # Consider stats stale after 10 minutes
PROBE_UNKNOWN_PCT = 0.03  # 3% chance to probe unknown providers


class HealthFirstPlanner:
    """
    Select provider by health-based criteria.

    1) Filter providers with high error rates, outdated stats, etc.
    2) Sort by (err_rate ASC, p50_ttft ASC, price ASC, p50_lat ASC)
    3) Brief exploration for providers without metrics (PROBE_UNKNOWN_PCT)
    """

    def __init__(
        self,
        stats_provider: StatsProvider,
        pricing_provider: PricingProvider,
        max_err_rate: float = MAX_ERR_RATE,
        max_stats_age_sec: float = MAX_STATS_AGE_SEC,
        probe_unknown_pct: float = PROBE_UNKNOWN_PCT,
    ):
        """
        Initialize planner.

        Args:
            stats_provider: Provider for health statistics
            pricing_provider: Provider for pricing information
            max_err_rate: Maximum acceptable error rate
            max_stats_age_sec: Maximum age for stats to be considered fresh
            probe_unknown_pct: Probability of probing unknown providers
        """
        self.stats_provider = stats_provider
        self.pricing_provider = pricing_provider
        self.max_err_rate = max_err_rate
        self.max_stats_age_sec = max_stats_age_sec
        self.probe_unknown_pct = probe_unknown_pct

    async def rank(
        self,
        model: str,
        candidates: List[ProviderAdapter],
    ) -> List[ProviderAdapter]:
        """
        Rank candidate providers for a model.

        Args:
            model: Model identifier
            candidates: List of candidate ProviderAdapter instances

        Returns:
            Sorted list of ProviderAdapter instances (best first)
        """
        # Check cache first
        cache = get_ranking_cache()
        cached_result = cache.get(model)
        if cached_result is not None:
            # Return cached ranking, filtered to available candidates
            candidate_names = {c.name for c in candidates}
            filtered = [a for a in cached_result if a.name in candidate_names]
            if filtered:
                return filtered

        # Score candidates
        healthy: List[Tuple[float, float, float, float, ProviderAdapter]] = []
        unknown: List[ProviderAdapter] = []

        now = time.time()
        for candidate in candidates:
            summary = await self.stats_provider.get_summary(model, candidate.name)

            if summary.n == 0:
                # No stats yet - mark as unknown
                unknown.append(candidate)
                continue

            if summary.err_rate >= self.max_err_rate:
                # Too many errors - skip
                continue

            if (now - summary.updated_at) > self.max_stats_age_sec:
                # Stats too old - treat as unknown
                unknown.append(candidate)
                continue

            # Get pricing
            pricing = await self.pricing_provider.get_price(candidate.name, model)
            if pricing is None:
                unknown.append(candidate)
                continue

            price_value = (
                pricing.input_per_million +
                pricing.output_per_million +
                pricing.cache_input_per_million
            )

            healthy.append((
                summary.err_rate,
                summary.p50_ttft,
                price_value,
                summary.p50_lat,
                candidate,
            ))

        # Sort by err_rate -> ttft -> price -> latency
        healthy.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        ordered = [t[4] for t in healthy]

        # Occasionally probe unknown providers (exploration)
        if unknown and random.random() < self.probe_unknown_pct:
            ordered = [random.choice(unknown)] + ordered

        # Fallback if no healthy providers
        if not ordered:
            ordered = unknown[:] if unknown else candidates[:]

        # Cache the result asynchronously (non-blocking)
        asyncio.create_task(cache.set_async(model, ordered))

        return ordered

    async def select_best(
        self,
        model: str,
        candidates: List[ProviderAdapter],
        forced_provider: Optional[str] = None,
    ) -> Optional[ProviderAdapter]:
        """
        Select the best provider for a model.

        Args:
            model: Model identifier
            candidates: List of candidate ProviderAdapter instances
            forced_provider: Optional provider name to force selection

        Returns:
            Best ProviderAdapter, or None if no candidates
        """
        if not candidates:
            return None

        # Handle forced provider
        if forced_provider:
            for candidate in candidates:
                if candidate.name.lower() == forced_provider.lower():
                    return candidate
            # Forced provider not found - fall back to ranking
            return None

        ranked = await self.rank(model, candidates)
        return ranked[0] if ranked else None


class RoundRobinPlanner:
    """
    Simple round-robin provider selection.

    Useful for testing or when health metrics are not available.
    """

    def __init__(self):
        self._index: dict = {}

    async def rank(
        self,
        model: str,
        candidates: List[ProviderAdapter],
    ) -> List[ProviderAdapter]:
        """Rotate candidates in round-robin fashion."""
        if not candidates:
            return []

        idx = self._index.get(model, 0)
        rotated = candidates[idx:] + candidates[:idx]
        self._index[model] = (idx + 1) % len(candidates)
        return rotated

    async def select_best(
        self,
        model: str,
        candidates: List[ProviderAdapter],
        forced_provider: Optional[str] = None,
    ) -> Optional[ProviderAdapter]:
        """Select next provider in rotation."""
        if forced_provider:
            for c in candidates:
                if c.name.lower() == forced_provider.lower():
                    return c
            return None

        ranked = await self.rank(model, candidates)
        return ranked[0] if ranked else None
