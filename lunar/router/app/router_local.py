# app/router_local.py
"""
Health-First Planner for local development.
Uses local JSON-based handlers instead of DynamoDB.
"""
import asyncio
import time
import random
from typing import List, Tuple
from .adapters import ProviderAdapter
from .database.local import LocalGlobalStatsHandler as GlobalStatsHandler
from .database.local import LocalPricingHandler as PricingHandler
from .cache import get_ranking_cache

MAX_ERR = 0.20
MAX_STATS_AGE_SEC = 600
PROBE_UNKNOWN_PCT = 0.03


class HealthFirstPlanner:
    """
    Select provider by the following criteria:
    1) Filter providers with high error rates, outdated stats, etc.
    2) Sort by (err_rate ASC, p50_lat ASC, p50_ttft ASC)
    3) Brief exploration for providers without metrics (PROBE_UNKNOWN_PCT)
    """

    async def rank(self, model: str, candidates: List[ProviderAdapter]) -> List[ProviderAdapter]:
        # OPTIMIZATION: Check cache first
        cache = get_ranking_cache()
        cached_result = cache.get(model)
        if cached_result is not None:
            candidate_names = {c.name for c in candidates}
            filtered = [a for a in cached_result if a.name in candidate_names]
            if filtered:
                return filtered

        healthy: List[Tuple[float, float, float, ProviderAdapter]] = []
        unknown: List[ProviderAdapter] = []

        now = time.time()
        for c in candidates:
            summary = await GlobalStatsHandler.summary(model, c.name)
            if summary.n == 0:
                unknown.append(c)
                continue
            if summary.err_rate >= MAX_ERR:
                continue
            if (now - summary.updated_at) > MAX_STATS_AGE_SEC:
                unknown.append(c)
                continue

            pricing = await PricingHandler.get_price(c.name, model)
            if pricing is None:
                unknown.append(c)
                continue

            price_value = (pricing.input_per_million + pricing.output_per_million + pricing.cache_input_per_million)

            healthy.append((summary.err_rate, summary.p50_ttft, price_value, summary.p50_lat, c))

        # Sort by err_rate -> ttft -> price -> total_time
        healthy.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        ordered = [t[4] for t in healthy]

        if unknown and random.random() < PROBE_UNKNOWN_PCT:
            ordered = [random.choice(unknown)] + ordered

        if not ordered:
            ordered = unknown[:] if unknown else candidates[:]

        # OPTIMIZATION: Cache the result asynchronously (non-blocking)
        asyncio.create_task(cache.set_async(model, ordered))

        return ordered
