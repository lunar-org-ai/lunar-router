"""
Abstract interfaces for router-core providers.

These interfaces allow different implementations:
- SaaS: DynamoDB + Secrets Manager
- Data Plane: Local config + Control Plane API
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class StatsSummary:
    """Summary of provider performance statistics."""
    provider: str
    model: str
    n: int = 0
    p50_lat: float = 0.0
    p50_ttft: float = 0.0
    err_rate: float = 0.0
    updated_at: float = 0.0


@dataclass
class PricingInfo:
    """Pricing information for a model+provider combination."""
    provider: str
    model: str
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cache_input_per_million: float = 0.0

    def breakdown_usd(
        self,
        tokens_in: int,
        tokens_out: int,
        cache_tokens: int = 0,
    ) -> Dict[str, float]:
        """Calculate cost breakdown in USD."""
        input_cost = (tokens_in / 1_000_000) * self.input_per_million
        output_cost = (tokens_out / 1_000_000) * self.output_per_million
        cache_cost = (cache_tokens / 1_000_000) * self.cache_input_per_million

        return {
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "cache_input_cost_usd": cache_cost,
            "total_cost_usd": input_cost + output_cost + cache_cost,
        }


class StatsProvider(ABC):
    """
    Abstract interface for provider statistics.

    SaaS implementation: DynamoDB GlobalStatsHandler
    Data Plane implementation: Local metrics or Control Plane sync
    """

    @abstractmethod
    async def get_summary(self, model: str, provider: str) -> StatsSummary:
        """Get performance summary for a model+provider combination."""
        pass

    @abstractmethod
    async def record_request(
        self,
        model: str,
        provider: str,
        latency_ms: float,
        ttft_ms: float,
        success: bool,
        error_category: Optional[str] = None,
    ) -> None:
        """Record a request's metrics."""
        pass


class PricingProvider(ABC):
    """
    Abstract interface for pricing information.

    SaaS implementation: DynamoDB PricingHandler
    Data Plane implementation: Control Plane config sync
    """

    @abstractmethod
    async def get_price(self, provider: str, model: str) -> Optional[PricingInfo]:
        """Get pricing for a model+provider combination."""
        pass

    @abstractmethod
    async def get_providers_for_model(self, model: str) -> List[str]:
        """Get all providers that support a given model."""
        pass


class SecretsProvider(ABC):
    """
    Abstract interface for credentials/secrets.

    SaaS implementation: AWS Secrets Manager (BYOK)
    Data Plane implementation: Local environment or customer secrets
    """

    @abstractmethod
    async def get_credentials(
        self,
        provider: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get credentials for a provider.

        Returns dict with provider-specific keys:
        - OpenAI: {"api_key": "sk-..."}
        - Anthropic: {"api_key": "sk-ant-..."}
        - Bedrock: {"aws_access_key_id": "...", "aws_secret_access_key": "...", "aws_region": "..."}
        """
        pass


class TelemetryProvider(ABC):
    """
    Abstract interface for telemetry/usage tracking.

    SaaS implementation: DynamoDB TenantStatsHandler + SQS billing
    Data Plane implementation: Control Plane telemetry API
    """

    @abstractmethod
    async def record_usage(
        self,
        tenant_id: str,
        request_id: str,
        model: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float,
        ttft_ms: float,
        success: bool,
        error_category: Optional[str] = None,
    ) -> None:
        """Record usage telemetry for billing and analytics."""
        pass


# Default implementations for testing/development

class InMemoryStatsProvider(StatsProvider):
    """In-memory stats provider for testing."""

    def __init__(self):
        self._stats: Dict[str, StatsSummary] = {}

    async def get_summary(self, model: str, provider: str) -> StatsSummary:
        key = f"{provider}:{model}"
        return self._stats.get(key, StatsSummary(provider=provider, model=model))

    async def record_request(
        self,
        model: str,
        provider: str,
        latency_ms: float,
        ttft_ms: float,
        success: bool,
        error_category: Optional[str] = None,
    ) -> None:
        import time
        key = f"{provider}:{model}"
        existing = self._stats.get(key, StatsSummary(provider=provider, model=model))

        n = existing.n + 1
        # Running average
        existing.p50_lat = (existing.p50_lat * existing.n + latency_ms) / n
        existing.p50_ttft = (existing.p50_ttft * existing.n + ttft_ms) / n
        if not success:
            existing.err_rate = (existing.err_rate * existing.n + 1.0) / n
        else:
            existing.err_rate = (existing.err_rate * existing.n) / n
        existing.n = n
        existing.updated_at = time.time()

        self._stats[key] = existing


class StaticPricingProvider(PricingProvider):
    """Static pricing provider with hardcoded values."""

    PRICING = {
        ("openai", "gpt-4o-mini"): PricingInfo("openai", "gpt-4o-mini", 0.15, 0.60, 0.075),
        ("openai", "gpt-4o"): PricingInfo("openai", "gpt-4o", 2.50, 10.00, 1.25),
        ("openai", "gpt-4-turbo"): PricingInfo("openai", "gpt-4-turbo", 10.00, 30.00, 5.00),
        ("anthropic", "claude-3-5-sonnet-20241022"): PricingInfo("anthropic", "claude-3-5-sonnet-20241022", 3.00, 15.00, 0.30),
        ("anthropic", "claude-3-5-haiku-20241022"): PricingInfo("anthropic", "claude-3-5-haiku-20241022", 0.80, 4.00, 0.08),
        ("anthropic", "claude-3-opus-20240229"): PricingInfo("anthropic", "claude-3-opus-20240229", 15.00, 75.00, 1.50),
        ("google", "gemini-1.5-pro"): PricingInfo("google", "gemini-1.5-pro", 1.25, 5.00, 0.3125),
        ("google", "gemini-1.5-flash"): PricingInfo("google", "gemini-1.5-flash", 0.075, 0.30, 0.01875),
        ("mistral", "mistral-large-latest"): PricingInfo("mistral", "mistral-large-latest", 2.00, 6.00, 0.20),
        ("groq", "llama-3.1-70b-versatile"): PricingInfo("groq", "llama-3.1-70b-versatile", 0.59, 0.79, 0.0),
        ("groq", "llama-3.1-8b-instant"): PricingInfo("groq", "llama-3.1-8b-instant", 0.05, 0.08, 0.0),
    }

    async def get_price(self, provider: str, model: str) -> Optional[PricingInfo]:
        return self.PRICING.get((provider, model))

    async def get_providers_for_model(self, model: str) -> List[str]:
        return [p for (p, m) in self.PRICING.keys() if m == model]


class EnvSecretsProvider(SecretsProvider):
    """Secrets provider that reads from environment variables."""

    PROVIDER_ENV_KEYS = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
        "perplexity": ["PERPLEXITY_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "cerebras": ["CEREBRAS_API_KEY"],
        "cohere": ["COHERE_API_KEY", "CO_API_KEY"],
        "sambanova": ["SAMBANOVA_API_KEY"],
        "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    }

    async def get_credentials(
        self,
        provider: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        import os

        provider_lower = provider.lower()
        env_keys = self.PROVIDER_ENV_KEYS.get(provider_lower, [])

        if provider_lower == "bedrock":
            return {
                "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
            }

        for key in env_keys:
            value = os.environ.get(key)
            if value:
                return {"api_key": value}

        return {}


class NoOpTelemetryProvider(TelemetryProvider):
    """No-op telemetry provider for testing."""

    async def record_usage(
        self,
        tenant_id: str,
        request_id: str,
        model: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float,
        ttft_ms: float,
        success: bool,
        error_category: Optional[str] = None,
    ) -> None:
        pass  # No-op
