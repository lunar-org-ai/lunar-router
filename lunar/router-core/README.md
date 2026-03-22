# router-core

Core LLM routing logic for Lunar Router. This package provides the shared routing functionality used by both the SaaS router and Data Plane router.

## Installation

```bash
# Basic installation
pip install -e .

# With LiteLLM adapters
pip install -e ".[litellm]"

# With AWS adapters (Bedrock, Secrets Manager)
pip install -e ".[aws]"

# All adapters
pip install -e ".[all]"
```

## Architecture

```
router-core/
├── router_core/
│   ├── __init__.py          # Main exports
│   ├── planner.py           # HealthFirstPlanner, RoundRobinPlanner
│   ├── error_classifier.py  # Error categorization
│   ├── adapters/            # LLM provider adapters
│   │   ├── base.py          # ProviderAdapter base class
│   │   ├── litellm_adapter.py  # LiteLLM-based adapters
│   │   └── mock_adapter.py  # Mock adapter for testing
│   ├── cache/               # Caching layer
│   │   ├── ranking_cache.py # Provider ranking cache (10s TTL)
│   │   └── pricing_cache.py # Pricing cache (5min TTL)
│   ├── providers/           # Abstract interfaces
│   │   └── base.py          # StatsProvider, PricingProvider, etc.
│   └── models/              # Pydantic schemas
│       └── schemas.py       # OpenAI-compatible models
└── pyproject.toml
```

## Usage

### Basic Usage

```python
from router_core import (
    HealthFirstPlanner,
    InMemoryStatsProvider,
    StaticPricingProvider,
)
from router_core.adapters.mock_adapter import MockAdapter

# Create providers
stats = InMemoryStatsProvider()
pricing = StaticPricingProvider()

# Create planner
planner = HealthFirstPlanner(stats, pricing)

# Create adapters
adapters = [
    MockAdapter("openai", "gpt-4o-mini"),
    MockAdapter("anthropic", "claude-3-5-sonnet"),
]

# Rank providers
ranked = await planner.rank("gpt-4o-mini", adapters)
best = ranked[0]

# Send request
response, metrics = await best.send({
    "messages": [{"role": "user", "content": "Hello!"}],
})
```

### With LiteLLM Adapters

```python
from router_core.adapters.litellm_adapter import OpenAIAdapter, AnthropicAdapter

adapters = [
    OpenAIAdapter("gpt-4o-mini", api_key="sk-..."),
    AnthropicAdapter("claude-3-5-sonnet-20241022", api_key="sk-ant-..."),
]

# Use with planner
best = await planner.select_best("gpt-4o-mini", adapters)
response, metrics = await best.send(request)
```

### Custom Provider Implementations

For SaaS or Data Plane deployments, implement the provider interfaces:

```python
from router_core.providers import StatsProvider, PricingProvider, StatsSummary, PricingInfo

class DynamoDBStatsProvider(StatsProvider):
    """DynamoDB-backed stats provider for SaaS."""

    async def get_summary(self, model: str, provider: str) -> StatsSummary:
        # Query DynamoDB GlobalStats table
        ...

    async def record_request(self, ...):
        # Update DynamoDB stats
        ...

class ControlPlaneConfigProvider(PricingProvider):
    """Config sync from Control Plane for Data Plane."""

    async def get_price(self, provider: str, model: str) -> PricingInfo:
        # Fetch from Control Plane API
        ...
```

## Provider Interfaces

| Interface | Purpose | SaaS Implementation | Data Plane Implementation |
|-----------|---------|---------------------|---------------------------|
| `StatsProvider` | Provider health metrics | DynamoDB GlobalStatsHandler | Local metrics / Control Plane sync |
| `PricingProvider` | Model pricing | DynamoDB PricingHandler | Control Plane config sync |
| `SecretsProvider` | API credentials | AWS Secrets Manager | Local env / customer secrets |
| `TelemetryProvider` | Usage tracking | DynamoDB + SQS | Control Plane telemetry API |

## Error Classification

The `error_classifier` module categorizes errors for intelligent routing:

```python
from router_core import classify_error, is_retryable, ErrorCategory

category = classify_error(exception)
if is_retryable(category):
    # Retry with backoff
    ...
elif category == ErrorCategory.AUTH_ERROR:
    # Skip this provider
    ...
```

## Caching

Two caches optimize routing performance:

- **RankingCache** (10s TTL): Caches provider rankings to reduce stats DB calls
- **PricingCache** (5min TTL): Caches pricing to reduce pricing DB calls

These caches are the primary optimization for TTFT reduction.

## License

MIT
