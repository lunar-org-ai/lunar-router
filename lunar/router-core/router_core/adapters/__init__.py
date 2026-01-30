"""LLM Provider adapters for router-core."""

from .base import ProviderAdapter, _extract_prompt

__all__ = [
    "ProviderAdapter",
    "_extract_prompt",
]

# Optional adapters (require additional dependencies)
# These are imported dynamically to avoid import errors when deps are missing

def get_litellm_adapter():
    """Get LiteLLM-based OpenAI adapter (requires litellm extra)."""
    from .litellm_adapter import LiteLLMAdapter
    return LiteLLMAdapter

def get_anthropic_adapter():
    """Get Anthropic adapter (requires litellm extra)."""
    from .litellm_adapter import LiteLLMAdapter
    return LiteLLMAdapter  # Uses LiteLLM for Anthropic

def get_mock_adapter():
    """Get mock adapter for testing."""
    from .mock_adapter import MockAdapter
    return MockAdapter
