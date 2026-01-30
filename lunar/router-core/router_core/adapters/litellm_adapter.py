"""
LiteLLM-based adapter for multiple providers.

Supports: OpenAI, Anthropic, Google, Mistral, Groq, DeepSeek, Perplexity, etc.
"""

import time
from typing import Dict, Any, Tuple, Optional, AsyncGenerator

from .base import ProviderAdapter
from ..error_classifier import classify_error, ErrorCategory


class LiteLLMAdapter(ProviderAdapter):
    """
    LiteLLM-based adapter supporting multiple providers.

    Uses LiteLLM's unified API to call various LLM providers.
    """

    # Provider prefixes for LiteLLM
    PROVIDER_PREFIXES = {
        "openai": "",  # No prefix needed
        "anthropic": "anthropic/",
        "google": "gemini/",
        "mistral": "mistral/",
        "groq": "groq/",
        "deepseek": "deepseek/",
        "perplexity": "perplexity/",
        "cerebras": "cerebras/",
        "cohere": "cohere/",
        "sambanova": "sambanova/",
    }

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize LiteLLM adapter.

        Args:
            name: Provider name (e.g., "openai", "anthropic")
            model: Model identifier (e.g., "gpt-4o-mini")
            api_key: Optional API key (overrides environment)
            api_base: Optional API base URL
        """
        super().__init__(name, model)
        self.api_key = api_key
        self.api_base = api_base

    def _get_litellm_model(self) -> str:
        """Get the LiteLLM model string with provider prefix."""
        prefix = self.PROVIDER_PREFIXES.get(self.name.lower(), "")
        return f"{prefix}{self.model}"

    async def send(
        self,
        req: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Send a request via LiteLLM.

        Args:
            req: Request dictionary with "messages", "model", etc.
            credentials: Optional credentials dict with "api_key"

        Returns:
            Tuple of (response_dict, metrics_dict)
        """
        try:
            import litellm
            litellm.drop_params = True  # Ignore unsupported params
        except ImportError:
            return (
                {"error": "litellm not installed"},
                {"error": 1.0, "error_category": "invalid_request"},
            )

        start = time.perf_counter()
        ttft = 0.0

        # Resolve API key
        api_key = (
            (credentials or {}).get("api_key") or
            self.api_key
        )

        # Build LiteLLM call params
        litellm_model = self._get_litellm_model()
        messages = req.get("messages", [])

        # Estimate input tokens
        tokens_in = self._estimate_tokens_in(messages)

        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                api_key=api_key,
                api_base=self.api_base,
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens"),
                stream=False,
            )

            latency = (time.perf_counter() - start) * 1000
            ttft = latency  # For non-streaming, TTFT = latency

            # Extract response
            text = response.choices[0].message.content or ""
            tokens_out = response.usage.completion_tokens if response.usage else len(text.split())

            return (
                {"text": text},
                {
                    "ttft_ms": ttft,
                    "latency_ms": latency,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "error": 0.0,
                },
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            error_category = classify_error(e)

            return (
                {"error": str(e)},
                {
                    "ttft_ms": 0.0,
                    "latency_ms": latency,
                    "tokens_in": tokens_in,
                    "tokens_out": 0,
                    "error": 1.0,
                    "error_category": error_category.value,
                },
            )

    async def send_stream(
        self,
        req: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a streaming request via LiteLLM.

        Yields chunks in OpenAI-compatible format.
        """
        try:
            import litellm
            litellm.drop_params = True
        except ImportError:
            yield {"error": "litellm not installed"}
            return

        start = time.perf_counter()
        first_chunk = True

        api_key = (credentials or {}).get("api_key") or self.api_key
        litellm_model = self._get_litellm_model()
        messages = req.get("messages", [])

        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                api_key=api_key,
                api_base=self.api_base,
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens"),
                stream=True,
            )

            async for chunk in response:
                if first_chunk:
                    first_chunk = False
                    ttft = (time.perf_counter() - start) * 1000
                    # Inject TTFT into first chunk metadata
                    chunk._ttft_ms = ttft

                yield chunk

        except Exception as e:
            yield {"error": str(e)}


class OpenAIAdapter(LiteLLMAdapter):
    """OpenAI-specific adapter."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("openai", model, api_key)


class AnthropicAdapter(LiteLLMAdapter):
    """Anthropic-specific adapter."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("anthropic", model, api_key)


class GoogleAdapter(LiteLLMAdapter):
    """Google/Gemini-specific adapter."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("google", model, api_key)


class GroqAdapter(LiteLLMAdapter):
    """Groq-specific adapter."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("groq", model, api_key)


class MistralAdapter(LiteLLMAdapter):
    """Mistral-specific adapter."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("mistral", model, api_key)
