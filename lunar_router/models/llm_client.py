"""
LLM Client: Abstract interface for interacting with LLMs.

Provides a unified interface for different LLM providers
(OpenAI, Anthropic, Bedrock, local models, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class LLMResponse:
    """
    Response from an LLM.

    Attributes:
        text: The generated text response.
        tokens_used: Total tokens consumed (input + output).
        latency_ms: Response latency in milliseconds.
        model_id: ID of the model that generated the response.
        input_tokens: Number of input/prompt tokens (if available).
        output_tokens: Number of output/completion tokens (if available).
    """
    text: str
    tokens_used: int
    latency_ms: float
    model_id: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    @property
    def cost(self) -> Optional[float]:
        """Estimated cost (requires cost_per_1k_tokens to be set externally)."""
        return None  # Would need cost info to calculate


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Subclasses implement specific provider integrations.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def cost_per_1k_tokens(self) -> float:
        """Return the cost per 1000 tokens."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            LLMResponse with the generated text and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"


class OpenAIClient(LLMClient):
    """
    Client for OpenAI models.

    Supports GPT-4, GPT-3.5, O1 reasoning models, and other OpenAI models.
    """

    # Cost per 1k tokens (average of input/output, approximate)
    COSTS = {
        # GPT-4o family
        "gpt-4o": 0.00625,  # ($2.50 + $10) / 2 / 1000
        "gpt-4o-mini": 0.000375,  # ($0.15 + $0.60) / 2 / 1000
        # GPT-4 family
        "gpt-4-turbo": 0.02,  # ($10 + $30) / 2 / 1000
        "gpt-4": 0.045,  # ($30 + $60) / 2 / 1000
        # GPT-3.5
        "gpt-3.5-turbo": 0.001,  # ($0.50 + $1.50) / 2 / 1000
        # O1 reasoning models
        "o1": 0.0375,  # ($15 + $60) / 2 / 1000
        "o1-mini": 0.0075,  # ($3 + $12) / 2 / 1000
        "o1-preview": 0.0375,
        # O3 models
        "o3-mini": 0.00275,  # ($1.10 + $4.40) / 2 / 1000
    }

    # Models that use reasoning API (different parameters)
    REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3-mini"}

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o", "o1-mini").
            api_key: Optional API key (defaults to OPENAI_API_KEY env var).
            base_url: Optional base URL for API-compatible services.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self._model = model
        self._cost = self.COSTS.get(model, 0.001)  # Default cost
        self._is_reasoning = model in self.REASONING_MODELS
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start_time = time.time()

        if self._is_reasoning:
            # O1/O3 models use different API parameters
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                # Note: O1 models don't support temperature parameter
            )
        else:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            text=response.choices[0].message.content or "",
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )


class AnthropicClient(LLMClient):
    """
    Client for Anthropic Claude models.
    """

    COSTS = {
        "claude-3-5-sonnet-20241022": 0.003,
        "claude-3-5-haiku-20241022": 0.0008,
        "claude-3-opus-20240229": 0.015,
        "claude-3-sonnet-20240229": 0.003,
        "claude-3-haiku-20240307": 0.00025,
    }

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Model name.
            api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var).
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self._model = model
        self._cost = self.COSTS.get(model, 0.003)
        self.client = Anthropic(api_key=api_key)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start_time = time.time()

        response = self.client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.time() - start_time) * 1000

        text = ""
        if response.content and len(response.content) > 0:
            text = response.content[0].text

        return LLMResponse(
            text=text,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class MistralClient(LLMClient):
    """
    Client for Mistral AI models.

    Available models (January 2025):
    - Premier: mistral-large-latest, mistral-medium-latest, mistral-small-latest
    - Ministral: ministral-3b-latest, ministral-8b-latest
    - Open: open-mistral-nemo, open-mixtral-8x7b, open-mixtral-8x22b
    - Code: codestral-latest
    - Vision: pixtral-large-latest, pixtral-12b-2409 (requires image input)
    """

    # Costs per 1k tokens (average of input/output)
    COSTS = {
        # Premier models
        "mistral-large-latest": 0.002,
        "mistral-medium-latest": 0.0012,
        "mistral-small-latest": 0.0002,
        # Ministral family
        "ministral-3b-latest": 0.0001,
        "ministral-8b-latest": 0.00015,
        # Open models
        "open-mistral-nemo": 0.00015,
        "open-mixtral-8x7b": 0.0007,
        "open-mixtral-8x22b": 0.002,
        # Code models
        "codestral-latest": 0.0003,
        # Vision models (work for text too)
        "pixtral-large-latest": 0.002,
        "pixtral-12b-2409": 0.00015,
    }

    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Mistral client.

        Args:
            model: Model name.
            api_key: Optional API key (defaults to MISTRAL_API_KEY env var).
        """
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package required. Install with: pip install mistralai")

        import os
        self._model = model
        self._cost = self.COSTS.get(model, 0.001)
        self.client = Mistral(api_key=api_key or os.environ.get("MISTRAL_API_KEY"))

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start_time = time.time()

        response = self.client.chat.complete(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.time() - start_time) * 1000

        text = ""
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )


class GoogleClient(LLMClient):
    """
    Client for Google Gemini models.

    Available models (January 2025):
    - gemini-2.0-flash-exp: Latest experimental model
    - gemini-1.5-pro: Best for complex tasks
    - gemini-1.5-flash: Fast and efficient
    - gemini-1.5-flash-8b: Smallest and fastest
    """

    COSTS = {
        # Gemini 2.0
        "gemini-2.0-flash-exp": 0.0,  # Free during preview
        # Gemini 1.5 Pro
        "gemini-1.5-pro": 0.00175,  # ($1.25 + $2.50) / 2 / 1000 (under 128k)
        "gemini-1.5-pro-latest": 0.00175,
        # Gemini 1.5 Flash
        "gemini-1.5-flash": 0.0001125,  # ($0.075 + $0.15) / 2 / 1000
        "gemini-1.5-flash-latest": 0.0001125,
        "gemini-1.5-flash-8b": 0.00005625,  # ($0.0375 + $0.075) / 2 / 1000
        # Legacy
        "gemini-pro": 0.0005,
    }

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Google Gemini client.

        Args:
            model: Model name.
            api_key: Optional API key (defaults to GOOGLE_API_KEY env var).
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )

        import os
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        self._model = model
        self._cost = self.COSTS.get(model, 0.001)
        self.client = genai.GenerativeModel(model)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start_time = time.time()

        response = self.client.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        latency_ms = (time.time() - start_time) * 1000

        text = response.text if response.text else ""

        # Token counting
        input_tokens = None
        output_tokens = None
        total_tokens = 0

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

        return LLMResponse(
            text=text,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class GroqClient(LLMClient):
    """
    Client for Groq inference (ultra-fast inference).

    Available models (January 2025):
    - llama-3.3-70b-versatile: Latest Llama 3.3
    - llama-3.1-70b-versatile: Llama 3.1 70B
    - llama-3.1-8b-instant: Fast Llama 3.1 8B
    - mixtral-8x7b-32768: Mixtral MoE
    - gemma2-9b-it: Google Gemma 2
    """

    COSTS = {
        # Llama 3.3
        "llama-3.3-70b-versatile": 0.00059,  # $0.59/M tokens
        "llama-3.3-70b-specdec": 0.00059,
        # Llama 3.1
        "llama-3.1-70b-versatile": 0.00059,
        "llama-3.1-8b-instant": 0.00005,  # $0.05/M tokens
        # Llama 3.2
        "llama-3.2-90b-vision-preview": 0.0009,
        "llama-3.2-11b-vision-preview": 0.00018,
        "llama-3.2-3b-preview": 0.00006,
        "llama-3.2-1b-preview": 0.00004,
        # Mixtral
        "mixtral-8x7b-32768": 0.00024,  # $0.24/M tokens
        # Gemma
        "gemma2-9b-it": 0.0002,  # $0.20/M tokens
        # Whisper (audio)
        "whisper-large-v3": 0.000111,  # $0.111/hour
        "whisper-large-v3-turbo": 0.00004,
    }

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Groq client.

        Args:
            model: Model name.
            api_key: Optional API key (defaults to GROQ_API_KEY env var).
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq package required. Install with: pip install groq")

        import os
        self._model = model
        self._cost = self.COSTS.get(model, 0.0005)
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.time() - start_time) * 1000

        text = ""
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )


class VLLMClient(LLMClient):
    """
    Client for vLLM-served models.

    Connects to a vLLM server running locally or remotely.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        cost_per_1k: float = 0.0,
    ):
        """
        Initialize vLLM client.

        Args:
            base_url: URL of the vLLM server (e.g., "http://localhost:8000").
            model: Model name as served by vLLM.
            cost_per_1k: Cost per 1k tokens (typically 0 for local).
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for vLLM client")

        self._model = model
        self._cost = cost_per_1k
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="not-needed",  # vLLM doesn't require API key
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            text=response.choices[0].message.content or "",
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.

    Returns deterministic responses based on the prompt.
    """

    def __init__(
        self,
        model: str = "mock-model",
        cost_per_1k: float = 0.001,
        responses: Optional[dict[str, str]] = None,
        default_response: str = "Mock response",
        error_rate: float = 0.0,
    ):
        """
        Initialize mock client.

        Args:
            model: Identifier for this mock model.
            cost_per_1k: Simulated cost.
            responses: Optional mapping of prompts to responses.
            default_response: Response to return for unmapped prompts.
            error_rate: Probability of returning wrong answer (for testing).
        """
        self._model_id = model
        self._cost = cost_per_1k
        self._responses = responses or {}
        self._default_response = default_response
        self._error_rate = error_rate

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import random

        # Simulate some latency
        latency_ms = random.uniform(10, 100)

        # Get response
        if prompt in self._responses:
            text = self._responses[prompt]
        else:
            text = self._default_response

        # Simulate errors
        if self._error_rate > 0 and random.random() < self._error_rate:
            text = f"WRONG: {text}"

        # Estimate tokens
        tokens = len(prompt.split()) + len(text.split())

        return LLMResponse(
            text=text,
            tokens_used=tokens,
            latency_ms=latency_ms,
            model_id=self._model_id,
        )


class UnifiedClient(LLMClient):
    """Generic LLMClient backed by ``sdk.completion``.

    Lets ``create_client`` cover every provider registered in
    ``sdk.PROVIDERS`` without requiring a hand-written subclass for each.
    Used for providers like DeepSeek, Perplexity, Cerebras, Fireworks,
    Together, Cohere, Sambanova — anything OpenAI-chat-compatible.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def cost_per_1k_tokens(self) -> float:
        from ..model_prices import MODEL_INFO
        info = MODEL_INFO.get(self._model)
        if not info:
            return 0.001  # unknown — conservative default
        input_cost, output_cost = info[0], info[1]
        return (input_cost + output_cost) / 2 * 1000

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Lazy import to avoid circular dependency (sdk imports from models/*).
        from ..sdk import completion

        start = time.time()
        resp = completion(
            model=f"{self._provider}/{self._model}",
            messages=[{"role": "user", "content": prompt}],
            api_key=self._api_key,
            api_base=self._base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            force_direct=True,  # create_client is the "direct provider" entry point
        )
        latency_ms = (time.time() - start) * 1000

        usage = resp.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        text = ""
        choices = resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""

        return LLMResponse(
            text=text,
            tokens_used=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# Dedicated subclasses get first pick; anything else falls through to UnifiedClient.
_DEDICATED_CLIENTS: dict[str, type] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "mistral": MistralClient,
    "google": GoogleClient,
    "groq": GroqClient,
    "vllm": VLLMClient,
    "mock": MockLLMClient,
}

# Aliases between the two naming conventions in the codebase
# (models/llm_client.py uses "google"; sdk.PROVIDERS uses "gemini").
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "google": "google",
}


def create_client(
    provider: str,
    model: str,
    **kwargs,
) -> LLMClient:
    """Factory for LLM clients covering every provider in ``sdk.PROVIDERS``.

    Resolution order:
      1. If ``provider`` (or its alias) has a dedicated client class
         (OpenAI, Anthropic, Mistral, Google, Groq, vLLM, Mock), use it.
      2. Otherwise, if ``provider`` is registered in ``sdk.PROVIDERS``
         (DeepSeek, Perplexity, Cerebras, Sambanova, Together, Fireworks,
         Cohere, Bedrock, …), return a ``UnifiedClient`` that dispatches
         through ``sdk.completion``.
      3. Otherwise, raise ``ValueError``.

    Bedrock is registered but not directly usable via UnifiedClient
    (AWS SigV4 auth, non-OpenAI-compatible) — it will still raise.

    Args:
        provider: Provider name — see ``sdk.PROVIDERS`` for the full list.
        model: Model name.
        **kwargs: Passed to the client constructor (e.g. ``api_key``,
            ``base_url`` for vLLM / UnifiedClient).

    Example:
        >>> from lunar_router import create_client
        >>> create_client("openai", "gpt-4o-mini")        # dedicated
        >>> create_client("groq", "llama-3.1-8b-instant") # dedicated
        >>> create_client("deepseek", "deepseek-chat")    # via UnifiedClient
        >>> create_client("together", "meta-llama/...")   # via UnifiedClient
    """
    # Lazy import: PROVIDERS lives in sdk.py which imports from this module.
    from ..sdk import PROVIDERS

    canonical = _PROVIDER_ALIASES.get(provider, provider)

    if canonical in _DEDICATED_CLIENTS:
        return _DEDICATED_CLIENTS[canonical](model=model, **kwargs)

    # Bedrock uses a non-OpenAI-compatible format; UnifiedClient can't speak it.
    if canonical == "bedrock":
        raise ValueError(
            "Bedrock is not yet supported via create_client. "
            "Use sdk.completion with force_engine=True, or call the Bedrock SDK directly."
        )

    if canonical in PROVIDERS:
        return UnifiedClient(provider=canonical, model=model, **kwargs)

    known = sorted(set(_DEDICATED_CLIENTS) | set(PROVIDERS) | set(_PROVIDER_ALIASES))
    raise ValueError(f"Unknown provider: {provider}. Options: {known}")
