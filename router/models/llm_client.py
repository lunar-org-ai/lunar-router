"""
LLM Client: Abstract interface for interacting with LLMs.

P15.3.1 ports the abstract LLMClient + LLMResponse + AnthropicClient
(this project's only Anthropic-using client by default) + MockLLMClient
(test utility). Other concrete clients are stubbed with NotImplementedError;
land them when the first non-Anthropic routing target ships.
See ROADMAP_P15.3.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


_DEFERRED_MSG = (
    "{cls} is deferred — see ROADMAP_P15.3.md. "
    "Land this when the first non-Anthropic routing target ships."
)


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


class AnthropicClient(LLMClient):
    """
    Client for Anthropic Claude models.
    """

    COSTS = {
        "claude-haiku-4-5": 0.001,
        "claude-haiku-4-5-20251001": 0.001,
        "claude-sonnet-4-6": 0.003,
        "claude-opus-4-7": 0.015,
        "claude-3-5-sonnet-20241022": 0.003,
        "claude-3-5-haiku-20241022": 0.0008,
        "claude-3-opus-20240229": 0.015,
        "claude-3-sonnet-20240229": 0.003,
        "claude-3-haiku-20240307": 0.00025,
    }

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
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


# --- Deferred concrete clients ---


class OpenAIClient(LLMClient):
    """OpenAI client — DEFERRED. See ROADMAP_P15.3.md."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEFERRED_MSG.format(cls="OpenAIClient"))

    @property
    def model_id(self) -> str:
        raise NotImplementedError(_DEFERRED_MSG.format(cls="OpenAIClient"))

    @property
    def cost_per_1k_tokens(self) -> float:
        raise NotImplementedError(_DEFERRED_MSG.format(cls="OpenAIClient"))

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> LLMResponse:
        raise NotImplementedError(_DEFERRED_MSG.format(cls="OpenAIClient"))


class MistralClient(LLMClient):
    """Mistral client — DEFERRED. See ROADMAP_P15.3.md."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEFERRED_MSG.format(cls="MistralClient"))

    @property
    def model_id(self) -> str:
        raise NotImplementedError(_DEFERRED_MSG.format(cls="MistralClient"))

    @property
    def cost_per_1k_tokens(self) -> float:
        raise NotImplementedError(_DEFERRED_MSG.format(cls="MistralClient"))

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> LLMResponse:
        raise NotImplementedError(_DEFERRED_MSG.format(cls="MistralClient"))


# --- Test utility ---


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
