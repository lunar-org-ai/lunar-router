"""
Mock adapter for testing.

Simulates LLM responses with configurable latency and error rates.
"""

import asyncio
import random
from typing import Dict, Any, Tuple, Optional, AsyncGenerator

from .base import ProviderAdapter, _extract_prompt


class MockAdapter(ProviderAdapter):
    """
    Mock adapter for testing routing logic.

    Simulates LLM responses with configurable:
    - Base latency
    - Jitter
    - Time to first token (TTFT)
    - Error rate
    """

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-model",
        base_latency_ms: int = 100,
        jitter_ms: int = 20,
        ttft_ms: int = 50,
        error_rate: float = 0.0,
    ):
        """
        Initialize mock adapter.

        Args:
            name: Provider name
            model: Model identifier
            base_latency_ms: Base response latency in ms
            jitter_ms: Latency jitter (random +/-)
            ttft_ms: Time to first token in ms
            error_rate: Probability of error (0.0-1.0)
        """
        super().__init__(name, model)
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.base_ttft_ms = ttft_ms
        self.error_rate = error_rate

    async def send(
        self,
        req: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Send a mock request.

        Simulates latency and optionally returns errors.
        """
        prompt = _extract_prompt(req)

        # Simulate TTFT
        ttft = max(0, int(random.gauss(
            self.base_ttft_ms,
            self.base_ttft_ms * 0.15 if self.base_ttft_ms else 0
        )))
        if ttft > 0:
            await asyncio.sleep(ttft / 1000.0)

        # Simulate remaining latency
        rest_ms = max(0, int(
            self.base_latency_ms +
            random.uniform(-self.jitter_ms, self.jitter_ms)
        ))
        if rest_ms > 0:
            await asyncio.sleep(rest_ms / 1000.0)

        latency = ttft + rest_ms

        # Estimate tokens
        tokens_in = max(1, int(len(prompt.split()) * 1.2))
        tokens_out = max(5, int(15 + random.random() * 40))

        # Check for error
        if random.random() < self.error_rate:
            return (
                {"error": "Mock failure (simulated)"},
                {
                    "ttft_ms": float(ttft),
                    "latency_ms": float(latency),
                    "tokens_in": tokens_in,
                    "tokens_out": 0,
                    "error": 1.0,
                    "error_category": "server_error",
                },
            )

        # Success response
        text = f"[{self.name}] Mock response for model {self.model}: {prompt[:64]}..."

        return (
            {"text": text},
            {
                "ttft_ms": float(ttft),
                "latency_ms": float(latency),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "error": 0.0,
            },
        )

    async def send_stream(
        self,
        req: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a mock streaming request.

        Yields chunks to simulate streaming.
        """
        prompt = _extract_prompt(req)

        # Simulate TTFT
        ttft = max(0, int(random.gauss(
            self.base_ttft_ms,
            self.base_ttft_ms * 0.15 if self.base_ttft_ms else 0
        )))
        if ttft > 0:
            await asyncio.sleep(ttft / 1000.0)

        # Check for error
        if random.random() < self.error_rate:
            yield {"error": "Mock streaming failure"}
            return

        # Generate mock response in chunks
        text = f"[{self.name}] Mock streaming response for: {prompt[:32]}..."
        words = text.split()

        for i, word in enumerate(words):
            chunk = {
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None if i < len(words) - 1 else "stop",
                }]
            }
            yield chunk
            await asyncio.sleep(0.01)  # Small delay between chunks

    def healthy(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "ok": True,
            "err_rate": self.error_rate,
            "headroom": 1.0,
        }
