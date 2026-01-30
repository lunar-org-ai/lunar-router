"""
Base ProviderAdapter class for LLM providers.

All provider adapters inherit from this class and implement the send() method.
"""

import asyncio
from typing import Dict, Any, List, Tuple, AsyncGenerator, Optional


class ProviderAdapter:
    """
    Base class for LLM provider adapters.

    Each adapter wraps a specific provider's API and normalizes
    the request/response format.
    """

    def __init__(self, name: str, model: str):
        """
        Initialize adapter.

        Args:
            name: Provider name (e.g., "openai", "anthropic")
            model: Model identifier (e.g., "gpt-4o-mini")
        """
        self.name = name
        self.model = model

    async def send(
        self,
        req: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Send a request to the provider.

        Args:
            req: Request dictionary with keys like "messages", "model", "stream", etc.
            credentials: Optional credentials dict (api_key, etc.)

        Returns:
            Tuple of (response_dict, metrics_dict)
            - response_dict: Contains "text" or "error" key
            - metrics_dict: Contains "ttft_ms", "latency_ms", "tokens_in", "tokens_out", "error"
        """
        raise NotImplementedError("Subclasses must implement send()")

    async def send_stream(
        self,
        req: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a streaming request to the provider.

        Yields chunks with format compatible with OpenAI streaming.

        Args:
            req: Request dictionary
            credentials: Optional credentials dict

        Yields:
            Streaming chunks in OpenAI format
        """
        raise NotImplementedError("Subclasses must implement send_stream()")

    def healthy(self) -> Dict[str, Any]:
        """
        Return health status of the provider.

        Returns:
            Dict with "ok", "err_rate", "headroom" keys
        """
        return {"ok": True, "err_rate": 0.0, "headroom": 1.0}

    def _estimate_tokens_in(self, messages: List[Dict[str, str]]) -> int:
        """
        Fast heuristic for estimating input tokens WITHOUT calling token_counter.

        Uses simple word count heuristic (~1.25 tokens per word for English).
        This avoids the ~30ms LiteLLM overhead.

        Args:
            messages: List of message dicts with "content" keys

        Returns:
            Estimated token count
        """
        text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
        word_count = len(text.split())
        estimated = max(1, int(word_count * 1.25))
        return estimated

    async def _stream_with_metrics(
        self,
        response: AsyncGenerator,
        start: float,
    ) -> AsyncGenerator[Any, None]:
        """
        Generator that yields chunks from the stream while tracking metrics.

        Args:
            response: Async generator from provider
            start: Start timestamp for TTFT calculation

        Yields:
            Chunks from the provider
        """
        first_chunk = True
        async for chunk in response:
            if first_chunk:
                first_chunk = False
                # TTFT is calculated at first chunk arrival

            yield chunk


def _extract_prompt(req: Dict[str, Any]) -> str:
    """
    Extract prompt text from request for token estimation.

    Args:
        req: Request dictionary

    Returns:
        Concatenated prompt text
    """
    if "prompt" in req and isinstance(req["prompt"], str):
        return req["prompt"]

    msgs = req.get("messages") or []
    if isinstance(msgs, list):
        user_contents = [
            (m.get("content") if isinstance(m, dict) else None)
            for m in msgs
            if isinstance(m, dict) and m.get("role") in ("user", "system")
        ]
        user_contents = [c for c in user_contents if c]
        if user_contents:
            return "\n".join(user_contents)

    return ""
