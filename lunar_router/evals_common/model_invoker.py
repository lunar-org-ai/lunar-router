"""
Model Invoker — calls the Go engine at LUNAR_ENGINE_URL/v1/chat/completions.

Replaces the old evaluations-api ModelInvoker that called ROUTER_BASE_URL.
Locally, the Go engine runs on port 8080.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENGINE_URL = os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")


class ModelInvoker:
    """Invoke an LLM via the Go engine's /v1/chat/completions endpoint."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or ENGINE_URL).rstrip("/")

    def _build_headers(self, authorization: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Lunar-Internal": "true",
        }
        if authorization:
            if authorization.startswith("sk_") or authorization.startswith("pk_"):
                headers["x-api-key"] = authorization
            elif authorization.startswith("Bearer "):
                headers["Authorization"] = authorization
            else:
                headers["Authorization"] = f"Bearer {authorization}"
        return headers

    def invoke(
        self,
        model: str,
        prompt: str,
        *,
        authorization: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Invoke a model with a single user prompt.

        Returns:
            {"output": str, "latency": float, "cost": float, "usage": dict}
        """
        messages = [{"role": "user", "content": prompt}]
        return self.invoke_with_messages(
            model, messages,
            authorization=authorization,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Models that require max_completion_tokens instead of max_tokens
    _COMPLETION_TOKEN_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}

    @staticmethod
    def _needs_completion_tokens(model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens."""
        bare = model.split("/")[-1]
        return bare in ModelInvoker._COMPLETION_TOKEN_MODELS

    def invoke_with_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        authorization: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **extra: Any,
    ) -> dict[str, Any]:
        """
        Invoke a model with structured messages.

        Args:
            extra: Additional payload fields (e.g. tools, tool_choice)

        Returns:
            {"output": str, "latency": float, "cost": float, "usage": dict,
             "tool_calls": list | None}
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._build_headers(authorization)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # o-series models require max_completion_tokens instead of max_tokens
        if self._needs_completion_tokens(model):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        # Merge extra fields (tools, tool_choice, etc.)
        payload.update(extra)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("Model invocation failed: %s %s — %s", e.code, e.reason, error_body)
            raise RuntimeError(f"Model invocation failed ({e.code}): {error_body}") from e
        except Exception as e:
            logger.error("Model invocation error: %s", e)
            raise

        latency = time.time() - start

        # Extract output
        output = ""
        tool_calls = None
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            output = message.get("content", "") or ""
            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]

        # Extract usage / cost
        usage = body.get("usage", {})
        cost = 0.0
        cost_data = body.get("cost")
        if isinstance(cost_data, (int, float)):
            cost = float(cost_data)
        elif isinstance(cost_data, dict):
            cost = float(cost_data.get("total_cost_usd", 0))
        elif isinstance(cost_data, str):
            cost = float(cost_data)
        elif "total_cost" in usage:
            cost = float(usage["total_cost"])

        return {
            "output": output,
            "latency": latency,
            "cost": cost,
            "usage": usage,
            "tool_calls": tool_calls,
        }
