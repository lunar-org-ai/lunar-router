import time
import httpx
import logging
from typing import Tuple, Dict, Any, List, Optional, Union
from litellm import acompletion, token_counter
from .adapters import ProviderAdapter, _extract_prompt
from .helpers.secrets_config import is_byok_required, get_mistral_key_for_tenant, DEFAULT_MISTRAL_API_KEY

logger = logging.getLogger(__name__)

# Mistral API endpoint for direct calls
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Models that use extended thinking format - LiteLLM doesn't support this yet
# These models need direct API calls to bypass LiteLLM's response parsing
MAGISTRAL_MODELS = ["magistral-small", "magistral-medium", "magistral-large"]


def _is_magistral_model(model_name: str) -> bool:
    """Check if model is a Magistral model that uses thinking format."""
    model_lower = model_name.lower()
    return any(m in model_lower for m in MAGISTRAL_MODELS)


def _extract_text_from_content(content: Union[str, List, None]) -> str:
    """
    Extract text from Mistral response content.

    Handles both standard string content and Magistral's extended thinking format:
    - Standard: "Hello, how can I help?"
    - Magistral thinking format:
      [
        {"type": "thinking", "thinking": [{"type": "text", "text": "...reasoning..."}]},
        {"type": "text", "text": "actual response"}
      ]

    Args:
        content: The message content (string, list, or None)

    Returns:
        Extracted text string
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Magistral thinking format - extract text blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    # Direct text block
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    # Skip thinking blocks - only include final response
                    # But we could optionally extract thinking text if needed:
                    # thinking_content = block.get("thinking", [])
                    # for t in thinking_content:
                    #     if isinstance(t, dict) and t.get("type") == "text":
                    #         text_parts.append(f"[Thinking: {t.get('text', '')}]")
                    pass
            elif isinstance(block, str):
                text_parts.append(block)

        return "".join(text_parts)

    # Fallback - try to convert to string
    return str(content) if content else ""


class MistralAdapter(ProviderAdapter):
    """
    Adapter for Mistral AI models via LiteLLM with:
    - BYOK per tenant (Secrets Manager)
    - Controlled fallback according to policy (managed vs byok_required)
    - Streaming + metrics (ttft, latency, tokens)
    - Direct API calls for Magistral models (LiteLLM doesn't support thinking format)
    """

    def __init__(self, name: str, logical_model: str, model_name: str):
        super().__init__(name=name, model=logical_model)
        self.model_name = model_name
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client for direct API calls."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=300.0)
        return self._http_client

    async def _send_magistral_direct(
        self,
        messages: List[Dict[str, str]],
        api_key: str,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Send request directly to Mistral API for Magistral models.

        Bypasses LiteLLM because it doesn't support Magistral's thinking format.
        The thinking format returns content as a list instead of string, which
        causes LiteLLM's pydantic validation to fail.
        """
        start = time.time()

        # Get the actual model name without provider prefix
        model = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name

        payload = {
            "model": model,
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            client = await self._get_http_client()
            response = await client.post(
                MISTRAL_API_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

            latency_ms = (time.time() - start) * 1000.0
            data = response.json()

            # Extract content from response, handling thinking format
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            text = _extract_text_from_content(raw_content)

            # Extract usage from response
            usage = data.get("usage", {})
            ti = usage.get("prompt_tokens", 0)
            to = usage.get("completion_tokens", 0)

            logger.info(f"Magistral direct API call successful: {len(text)} chars, {ti}+{to} tokens")

            return (
                {"text": text},
                {
                    "ttft_ms": float(latency_ms),
                    "latency_ms": float(latency_ms),
                    "tokens_in": ti,
                    "tokens_out": to,
                    "error": 0.0,
                },
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start) * 1000.0
            error_text = e.response.text[:500] if e.response else str(e)
            logger.error(f"Magistral API HTTP error: {e.response.status_code} - {error_text}")
            return (
                {"error": f"Mistral API error: {e.response.status_code} - {error_text}"},
                {
                    "ttft_ms": 0.0,
                    "latency_ms": float(latency_ms),
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000.0
            logger.error(f"Magistral API error: {e}")
            return (
                {"error": f"Mistral API error: {str(e)}"},
                {
                    "ttft_ms": 0.0,
                    "latency_ms": float(latency_ms),
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

    def _ensure_messages(self, req: Dict[str, Any]) -> List[Dict[str, str]]:
        if "messages" in req and isinstance(req["messages"], list):
            return req["messages"]

        prompt = _extract_prompt(req)
        return [{"role": "user", "content": prompt}]

    def _count_tokens(self, messages, completion_text: str) -> Tuple[int, int]:
        # tokens in
        try:
            ti = token_counter(model=self.model_name, messages=messages) or 0
        except Exception:
            ti = 0

        # tokens out
        try:
            to = token_counter(model=self.model_name, text=completion_text) or 0
        except Exception:
            to = max(5, int(len(completion_text.split()) * 1.3))

        if ti == 0:
            text = " ".join([m.get("content", "") for m in messages])
            ti = max(1, int(len(text.split()) * 1.2))

        return int(ti), int(to)

    async def send(self, req: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        tenant_id = str(req.get("tenant") or "default")
        stream_requested = req.get("stream", False)
        byok_required = is_byok_required(tenant_id)

        try:
            api_key = get_mistral_key_for_tenant(tenant_id, byok_required)
        except RuntimeError as e:
            return (
                {"error": str(e)},
                {
                    "ttft_ms": 0.0,
                    "latency_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

        if not api_key:
            return (
                {"error": "No Mistral API key available for this environment"},
                {
                    "ttft_ms": 0.0,
                    "latency_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

        messages = self._ensure_messages(req)

        # Use direct API call for Magistral models (LiteLLM doesn't support thinking format)
        if _is_magistral_model(self.model_name) and not stream_requested:
            logger.info(f"Using direct API call for Magistral model: {self.model_name}")
            return await self._send_magistral_direct(messages, api_key)

        start = time.time()
        ttft_ms: Optional[float] = None
        text = ""

        try:
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                stream=stream_requested,
                api_key=api_key,
            )

            if stream_requested:
                ti = self._estimate_tokens_in(messages)
                
                return (
                    {
                        "stream": self._stream_with_metrics(response, start),
                        "messages": messages,
                        "model_name": self.model_name,
                    },
                    {
                        "ttft_ms": 0.0,
                        "latency_ms": 0.0,
                        "tokens_in": ti,
                        "tokens_out": 0,
                        "error": 0.0,
                    },
                )
            else:
                # Non-streaming response
                latency_ms = (time.time() - start) * 1000.0
                ttft_ms = latency_ms

                try:
                    # Handle Magistral's extended thinking format
                    raw_content = response.choices[0].message.content
                    text = _extract_text_from_content(raw_content)
                except Exception:
                    text = ""

                ti, to = self._count_tokens(messages, text)

                return (
                    {"text": text},
                    {
                        "ttft_ms": float(ttft_ms),
                        "latency_ms": float(latency_ms),
                        "tokens_in": ti,
                        "tokens_out": to,
                        "error": 0.0,
                    },
                )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000.0
            return (
                {"error": f"mistral error: {e}"},
                {
                    "ttft_ms": float(ttft_ms or 0.0),
                    "latency_ms": float(latency_ms),
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

    def healthy(self) -> Dict[str, Any]:
        """
        Generic Health:
        - Considers whether any global keys are configured.
        - BYOK-specific checks are evaluated at runtime in send().
        """
        has_global = bool(DEFAULT_MISTRAL_API_KEY)
        return {
            "ok": has_global,
            "err_rate": 0.0 if has_global else 1.0,
            "headroom": 1.0,
        }
