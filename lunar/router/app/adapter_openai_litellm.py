import time
from typing import Tuple, Dict, Any, List, Optional
from litellm import acompletion, token_counter
from .adapters import ProviderAdapter, _extract_prompt
from .helpers.secrets_config import is_byok_required, get_key_for_tenant, DEFAULT_OPENAI_API_KEY
from .helpers.error_classifier import classify_error
from .models.error_types import ErrorCategory

class OpenAILiteLLMAdapter(ProviderAdapter):
    """
    Adapter for OpenAI models via LiteLLM with:
    - Byok per tenant (Secrets Manager)
    - Controlled fallback according to policy (managed vs byok_required)    
    - Streaming + metrics (ttft, latency, tokens)
    """

    def __init__(self, name: str, logical_model: str, model_name: str):
        super().__init__(name=name, model=logical_model)
        self.model_name = model_name

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
        stream = req.get("stream", False)
        byok_required = is_byok_required(tenant_id)

        try:
            api_key = get_key_for_tenant(tenant_id, byok_required)
        except RuntimeError as e:
            return (
                {
                    "error": str(e),
                    "error_category": ErrorCategory.AUTH_ERROR.value,
                    "error_details": {
                        "exception_type": type(e).__name__,
                        "provider": self.name,
                    },
                },
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
                {
                    "error": "No API key available for this environment",
                    "error_category": ErrorCategory.AUTH_ERROR.value,
                    "error_details": {
                        "exception_type": "MissingAPIKey",
                        "provider": self.name,
                    },
                },
                {
                    "ttft_ms": 0.0,
                    "latency_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": 1.0,
                },
            )

        messages = self._ensure_messages(req)
        start = time.time()
        ttft_ms: Optional[float] = None
        text = ""

        try:
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                stream=stream,
                api_key=api_key,
            )

            if stream:
                # For streaming: estimate tokens_in WITHOUT blocking
                # Use word count heuristic instead of token_counter call
                ti = self._estimate_tokens_in(messages)
                
                # Return immediately with the stream generator
                # Don't wait for token_counter to complete - this is what's slowing down TTFT
                return (
                    {
                        "stream": self._stream_with_metrics(response, start),
                        "messages": messages,
                        "model_name": self.model_name,
                    },
                    {
                        "ttft_ms": 0.0,  # Will be calculated during streaming
                        "latency_ms": 0.0,  # Will be calculated at the end
                        "tokens_in": ti,
                        "tokens_out": 0,
                        "error": 0.0,
                    },
                )
            else:
                # Handle non-streaming response (ModelResponse object)
                latency_ms = (time.time() - start) * 1000.0
                ttft_ms = latency_ms  # For non-streaming, ttft equals total latency
                
                # Extract text from ModelResponse
                try:
                    text = response.choices[0].message.content or ""
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
            error_category = classify_error(e)
            return (
                {
                    "error": f"openai error: {e}",
                    "error_category": error_category,
                    "error_details": {
                        "exception_type": type(e).__name__,
                        "provider": self.name,
                    },
                },
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
        has_global = bool(DEFAULT_OPENAI_API_KEY)
        return {
            "ok": has_global,
            "err_rate": 0.0 if has_global else 1.0,
            "headroom": 1.0,
        }
