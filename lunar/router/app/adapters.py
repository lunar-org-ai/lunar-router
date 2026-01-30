# app/adapters.py
import asyncio
import random
from typing import Tuple, Dict, Any, List

class ProviderAdapter:
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    async def send(self, req: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        raise NotImplementedError

    def healthy(self) -> Dict[str, Any]:
        return {"ok": True, "err_rate": 0.0, "headroom": 1.0}

    def _estimate_tokens_in(self, messages: List[Dict[str, str]]) -> int:
        """
        Fast heuristic for estimating input tokens WITHOUT calling token_counter.
        Avoids the ~30ms LiteLLM overhead.
        
        Uses simple word count heuristic:
        - ~1.2-1.3 tokens per word for English
        """
        text = " ".join([m.get("content", "") for m in messages])
        word_count = len(text.split())
        estimated = max(1, int(word_count * 1.25))
        return estimated

    async def _stream_with_metrics(self, response, start: float):
        """
        Generator that yields chunks from the stream while tracking metrics.
        This runs async and doesn't block the TTFT calculation.
        """
        first_chunk = True
        async for chunk in response:
            if first_chunk:
                first_chunk = False
                # TTFT is calculated at first chunk arrival - no overhead
            
            delta_content = ""
            try:
                delta_content = chunk.choices[0].delta.get("content", "") or ""
            except Exception:
                delta_content = getattr(
                    getattr(chunk.choices[0], "delta", None),
                    "content",
                    "",
                ) or ""
            
            yield chunk

def _extract_prompt(req: Dict[str, Any]) -> str:
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

class MockProvider(ProviderAdapter):
    def __init__(
        self,
        name: str,
        model: str,
        base_latency_ms: int | None = None,
        jitter_ms: int | None = None,
        ttft_ms: int | None = None,
        error_rate: float | None = None,
    ):
        super().__init__(name, model)
        self.base_latency_ms = 0 if base_latency_ms is None else int(base_latency_ms)
        self.jitter_ms = 0 if jitter_ms is None else int(jitter_ms)
        self.base_ttft_ms = 0 if ttft_ms is None else int(ttft_ms)
        self.error_rate = 0.0 if error_rate is None else float(error_rate)

    async def send(self, req: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        prompt = _extract_prompt(req)

        ttft = max(0, int(random.gauss(self.base_ttft_ms, self.base_ttft_ms * 0.15 if self.base_ttft_ms else 0)))
        if ttft > 0:
            await asyncio.sleep(ttft / 1000.0)

        rest_ms = max(0, int(
            self.base_latency_ms +
            (random.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms else 0)
        ))
        if rest_ms > 0:
            await asyncio.sleep(rest_ms / 1000.0)

        latency = ttft + rest_ms
        error = random.random() < self.error_rate if self.error_rate else False

        tokens_in = max(1, int(len(prompt.split()) * 1.2))
        tokens_out = max(5, int(15 + random.random() * 40))

        if error:
            return (
                {"error": "mock failure"},
                {
                    "ttft_ms": float(ttft),
                    "latency_ms": float(latency),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "error": 1.0,
                },
            )

        text = f"[{self.name}] Mocked response ({self.model}) to: {prompt[:64]}..."
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

    def healthy(self) -> Dict[str, Any]:
        return {"ok": True, "err_rate": self.error_rate, "headroom": 1.0}
