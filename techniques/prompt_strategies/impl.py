"""prompt_strategies technique — REAL Anthropic SDK call (P1.9).

The ``direct`` variant loads a prompt template, renders it with the
retrieved documents + the user request, and calls
``AnthropicClient.generate``. The routing decision picks the model
(Haiku/Sonnet/Opus). Token usage is stashed in ``ctx.state["llm_usage"]``
so the executor can compute exact cost instead of the char-based
estimate it falls back to.

Offline behavior — when ``ANTHROPIC_API_KEY`` is missing OR the
``anthropic`` package can't be imported, ``execute`` populates
``context.response`` with a clearly-marked offline marker. This keeps
the contract intact (response is always a non-empty string with a
positive token count) so tests that don't ship a key still run, and
so a fresh clone of the repo works before the operator wires their
key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from runtime.protocols import BaseTechnique, Context, Document, Stage


logger = logging.getLogger("techniques.prompt_strategies")

_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer based on the provided context. "
    "If the context does not contain the answer, say so explicitly."
)


class _Direct:
    def __init__(self, knobs: dict[str, Any]) -> None:
        self.prompt_path: str = str(knobs.get("prompt", "../prompts/system.md"))
        self.max_tokens: int = int(knobs.get("max_tokens", 2048))
        self.temperature: float = float(knobs.get("temperature", 0.3))

    def execute(self, context: Context) -> Context:
        model = (
            context.routing.model
            if context.routing and context.routing.model
            else _DEFAULT_MODEL
        )
        system = _load_system_prompt(self.prompt_path)
        user_msg = _render_user_message(context.documents, context.request)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            context.response = (
                f"[offline] No ANTHROPIC_API_KEY set — would have called "
                f"{model} with {len(context.documents)} doc(s). "
                f"Request: {context.request!r}"
            )
            return context

        try:
            from anthropic import Anthropic
        except ImportError:
            logger.warning("anthropic package not installed; running offline")
            context.response = (
                f"[offline] anthropic SDK not installed — would have called "
                f"{model}. Request: {context.request!r}"
            )
            return context

        client = Anthropic(api_key=api_key)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            logger.warning("Anthropic call failed (%s); returning error marker", e)
            context.response = (
                f"[error] LLM call failed: {type(e).__name__}: {e}. "
                f"Request: {context.request!r}"
            )
            return context

        text = ""
        if resp.content:
            parts = [getattr(block, "text", "") for block in resp.content]
            text = "".join(p for p in parts if p)
        context.response = text or "(empty response)"

        usage = getattr(resp, "usage", None)
        if usage is not None:
            context.state["llm_usage"] = {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "model": model,
            }
        return context


def _load_system_prompt(path: str) -> str:
    """Resolve the prompt path against a few likely roots; fall back to default."""
    candidates = [
        Path(path),
        Path("agent/pipeline") / path,
        Path("agent") / path,
        Path.cwd() / path,
    ]
    for p in candidates:
        try:
            resolved = p.resolve()
            if resolved.is_file():
                return resolved.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return _DEFAULT_SYSTEM


def _render_user_message(documents: list[Document], request: str) -> str:
    if not documents:
        return request
    blocks = []
    for i, doc in enumerate(documents, 1):
        blocks.append(f"[{i}] {doc.content}")
    context_str = "\n\n".join(blocks)
    return f"Context:\n{context_str}\n\nQuestion: {request}"


class PromptStrategiesTechnique(BaseTechnique):
    name = "prompt_strategies"
    variants = ("direct",)

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(
                f"prompt_strategies: unknown variant {variant!r}; expected one of {self.variants}"
            )
        return _Direct(knobs)
