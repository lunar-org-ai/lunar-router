"""prompt_strategies technique — STUB implementations.

Phase 1.4 stub: returns a canned response that summarizes what *would* have
happened. Phase 1.9 swaps this for a real Anthropic SDK call.
"""

from __future__ import annotations

from typing import Any

from runtime.protocols import BaseTechnique, Context, Stage


class _DirectStub:
    def __init__(self, knobs: dict[str, Any]) -> None:
        self.prompt_path: str = str(knobs.get("prompt", "../prompts/system.md"))
        self.max_tokens: int = int(knobs.get("max_tokens", 2048))
        self.temperature: float = float(knobs.get("temperature", 0.3))

    def execute(self, context: Context) -> Context:
        model = context.routing.model if context.routing else "<no routing decision>"
        n_docs = len(context.documents)
        context.response = (
            f"[stub response] Would have called {model} "
            f"(max_tokens={self.max_tokens}, temperature={self.temperature}) "
            f"with prompt template {self.prompt_path!r} "
            f"and {n_docs} retrieved doc(s). "
            f"Request was: {context.request!r}"
        )
        return context


class PromptStrategiesTechnique(BaseTechnique):
    name = "prompt_strategies"
    variants = ("direct",)

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(
                f"prompt_strategies: unknown variant {variant!r}; expected one of {self.variants}"
            )
        return _DirectStub(knobs)
