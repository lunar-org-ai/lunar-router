"""reranking technique — STUB implementations.

Phase 1.4 stub: keeps the first top_n documents and marks them as reranked.
A real cross_encoder variant arrives in a later phase.
"""

from __future__ import annotations

from typing import Any

from runtime.protocols import BaseTechnique, Context, Stage


class _StubReranker:
    def __init__(self, knobs: dict[str, Any], variant: str) -> None:
        self.top_n: int = int(knobs.get("top_n", 4))
        self.score_threshold: float = float(knobs.get("score_threshold", 0.0))
        self.variant = variant

    def execute(self, context: Context) -> Context:
        kept = [d for d in context.documents if d.score >= self.score_threshold][: self.top_n]
        for d in kept:
            d.metadata["reranked"] = True
            d.metadata["reranker_variant"] = self.variant
        context.documents = kept
        return context


class RerankingTechnique(BaseTechnique):
    name = "reranking"
    variants = ("cross_encoder",)

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(
                f"reranking: unknown variant {variant!r}; expected one of {self.variants}"
            )
        return _StubReranker(knobs, variant)
