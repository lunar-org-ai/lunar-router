"""rag technique — STUB implementations.

Phase 1.4 stub: returns canned documents without hitting any vector store.
Phase 1.10 will replace these with real FAISS-backed retrievers.
"""

from __future__ import annotations

from typing import Any

from runtime.protocols import BaseTechnique, Context, Document, Stage


class _StubRetriever:
    """Returns k canned docs whose content echoes the request."""

    def __init__(self, knobs: dict[str, Any], variant: str) -> None:
        self.k: int = int(knobs.get("k", 8))
        self.chunk_size: int = int(knobs.get("chunk_size", 512))
        self.variant = variant

    def execute(self, context: Context) -> Context:
        context.documents = [
            Document(
                content=f"[stub doc {i}] (variant={self.variant}) relevant to: {context.request[:80]}",
                score=round(1.0 - 0.05 * i, 3),
                metadata={"variant": self.variant, "rank": i, "stub": True},
            )
            for i in range(self.k)
        ]
        return context


class RagTechnique(BaseTechnique):
    name = "rag"
    variants = ("dense", "sparse", "hybrid")

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(f"rag: unknown variant {variant!r}; expected one of {self.variants}")
        return _StubRetriever(knobs, variant)
