"""rag technique — retrieve documents for the generate stage.

Variants:

  - ``dense`` — P1.10. Embeds the request with the shared PromptEmbedder
    and queries the persisted corpus index (``corpora/indexed/``).
    Cold-start safe: empty corpus → returns 0 docs, no crash; the
    generate stage answers from the LLM's own knowledge.
  - ``sparse`` and ``hybrid`` — still stubs (BM25 + fusion lands in
    P1.10.x). Wired so the pipeline can opt into them when the index
    grows enough to make sparse worthwhile.

Both stub variants and the real dense variant carry the same Document
shape, so the rerank stage doesn't care which retriever produced a doc.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from runtime.protocols import BaseTechnique, Context, Document, Stage


logger = logging.getLogger("techniques.rag")


class _StubRetriever:
    """Returns k canned docs whose content echoes the request.

    Kept for ``sparse`` and ``hybrid`` until those land. Marked
    ``stub=True`` so traces stay honest about what's real.
    """

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


class _DenseRetriever:
    """Embed the request, top-k against ``corpora/indexed/``."""

    def __init__(self, knobs: dict[str, Any]) -> None:
        self.k: int = int(knobs.get("k", 8))
        self._embedder: Optional[Any] = None
        self._store: Optional[Any] = None
        self._store_attempted = False

    def execute(self, context: Context) -> Context:
        store = self._load_store()
        if store is None or store.empty:
            context.documents = []
            return context

        embedder = self._load_embedder()
        if embedder is None:
            context.documents = []
            return context

        try:
            vec = embedder.embed(context.request)
        except Exception as e:
            logger.warning("embed failed (%s) — returning 0 docs", e)
            context.documents = []
            return context

        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=np.float32)
        hits = store.query(vec, k=self.k)
        context.documents = [
            Document(
                content=hit.text,
                score=hit.score,
                metadata={
                    "variant": "dense",
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    **hit.metadata,
                },
            )
            for hit in hits
        ]
        return context

    def _load_store(self):
        if self._store is not None:
            return self._store
        if self._store_attempted:
            return None
        self._store_attempted = True
        try:
            from corpora.store import CorpusStore
            self._store = CorpusStore.load()
        except Exception as e:
            logger.warning("CorpusStore.load failed (%s) — empty retrieve", e)
            self._store = None
        return self._store

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from runtime.embedder_pool import get_pool
            self._embedder = get_pool().get()
        except Exception as e:
            logger.warning("embedder load failed (%s)", e)
            self._embedder = None
        return self._embedder


class RagTechnique(BaseTechnique):
    name = "rag"
    variants = ("dense", "sparse", "hybrid")

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(f"rag: unknown variant {variant!r}; expected one of {self.variants}")
        if variant == "dense":
            return _DenseRetriever(knobs)
        return _StubRetriever(knobs, variant)
