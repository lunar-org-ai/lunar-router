"""Process-singleton PromptEmbedder for the runtime.

Sentence-transformers loads in 1-3s on first construction. Without
caching, every ``/run`` that uses the UniRoute variant — and every call
to ``POST /router/decide`` — would pay that cost. The pool keeps a
single embedder around, lazy-initialized on first ``get()`` and
optionally pre-warmed at agent compile.

Thread-safe: the lazy init happens under a lock so concurrent first
calls don't race. After init, ``get()`` is a fast attribute read.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional


logger = logging.getLogger("runtime.embedder_pool")


class EmbedderPool:
    """Singleton-style PromptEmbedder holder."""

    def __init__(self) -> None:
        self._embedder = None  # type: Optional[object]
        self._lock = threading.Lock()

    def get(self):
        """Return the cached PromptEmbedder, initializing on first call.

        Returns:
            ``router.core.embeddings.PromptEmbedder`` typed object. Kept
            untyped here to avoid importing sentence-transformers at
            module load (the [router] extra may not be installed in
            agent-only environments).
        """
        if self._embedder is not None:
            return self._embedder
        with self._lock:
            if self._embedder is None:
                from router.core.embeddings import (
                    PromptEmbedder,
                    SentenceTransformerProvider,
                )

                logger.info("loading PromptEmbedder (first call) — ~1-3s for MiniLM")
                provider = SentenceTransformerProvider()
                self._embedder = PromptEmbedder(provider)
        return self._embedder

    def warm(self) -> None:
        """Pre-load the model + run one no-op embed so the first ``/run``
        that needs routing doesn't pay the cold-load cost.

        Hook this into the agent compile path when ``route.variant ==
        "uniroute"`` so users on ``small_first`` don't pay it.
        """
        emb = self.get()
        try:
            _ = emb.embed("warmup")
            logger.info("embedder pool warmed")
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("embedder warmup failed (non-fatal): %s", e)


_pool: Optional[EmbedderPool] = None


def get_pool() -> EmbedderPool:
    """Return the process-wide EmbedderPool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = EmbedderPool()
    return _pool


def reset_pool() -> None:
    """For tests — clear the singleton so monkeypatched providers take effect."""
    global _pool
    _pool = None
