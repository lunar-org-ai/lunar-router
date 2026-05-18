"""reranking technique — re-score retrieved docs with a cross-encoder.

``cross_encoder`` variant — P1.10. Loads a sentence-transformers
CrossEncoder once (process-singleton, cached after first build) and
scores ``(request, doc.content)`` pairs. Docs below
``score_threshold`` are dropped; the rest are sorted descending and
truncated to ``top_n``.

If the cross-encoder model can't load (offline, dep missing, network
error), the stage falls back to the previous stub behavior: keep the
order from retrieve, top_n only. Same shape, same contract, no crash.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from runtime.protocols import BaseTechnique, Context, Stage


logger = logging.getLogger("techniques.reranking")

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class _CrossEncoderPool:
    """Process-wide cache for the CrossEncoder. ~30MB model; load once."""

    _model: Optional[Any] = None
    _model_name: Optional[str] = None
    _lock = threading.Lock()
    _load_failed: bool = False

    @classmethod
    def get(cls, model_name: str) -> Optional[Any]:
        if cls._load_failed:
            return None
        if cls._model is not None and cls._model_name == model_name:
            return cls._model
        with cls._lock:
            if cls._model is not None and cls._model_name == model_name:
                return cls._model
            try:
                from sentence_transformers import CrossEncoder
                logger.info("loading CrossEncoder %s (first call, ~30MB)", model_name)
                cls._model = CrossEncoder(model_name)
                cls._model_name = model_name
                return cls._model
            except Exception as e:
                logger.warning("CrossEncoder load failed (%s) — falling back to stub", e)
                cls._load_failed = True
                return None

    @classmethod
    def reset(cls) -> None:  # for tests
        with cls._lock:
            cls._model = None
            cls._model_name = None
            cls._load_failed = False


class _CrossEncoderReranker:
    def __init__(self, knobs: dict[str, Any]) -> None:
        self.top_n: int = int(knobs.get("top_n", 4))
        self.score_threshold: float = float(knobs.get("score_threshold", 0.0))
        self.model_name: str = str(knobs.get("model", _DEFAULT_MODEL))
        if self.model_name in {"bge-reranker-v2", "default"}:
            self.model_name = _DEFAULT_MODEL

    def execute(self, context: Context) -> Context:
        docs = context.documents
        if not docs:
            return context

        model = _CrossEncoderPool.get(self.model_name)
        if model is None:
            return self._stub_fallback(context)

        try:
            pairs = [(context.request, d.content) for d in docs]
            scores = model.predict(pairs)
        except Exception as e:
            logger.warning("cross-encoder predict failed (%s) — stub fallback", e)
            return self._stub_fallback(context)

        scored = list(zip(docs, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        kept = []
        for d, s in scored:
            if s < self.score_threshold:
                continue
            d.score = s
            d.metadata["reranked"] = True
            d.metadata["reranker_variant"] = "cross_encoder"
            d.metadata["reranker_model"] = self.model_name
            kept.append(d)
            if len(kept) >= self.top_n:
                break

        context.documents = kept
        return context

    def _stub_fallback(self, context: Context) -> Context:
        kept = [d for d in context.documents if d.score >= self.score_threshold][: self.top_n]
        for d in kept:
            d.metadata["reranked"] = True
            d.metadata["reranker_variant"] = "cross_encoder"
            d.metadata["reranker_fallback"] = "stub"
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
        return _CrossEncoderReranker(knobs)
