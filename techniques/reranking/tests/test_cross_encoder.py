"""Tests for the reranking/cross_encoder stage."""

from __future__ import annotations

import pytest

from runtime.protocols import Context, Document
from techniques.reranking.impl import (
    RerankingTechnique,
    _CrossEncoderPool,
    _CrossEncoderReranker,
)


@pytest.fixture(autouse=True)
def reset_pool():
    _CrossEncoderPool.reset()
    yield
    _CrossEncoderPool.reset()


class _FakeCE:
    """Predict score = number of shared lowercase tokens between query+doc."""

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q_tokens = set(query.lower().split())
            d_tokens = set(doc.lower().split())
            scores.append(float(len(q_tokens & d_tokens)))
        return scores


def _inject_ce(monkeypatch, model=None):
    """Bypass sentence-transformers; force a fake CrossEncoder."""
    ce = model or _FakeCE()
    monkeypatch.setattr(_CrossEncoderPool, "_model", ce)
    monkeypatch.setattr(_CrossEncoderPool, "_model_name", "fake")
    monkeypatch.setattr(_CrossEncoderPool, "_load_failed", False)


def _docs(*texts) -> list[Document]:
    return [Document(content=t, score=0.5, metadata={}) for t in texts]


def test_cross_encoder_reorders_by_relevance(monkeypatch):
    """Most-relevant doc moves to the top regardless of retrieve order."""
    _inject_ce(monkeypatch)
    stage = _CrossEncoderReranker(knobs={"top_n": 3, "score_threshold": 0.5, "model": "fake"})

    ctx = Context(
        request="how do refunds work",
        documents=_docs(
            "Random unrelated content about cats.",
            "Refunds work by returning the item and waiting.",
            "Refund policy: full refund within 7 days of return.",
        ),
    )
    stage.execute(ctx)
    assert len(ctx.documents) <= 3
    assert "refund" in ctx.documents[0].content.lower()
    assert ctx.documents[0].metadata["reranker_variant"] == "cross_encoder"
    assert ctx.documents[0].metadata["reranker_model"] == "fake"


def test_cross_encoder_applies_threshold(monkeypatch):
    """Docs scoring below the threshold are dropped."""
    _inject_ce(monkeypatch)
    stage = _CrossEncoderReranker(knobs={"top_n": 5, "score_threshold": 2.0, "model": "fake"})

    ctx = Context(
        request="refund policy",
        documents=_docs(
            "refund policy customers get money back",  # 2 shared (refund, policy)
            "unrelated text about something else",      # 0 shared
            "refund period is 14 days",                 # 1 shared (refund)
        ),
    )
    stage.execute(ctx)
    # Only doc 0 should survive threshold=2
    assert len(ctx.documents) == 1
    assert "customers" in ctx.documents[0].content.lower()


def test_cross_encoder_top_n(monkeypatch):
    """top_n caps result count even when many pass threshold."""
    _inject_ce(monkeypatch)
    stage = _CrossEncoderReranker(knobs={"top_n": 2, "score_threshold": 0.0, "model": "fake"})
    ctx = Context(
        request="alpha beta",
        documents=_docs("alpha", "beta", "alpha beta", "alpha beta gamma"),
    )
    stage.execute(ctx)
    assert len(ctx.documents) == 2


def test_empty_docs_returns_empty(monkeypatch):
    """No docs in → no docs out, no model call."""
    _inject_ce(monkeypatch)
    stage = _CrossEncoderReranker(knobs={"top_n": 4, "score_threshold": 0.0, "model": "fake"})
    ctx = Context(request="x", documents=[])
    stage.execute(ctx)
    assert ctx.documents == []


def test_model_load_failure_falls_back_to_stub(monkeypatch):
    """If sentence-transformers can't load, we keep the retrieve order
    truncated to top_n and mark fallback in metadata."""
    monkeypatch.setattr(_CrossEncoderPool, "_load_failed", True)
    stage = _CrossEncoderReranker(knobs={"top_n": 2, "score_threshold": 0.0, "model": "fake"})
    ctx = Context(
        request="x",
        documents=_docs("a", "b", "c", "d"),
    )
    stage.execute(ctx)
    assert len(ctx.documents) == 2
    assert ctx.documents[0].content == "a"  # original order preserved
    assert ctx.documents[0].metadata["reranker_fallback"] == "stub"


def test_predict_exception_falls_back_to_stub(monkeypatch):
    """A runtime predict() exception falls back to stub behavior."""
    class _Boom:
        def predict(self, pairs): raise RuntimeError("oom")
    _inject_ce(monkeypatch, model=_Boom())
    stage = _CrossEncoderReranker(knobs={"top_n": 2, "score_threshold": 0.0, "model": "fake"})
    ctx = Context(request="x", documents=_docs("a", "b", "c"))
    stage.execute(ctx)
    assert len(ctx.documents) == 2
    assert ctx.documents[0].metadata["reranker_fallback"] == "stub"


def test_default_model_aliases(monkeypatch):
    """Knob model='bge-reranker-v2' (legacy default in retrieve.yaml) maps to ms-marco."""
    _inject_ce(monkeypatch)
    stage = _CrossEncoderReranker(knobs={"model": "bge-reranker-v2"})
    assert stage.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_technique_compile():
    """RerankingTechnique compiles to _CrossEncoderReranker."""
    stage = RerankingTechnique().compile("cross_encoder", knobs={"top_n": 4})
    assert isinstance(stage, _CrossEncoderReranker)
