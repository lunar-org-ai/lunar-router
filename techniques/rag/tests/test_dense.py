"""Tests for the rag/dense retriever."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from runtime.protocols import Context
from techniques.rag.impl import RagTechnique, _DenseRetriever


class _BagOfWordsEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
    def embed(self, text: str) -> np.ndarray:
        import hashlib
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            bucket = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big")
            v[bucket % self.dim] += 1.0
        return v
    def embed_batch(self, texts):
        return np.stack([self.embed(t) for t in texts], axis=0)


@pytest.fixture
def small_index(tmp_path, monkeypatch):
    """Build a tiny index in tmp_path and point CorpusStore.load at it."""
    from corpora.ingest import ingest

    src = tmp_path / "src"
    src.mkdir()
    (src / "refund.md").write_text("Refund policy: full refund within 7 days of return.")
    (src / "shipping.md").write_text("Shipping to Brazil takes 8-14 business days.")
    (src / "cancel.md").write_text("Order cancellation must happen before warehouse pickup.")

    idx_root = tmp_path / "indexed"
    embedder = _BagOfWordsEmbedder()
    ingest(src, chunk_size=20, overlap=2, root=idx_root, embedder=embedder)

    monkeypatch.setattr("corpora.store._DEFAULT_ROOT", idx_root)
    return embedder


def test_dense_returns_docs_from_corpus(small_index, monkeypatch):
    """When corpus is populated, dense retrieve returns Documents with
    chunk_id + source metadata."""
    retriever = _DenseRetriever(knobs={"k": 3})
    retriever._embedder = small_index  # inject deterministic embedder
    ctx = Context(request="refund policy", documents=[])
    retriever.execute(ctx)
    assert len(ctx.documents) >= 1
    top = ctx.documents[0]
    assert "refund" in top.content.lower()
    assert top.metadata["variant"] == "dense"
    assert top.metadata["chunk_id"].startswith("c_")
    assert top.metadata["source"].endswith("refund.md")
    assert "stub" not in top.metadata


def test_dense_empty_corpus_returns_no_docs(tmp_path, monkeypatch):
    """No index → returns 0 docs, no crash. Generate stage handles
    the zero-docs case gracefully."""
    monkeypatch.setattr("corpora.store._DEFAULT_ROOT", tmp_path / "empty")
    retriever = _DenseRetriever(knobs={"k": 8})
    ctx = Context(request="anything at all", documents=[])
    retriever.execute(ctx)
    assert ctx.documents == []


def test_dense_embedder_failure_returns_no_docs(small_index, monkeypatch):
    """If the embedder raises, we log + return [] instead of bubbling."""
    class _Bad:
        dim = 64
        def embed(self, text): raise RuntimeError("embedder down")
    retriever = _DenseRetriever(knobs={"k": 3})
    retriever._embedder = _Bad()
    ctx = Context(request="x", documents=[])
    retriever.execute(ctx)
    assert ctx.documents == []


def test_sparse_and_hybrid_still_stub():
    """Non-dense variants stay stubs for now."""
    for variant in ("sparse", "hybrid"):
        stage = RagTechnique().compile(variant, knobs={"k": 4})
        ctx = Context(request="anything", documents=[])
        stage.execute(ctx)
        assert len(ctx.documents) == 4
        assert all(d.metadata.get("stub") for d in ctx.documents)


def test_dense_respects_k_knob(small_index, monkeypatch):
    """k=1 returns at most 1 doc."""
    retriever = _DenseRetriever(knobs={"k": 1})
    retriever._embedder = small_index
    ctx = Context(request="shipping brazil", documents=[])
    retriever.execute(ctx)
    assert len(ctx.documents) == 1
