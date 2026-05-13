"""End-to-end tests for corpora.ingest + corpora.store.

Uses a tiny mock embedder so tests stay offline + deterministic. The
real PromptEmbedder is exercised separately in the runtime pipeline
test which actually loads MiniLM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corpora.ingest import ingest
from corpora.store import CorpusStore, save_index


class _BagOfWordsEmbedder:
    """Deterministic mini-embedder: hash bag-of-words into fixed dims.

    Two texts sharing many tokens get similar vectors; unrelated texts
    diverge. Enough for ranking-correctness tests; no ML dependency.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok) % self.dim] += 1.0
        return v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)


def _write_docs(root: Path, docs: dict[str, str]) -> None:
    for name, content in docs.items():
        (root / name).write_text(content, encoding="utf-8")


def test_ingest_round_trip(tmp_path):
    """Ingest a folder → load store → query returns relevant docs first."""
    src = tmp_path / "src"
    src.mkdir()
    _write_docs(
        src,
        {
            "refund.md": "Refunds are processed within 7 days of return receipt.",
            "shipping.md": "Shipping to Brazil takes 8-14 business days via DHL.",
            "cancel.md": "You can cancel an order before it ships from the warehouse.",
        },
    )

    idx_root = tmp_path / "indexed"
    embedder = _BagOfWordsEmbedder()
    summary = ingest(src, chunk_size=50, overlap=5, root=idx_root, embedder=embedder)

    assert summary["files"] == 3
    assert summary["chunks"] >= 3
    assert summary["dim"] == 64

    store = CorpusStore.load(idx_root)
    assert not store.empty
    assert store.size == summary["chunks"]
    assert store.dimension == 64

    # Query about refunds should pull the refund doc first.
    qvec = embedder.embed("how do refunds work")
    hits = store.query(qvec, k=3)
    assert hits, "expected at least one hit"
    assert "refund" in hits[0].text.lower() or "refund" in hits[0].source.lower()


def test_empty_corpus_cold_start_safe():
    """No index files → load returns empty store, query returns []."""
    store = CorpusStore.load(Path("/nonexistent/never/created"))
    assert store.empty
    assert store.size == 0
    assert store.dimension == 0
    hits = store.query(np.zeros(64, dtype=np.float32), k=5)
    assert hits == []


def test_manifest_without_vectors_is_empty(tmp_path):
    """Manifest present but no faiss/vectors → treated as empty (no crash)."""
    (tmp_path / "manifest.jsonl").write_text(
        '{"id":"c_1","text":"hi","source":"x","chunk_index":0,"n_chunks":1,"metadata":{}}\n'
    )
    store = CorpusStore.load(tmp_path)
    assert store.empty
    hits = store.query(np.zeros(64, dtype=np.float32), k=3)
    assert hits == []


def test_chunking_overlap(tmp_path):
    """Long doc → multiple chunks with overlap."""
    src = tmp_path / "long.md"
    words = ["word%d" % i for i in range(120)]
    src.write_text(" ".join(words))

    embedder = _BagOfWordsEmbedder()
    summary = ingest(src, chunk_size=30, overlap=5, root=tmp_path / "idx", embedder=embedder)
    # 120 words, step=25 → ceil(120/25) = 5 chunks
    assert summary["chunks"] >= 4
    assert summary["chunks"] <= 6


def test_unicode_doc(tmp_path):
    """utf-8 content (Portuguese accents) round-trips through the manifest."""
    src = tmp_path / "pt.md"
    src.write_text("Política de reembolso: até 30 dias após a compra.", encoding="utf-8")
    embedder = _BagOfWordsEmbedder()
    ingest(src, chunk_size=20, overlap=2, root=tmp_path / "idx", embedder=embedder)
    store = CorpusStore.load(tmp_path / "idx")
    hits = store.query(embedder.embed("reembolso"), k=1)
    assert hits
    assert "Política" in hits[0].text


def test_numpy_fallback_when_no_faiss(tmp_path, monkeypatch):
    """save_index falls back to vectors.npy when faiss isn't available.
    Mocking ImportError on faiss exercises that path."""
    import builtins

    real_import = builtins.__import__
    def _fail_faiss(name, *a, **kw):
        if name == "faiss":
            raise ImportError("faiss not available in this test")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _fail_faiss)

    embedder = _BagOfWordsEmbedder()
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("Refund within 14 days of delivery.")
    summary = ingest(src, chunk_size=10, overlap=1, root=tmp_path / "idx", embedder=embedder)
    assert summary["chunks"] >= 1

    # Vectors path is what got written
    assert (tmp_path / "idx" / "vectors.npy").is_file()
    assert not (tmp_path / "idx" / "index.faiss").is_file()

    # And the store still loads + queries
    store = CorpusStore.load(tmp_path / "idx")
    assert not store.empty
    hits = store.query(embedder.embed("refund"), k=1)
    assert hits


def test_ingest_skips_unknown_extensions(tmp_path):
    """Only .md/.txt are ingested; other files are ignored."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "good.md").write_text("Take this content.")
    (src / "good.txt").write_text("Also this.")
    (src / "image.png").write_bytes(b"\x89PNG\r\n")
    (src / "code.py").write_text("print('hi')")

    embedder = _BagOfWordsEmbedder()
    summary = ingest(src, chunk_size=20, overlap=2, root=tmp_path / "idx", embedder=embedder)
    assert summary["files"] == 2
