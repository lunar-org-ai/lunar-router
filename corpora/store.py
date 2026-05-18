"""CorpusStore — load + query a persisted vector index.

The index lives in ``corpora/indexed/``:

  - ``index.faiss`` — FAISS IndexFlatIP. Vectors are L2-normalized so
    inner-product == cosine similarity. We use Flat because corpora are
    typically small (10k chunks fits in memory easily) and Flat is
    exact + has no training step.
  - ``manifest.jsonl`` — one row per chunk: ``{id, text, source,
    chunk_index, n_chunks, metadata}``. Loaded into a list so the index
    row position == manifest row position.

Cold-start safety: if either file is missing, ``CorpusStore.load()``
returns a store with ``empty=True``. ``query()`` on an empty store
returns ``[]`` — the rag/dense stage logs a warning and the pipeline
proceeds with zero docs (generate stage handles that gracefully).

FAISS fallback: if faiss-cpu is not installed, we fall back to a
pure-numpy brute-force cosine search. Same shape, just slower past a
few thousand vectors. Lets the repo work with the base ``uv sync`` and
upgrade with ``uv sync --extra rag`` when corpora grow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


logger = logging.getLogger("corpora.store")

_DEFAULT_ROOT = Path("corpora") / "indexed"
INDEX_FILENAME = "index.faiss"
MANIFEST_FILENAME = "manifest.jsonl"
VECTORS_FILENAME = "vectors.npy"  # numpy fallback path


def _corpus_root(root: Optional[Path]) -> Path:
    """Pick the effective root. Explicit ``root`` always wins (tests).

    OSS mode → legacy ``corpora/indexed/`` at project root.
    Infra mode (``OPENTRACY_MULTI_TENANT=1``) → ``tenants/<active>/corpora/indexed/``.
    """
    if root is not None:
        return Path(root)
    from runtime.tenants.feature import is_multi_tenant_enabled
    if not is_multi_tenant_enabled():
        return _DEFAULT_ROOT
    from runtime.tenant_context import get_active as _get_tenant
    from runtime.tenants.registry import get_tenant_dir
    return get_tenant_dir(_get_tenant()) / "corpora" / "indexed"


@dataclass
class CorpusHit:
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any]


class CorpusStore:
    """In-memory handle on the persisted corpus index."""

    def __init__(
        self,
        *,
        index: Optional[Any],          # faiss.Index or None (numpy fallback)
        vectors: Optional[np.ndarray], # populated when faiss missing
        manifest: list[dict[str, Any]],
        dimension: int,
    ) -> None:
        self._index = index
        self._vectors = vectors
        self._manifest = manifest
        self._dim = dimension

    @property
    def empty(self) -> bool:
        return len(self._manifest) == 0

    @property
    def size(self) -> int:
        return len(self._manifest)

    @property
    def dimension(self) -> int:
        return self._dim

    def query(self, vector: np.ndarray, k: int = 8) -> list[CorpusHit]:
        """Find the top-k chunks by cosine similarity.

        ``vector`` must be a 1-D numpy array of length ``self.dimension``;
        we L2-normalize it before searching. Returns at most ``min(k,
        size)`` hits, ordered best-first. Empty store → ``[]``.
        """
        if self.empty:
            return []
        q = _normalize(vector.astype(np.float32, copy=False).reshape(1, -1))
        kk = min(int(k), self.size)

        if self._index is not None:
            scores, ids = self._index.search(q, kk)
            scores = scores[0].tolist()
            ids = ids[0].tolist()
        else:
            sims = (self._vectors @ q.T).ravel()
            ids = np.argsort(-sims)[:kk].tolist()
            scores = [float(sims[i]) for i in ids]

        hits: list[CorpusHit] = []
        for rank, (idx, score) in enumerate(zip(ids, scores)):
            if idx < 0 or idx >= len(self._manifest):
                continue
            row = self._manifest[idx]
            hits.append(
                CorpusHit(
                    chunk_id=row["id"],
                    text=row["text"],
                    source=row.get("source", ""),
                    score=float(score),
                    metadata={
                        **row.get("metadata", {}),
                        "rank": rank,
                        "chunk_index": row.get("chunk_index", 0),
                        "n_chunks": row.get("n_chunks", 1),
                    },
                )
            )
        return hits

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "CorpusStore":
        """Open the persisted index. Returns an empty store if missing."""
        root = _corpus_root(root)
        manifest_path = root / MANIFEST_FILENAME
        index_path = root / INDEX_FILENAME
        vectors_path = root / VECTORS_FILENAME

        if not manifest_path.is_file():
            logger.info("corpus manifest missing at %s — empty store", manifest_path)
            return cls(index=None, vectors=None, manifest=[], dimension=0)

        manifest = _read_jsonl(manifest_path)
        if not manifest:
            return cls(index=None, vectors=None, manifest=[], dimension=0)

        if index_path.is_file():
            try:
                import faiss
                idx = faiss.read_index(str(index_path))
                return cls(
                    index=idx, vectors=None, manifest=manifest, dimension=idx.d,
                )
            except Exception as e:
                logger.warning("faiss load failed (%s) — falling back to numpy", e)

        # numpy fallback path
        if vectors_path.is_file():
            vectors = np.load(vectors_path)
            return cls(
                index=None,
                vectors=vectors,
                manifest=manifest,
                dimension=int(vectors.shape[1]),
            )

        logger.warning(
            "neither %s nor %s present — corpus manifest exists but vectors missing",
            index_path, vectors_path,
        )
        return cls(index=None, vectors=None, manifest=[], dimension=0)


def save_index(
    *,
    vectors: np.ndarray,
    manifest: list[dict[str, Any]],
    root: Optional[Path] = None,
) -> Path:
    """Persist a freshly-built index. Writes index.faiss when faiss is
    available; otherwise falls back to vectors.npy. Always writes the
    manifest. Returns the root directory."""
    root = Path(root) if root is not None else _DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    vectors = _normalize(vectors.astype(np.float32, copy=False))

    manifest_path = root / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        import faiss
        idx = faiss.IndexFlatIP(int(vectors.shape[1]))
        idx.add(vectors)
        faiss.write_index(idx, str(root / INDEX_FILENAME))
        # Clean up any leftover numpy fallback
        npy = root / VECTORS_FILENAME
        if npy.is_file():
            npy.unlink()
    except ImportError:
        logger.info("faiss not installed — writing numpy fallback")
        np.save(root / VECTORS_FILENAME, vectors)

    return root


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
