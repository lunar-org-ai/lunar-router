"""Ingest .md / .txt files into the corpus vector index.

CLI::

    uv run python -m corpora.ingest <path> [--chunk-size 512] [--overlap 50]

``<path>`` is a file or a directory. Directories are walked recursively
for ``.md`` and ``.txt`` files (skipping hidden dirs). Each file is
split into overlapping word-windows, embedded with the runtime's
shared PromptEmbedder (cached, MiniLM-L6 by default), and written to
``corpora/indexed/``. The build is from-scratch on each run — simple
and deterministic. Files in ``corpora/ingested/`` are moved aside or
left as-is depending on ``--copy-into-ingested``.

Behavior is idempotent under the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


logger = logging.getLogger("corpora.ingest")

_TEXT_EXTS = {".md", ".txt"}
_DEFAULT_INPUT = Path("corpora") / "ingested"


@dataclass
class _Chunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    n_chunks: int


def ingest(
    paths: Iterable[Path] | Path,
    *,
    chunk_size: int = 512,
    overlap: int = 50,
    root: Optional[Path] = None,
    embedder=None,
) -> dict:
    """Build the corpus index. Returns a summary dict."""
    from corpora.store import save_index

    if isinstance(paths, Path):
        paths = [paths]
    files = sorted(_collect_files(paths))
    if not files:
        logger.warning("no .md/.txt files found under inputs")
        return {"files": 0, "chunks": 0, "root": str(root or "")}

    chunks: list[_Chunk] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("skipping non-utf8 file: %s", f)
            continue
        rel = str(f)
        pieces = _split(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            cid = _chunk_id(rel, i, piece)
            chunks.append(
                _Chunk(
                    chunk_id=cid,
                    text=piece,
                    source=rel,
                    chunk_index=i,
                    n_chunks=len(pieces),
                )
            )

    if not chunks:
        logger.warning("no chunks produced")
        return {"files": len(files), "chunks": 0, "root": str(root or "")}

    if embedder is None:
        from runtime.embedder_pool import get_pool
        embedder = get_pool().get()

    texts = [c.text for c in chunks]
    logger.info("embedding %d chunk(s) from %d file(s)", len(texts), len(files))
    vectors = embedder.embed_batch(texts)
    if not isinstance(vectors, np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {vectors.shape}")

    manifest = [
        {
            "id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "n_chunks": c.n_chunks,
            "metadata": {},
        }
        for c in chunks
    ]
    out_root = save_index(vectors=vectors, manifest=manifest, root=root)
    return {
        "files": len(files),
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "root": str(out_root),
    }


def _collect_files(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in _TEXT_EXTS:
            out.append(p)
            continue
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and sub.suffix.lower() in _TEXT_EXTS:
                    if any(part.startswith(".") for part in sub.parts):
                        continue
                    out.append(sub)
    return out


def _split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split into overlapping word-windows. Empty text → []."""
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        return [" ".join(words)]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)
    step = max(1, chunk_size - overlap)
    out: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        out.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return out


def _chunk_id(source: str, index: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{index}|{text[:160]}".encode("utf-8")).hexdigest()[:12]
    return f"c_{h}"


def _cli_main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="corpora.ingest",
        description="Ingest .md/.txt files into the corpus vector index",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(_DEFAULT_INPUT),
        help=f"Path to a file or directory (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--root",
        default=None,
        help="Where to write the index (default: corpora/indexed/)",
    )
    args = parser.parse_args(argv)

    summary = ingest(
        Path(args.path),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        root=Path(args.root) if args.root else None,
    )
    print(
        f"indexed {summary['chunks']} chunk(s) from {summary['files']} file(s) "
        f"→ {summary['root']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
