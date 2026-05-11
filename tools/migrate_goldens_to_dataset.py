"""One-shot migration — evals/golden/*.yaml → datasets/<name>/v1.json.

The existing eval suite goldens live as individual YAML files under
``evals/golden/``. The P15.4 dataset backend stores datasets as a single
versioned JSON artifact per name (``datasets/<name>/v<n>.json``) with
each sample carrying its prompt, ground_truth, tag, and MiniLM
embedding.

This script does the one-time port. It also emits a ``v0.json`` schema
placeholder mirroring the ``router_config_v0.json`` convention, then
writes the real data as ``v1.json`` and flips the ``current`` pointer.

Usage:

    python -m tools.migrate_goldens_to_dataset
    python -m tools.migrate_goldens_to_dataset --name goldens --dry-run
    python -m tools.migrate_goldens_to_dataset --goldens-dir evals/golden \\
        --datasets-dir datasets

Idempotent: re-running produces byte-identical output when the embedder
+ tokenizer are unchanged (sample IDs hash from prompt+tag only).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from router.data.dataset import DatasetMetadata, DatasetSample
from router.data.dataset_io import (
    DEFAULT_DATASETS_DIR,
    get_current_version,
    now_iso,
    save_dataset,
)


logger = logging.getLogger("tools.migrate_goldens_to_dataset")


# A fixed timestamp keeps the migration output deterministic across runs.
# (Use a real time in the history entry but not in per-sample added_at —
# `added_at` is part of the JSON and would otherwise break byte-identity.)
_EPOCH_ADDED_AT = "1970-01-01T00:00:00Z"

DEFAULT_DATASET_NAME = "goldens"
DEFAULT_GOLDENS_DIR = Path("evals/golden")
DEFAULT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sample_id(prompt: str, tag: Optional[str]) -> str:
    """Stable short hash of prompt+tag. Determines idempotency of the migration."""
    key = f"{prompt}\0{tag or ''}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:12]
    return f"smp_{digest}"


def _load_goldens(goldens_dir: Path) -> list[dict]:
    """Read all *.yaml files in `goldens_dir` and return their parsed dicts, sorted by id."""
    if not goldens_dir.exists():
        raise FileNotFoundError(f"goldens dir not found: {goldens_dir}")
    files = sorted(goldens_dir.glob("*.yaml"))
    if not files:
        raise RuntimeError(f"no *.yaml files in {goldens_dir}")
    out = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "input" not in data or "request" not in data["input"]:
            logger.warning("skipping %s — missing input.request", path)
            continue
        out.append(data)
    out.sort(key=lambda d: d.get("id", ""))
    return out


def _ground_truth_from_expected(expected: dict) -> str:
    """Best-effort: prefer 'exact', then first 'contains' token, else empty."""
    if not expected:
        return ""
    if "exact" in expected:
        return str(expected["exact"])
    contains = expected.get("contains") or []
    if isinstance(contains, list) and contains:
        return str(contains[0])
    return ""


def _build_samples(
    goldens: list[dict],
    embedder,
) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    for g in goldens:
        prompt = str(g["input"]["request"])
        expected = g.get("expected") or {}
        ground_truth = _ground_truth_from_expected(expected)
        tag = expected.get("category")
        embedding = embedder.embed(prompt)
        # Normalize to a plain Python list[float] for JSON serialization.
        try:
            embedding_list = embedding.tolist()  # numpy ndarray
        except AttributeError:
            embedding_list = list(embedding)
        samples.append(
            DatasetSample(
                id=_sample_id(prompt, tag),
                prompt=prompt,
                ground_truth=ground_truth,
                tag=tag,
                trace_id=None,
                added_at=_EPOCH_ADDED_AT,
                source="manual",
                embedding=embedding_list,
            )
        )
    return samples


def _v0_placeholder_payload(name: str, embedder_model: str, embedding_dim: int) -> dict:
    """Schema-doc placeholder. Mirrors router_config_v0.json convention."""
    return {
        "version": 0,
        "name": name,
        "desc": "Schema documentation placeholder. v1 is the first real artifact.",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "created_at": _EPOCH_ADDED_AT,
        "embedder_model": embedder_model,
        "embedding_dim": embedding_dim,
        "samples": [],
        "history": [],
        "metadata": {
            "phase": "P15.4.1",
            "note": "Schema placeholder. v1 holds the migrated goldens.",
        },
    }


def _build_payload(
    name: str,
    samples: list[DatasetSample],
    embedder_model: str,
    embedding_dim: int,
    *,
    version: int,
    migration_source: str,
) -> dict:
    return {
        "version": version,
        "name": name,
        "desc": "Eval suite goldens. Migrated from evals/golden/*.yaml.",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "created_at": _EPOCH_ADDED_AT,
        "embedder_model": embedder_model,
        "embedding_dim": embedding_dim,
        "samples": [
            {
                "id": s.id,
                "prompt": s.prompt,
                "ground_truth": s.ground_truth,
                "tag": s.tag,
                "trace_id": s.trace_id,
                "added_at": s.added_at,
                "source": s.source,
                "embedding": s.embedding,
            }
            for s in samples
        ],
        "history": [
            {
                "when": _EPOCH_ADDED_AT,
                "what": f"Migrated from {migration_source} ({len(samples)} entries).",
            }
        ],
        "metadata": {
            "phase": "P15.4.1",
            "migration_source": migration_source,
        },
    }


def _get_embedder():
    """Return a warm PromptEmbedder. Late-imported so --dry-run without
    embeddings stays fast."""
    from runtime.embedder_pool import get_pool

    pool = get_pool()
    return pool.get()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--name",
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset name (default: {DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--goldens-dir",
        default=str(DEFAULT_GOLDENS_DIR),
        help=f"Source directory of *.yaml goldens (default: {DEFAULT_GOLDENS_DIR}).",
    )
    parser.add_argument(
        "--datasets-dir",
        default=str(DEFAULT_DATASETS_DIR),
        help=f"Target datasets directory (default: {DEFAULT_DATASETS_DIR}).",
    )
    parser.add_argument(
        "--embedder-model",
        default=DEFAULT_EMBEDDER_MODEL,
        help="Embedder model id (must match the one the runtime/embedder_pool uses).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read inputs, compute samples, but don't write anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing v1 artifact (default: abort if v1 exists).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    goldens_dir = Path(args.goldens_dir)
    datasets_dir = Path(args.datasets_dir)
    name = args.name

    logger.info("loading goldens from %s", goldens_dir)
    goldens = _load_goldens(goldens_dir)
    logger.info("loaded %d golden(s)", len(goldens))

    # Refuse to clobber unless --force.
    if not args.force and not args.dry_run:
        existing = get_current_version(name, datasets_dir=datasets_dir)
        if existing is not None and existing >= 1:
            print(
                f"refusing to overwrite existing dataset {name!r} at "
                f"v{existing}. Re-run with --force to clobber.",
                file=sys.stderr,
            )
            return 2

    logger.info("loading embedder (%s)…", args.embedder_model)
    embedder = _get_embedder()
    samples = _build_samples(goldens, embedder)

    # Determine embedding_dim from the first sample (all are produced by the
    # same model so they're all equal).
    if not samples:
        print("no samples produced — aborting", file=sys.stderr)
        return 1
    embedding_dim = len(samples[0].embedding)

    v0 = _v0_placeholder_payload(name, args.embedder_model, embedding_dim)
    v1 = _build_payload(
        name,
        samples,
        args.embedder_model,
        embedding_dim,
        version=1,
        migration_source=str(goldens_dir),
    )

    if args.dry_run:
        print(f"[dry-run] would write {len(samples)} samples to "
              f"{datasets_dir}/{name}/v1.json")
        print(f"[dry-run] sample IDs: {[s.id for s in samples[:3]]}…")
        print(f"[dry-run] embedding_dim = {embedding_dim}")
        return 0

    # Write v0 (placeholder) WITHOUT flipping pointer; then write v1 + flip.
    save_dataset(v0, datasets_dir=datasets_dir, update_pointer=False)
    json_path = save_dataset(v1, datasets_dir=datasets_dir, update_pointer=True)
    print(f"wrote {json_path} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
