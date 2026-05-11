"""datasets/_registry.json — single source of truth for "what datasets exist".

Every `save_dataset()` call from dataset_io flows through `sync_from_payload()`
to keep the registry in sync. The registry only stores summary metadata; the
heavy data (samples, embeddings, history) lives in datasets/<name>/v<n>.json.

Schema:
    {
      "datasets": {
        "<name>": {
          "current_version": <int>,
          "use": ["Eval", ...],
          "owner": "agent" | "human",
          "sourceType": "auto" | "manual",
          "growing": <bool>,
          "size": <int>,
          "desc": "...",
          "embedder_model": "...",
          "embedding_dim": <int>,
          "updated_at": "<iso>"
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from router.data.dataset import DatasetMetadata
from router.data.dataset_io import (
    _STAGING_DIR,
    _vd,
    now_iso,
)


_REGISTRY_NAME = "_registry.json"


def _registry_path(*, datasets_dir: Path) -> Path:
    return datasets_dir / _REGISTRY_NAME


def _load_registry(*, datasets_dir: Path) -> dict:
    path = _registry_path(datasets_dir=datasets_dir)
    if not path.exists():
        return {"datasets": {}}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"datasets": {}}


def _save_registry(registry: dict, *, datasets_dir: Path) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    target = _registry_path(datasets_dir=datasets_dir)
    staging = datasets_dir / _STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=_REGISTRY_NAME + ".", suffix=".tmp", dir=staging
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# --- Public API ---


def list_datasets(
    *,
    datasets_dir: Optional[Path] = None,
) -> list[DatasetMetadata]:
    """Return one DatasetMetadata per registered dataset."""
    datasets_dir = _vd(datasets_dir)
    registry = _load_registry(datasets_dir=datasets_dir)
    out: list[DatasetMetadata] = []
    for name, entry in (registry.get("datasets") or {}).items():
        if entry.get("deleted"):
            continue
        out.append(_entry_to_meta(name, entry))
    return out


def get_dataset_meta(
    name: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> Optional[DatasetMetadata]:
    """Return DatasetMetadata for `name`, or None if not registered or deleted."""
    datasets_dir = _vd(datasets_dir)
    registry = _load_registry(datasets_dir=datasets_dir)
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None or entry.get("deleted"):
        return None
    return _entry_to_meta(name, entry)


def register_dataset(
    meta: DatasetMetadata,
    *,
    current_version: int,
    size: int,
    datasets_dir: Optional[Path] = None,
) -> None:
    """Add or replace the registry entry for `meta.name`."""
    datasets_dir = _vd(datasets_dir)
    registry = _load_registry(datasets_dir=datasets_dir)
    registry.setdefault("datasets", {})
    registry["datasets"][meta.name] = {
        "current_version": int(current_version),
        "use": list(meta.use),
        "owner": meta.owner,
        "sourceType": meta.sourceType,
        "growing": bool(meta.growing),
        "size": int(size),
        "desc": meta.desc,
        "embedder_model": meta.embedder_model,
        "embedding_dim": int(meta.embedding_dim),
        "updated_at": now_iso(),
    }
    _save_registry(registry, datasets_dir=datasets_dir)


def update_dataset_meta(
    name: str,
    *,
    current_version: Optional[int] = None,
    size: Optional[int] = None,
    growing: Optional[bool] = None,
    use: Optional[list[str]] = None,
    owner: Optional[str] = None,
    sourceType: Optional[str] = None,
    desc: Optional[str] = None,
    datasets_dir: Optional[Path] = None,
) -> None:
    """Partial update of an existing registry entry. No-op if name not registered."""
    datasets_dir = _vd(datasets_dir)
    registry = _load_registry(datasets_dir=datasets_dir)
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None:
        return
    if current_version is not None:
        entry["current_version"] = int(current_version)
    if size is not None:
        entry["size"] = int(size)
    if growing is not None:
        entry["growing"] = bool(growing)
    if use is not None:
        entry["use"] = list(use)
    if owner is not None:
        entry["owner"] = owner
    if sourceType is not None:
        entry["sourceType"] = sourceType
    if desc is not None:
        entry["desc"] = desc
    entry["updated_at"] = now_iso()
    _save_registry(registry, datasets_dir=datasets_dir)


def delete_dataset(
    name: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> None:
    """Soft-delete: mark the entry as deleted. Files on disk are kept."""
    datasets_dir = _vd(datasets_dir)
    registry = _load_registry(datasets_dir=datasets_dir)
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None:
        return
    entry["deleted"] = True
    entry["updated_at"] = now_iso()
    _save_registry(registry, datasets_dir=datasets_dir)


def sync_from_payload(
    payload: dict,
    *,
    datasets_dir: Optional[Path] = None,
) -> None:
    """Called by save_dataset to keep the registry in sync with the artifact."""
    meta = DatasetMetadata(
        name=payload["name"],
        desc=payload.get("desc", ""),
        source=payload.get("source", "manual"),
        sourceType=payload.get("sourceType", "manual"),
        use=list(payload.get("use", ["Eval"])),
        owner=payload.get("owner", "human"),
        growing=bool(payload.get("growing", False)),
        embedder_model=payload.get(
            "embedder_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_dim=int(payload.get("embedding_dim", 384)),
    )
    register_dataset(
        meta,
        current_version=int(payload["version"]),
        size=len(payload.get("samples") or []),
        datasets_dir=datasets_dir,
    )


# --- Internal ---


def _entry_to_meta(name: str, entry: dict) -> DatasetMetadata:
    return DatasetMetadata(
        name=name,
        desc=entry.get("desc", ""),
        source=entry.get("source", "manual"),
        sourceType=entry.get("sourceType", "manual"),
        use=list(entry.get("use", ["Eval"])),
        owner=entry.get("owner", "human"),
        growing=bool(entry.get("growing", False)),
        embedder_model=entry.get(
            "embedder_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_dim=int(entry.get("embedding_dim", 384)),
    )
