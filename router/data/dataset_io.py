"""dataset_<n>.json load/save + current-pointer resolution.

Disk layout:
    datasets/
      _registry.json                       (top-level index — written by registry)
      .staging/                            (atomic-write scratch)
      goldens/
        v0.json                            (schema doc placeholder)
        v1.json                            (first artifact — emitted by migration)
        current                            (symlink → vN.json)
          OR
        current.txt                        (single-line "N" — symlink fallback)

Atomicity mirrors router/config_io.py: writes go through datasets/.staging/
then os.replace into the final location. Pointer flip tries os.symlink first,
falls back to a one-line .txt file. POSIX guarantees atomicity of os.replace
within the same filesystem.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from router.data.dataset import (
    Dataset,
    DatasetMetadata,
    DatasetSample,
)
from router.errors import DatasetInvalidError, DatasetNotFoundError


DEFAULT_DATASETS_DIR = Path("datasets")

_CURRENT_NAME = "current"
_CURRENT_TXT = "current.txt"
_STAGING_DIR = ".staging"


def _vd(datasets_dir: Optional[Path]) -> Path:
    """Resolve the datasets directory at call time so monkeypatching works."""
    if datasets_dir is not None:
        return datasets_dir
    import router.data.dataset_io as _self
    return _self.DEFAULT_DATASETS_DIR


# --- Path helpers ---


def _dataset_dir(name: str, *, datasets_dir: Path) -> Path:
    return datasets_dir / name


def _dataset_path(name: str, version: int, *, datasets_dir: Path) -> Path:
    return _dataset_dir(name, datasets_dir=datasets_dir) / f"v{version}.json"


def _symlink_current_path(name: str, *, datasets_dir: Path) -> Path:
    return _dataset_dir(name, datasets_dir=datasets_dir) / _CURRENT_NAME


def _txt_current_path(name: str, *, datasets_dir: Path) -> Path:
    return _dataset_dir(name, datasets_dir=datasets_dir) / _CURRENT_TXT


def _parse_version_from_filename(filename: str) -> Optional[int]:
    """`v3.json` → 3; anything else → None."""
    if not filename.startswith("v") or not filename.endswith(".json"):
        return None
    middle = filename[1:-len(".json")]
    try:
        return int(middle)
    except ValueError:
        return None


# --- Current-version resolution ---


def get_current_version(
    name: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> Optional[int]:
    """Return the current version for `name`, or None if cold-start."""
    datasets_dir = _vd(datasets_dir)
    sym = _symlink_current_path(name, datasets_dir=datasets_dir)
    if sym.is_symlink() or sym.exists():
        try:
            target = sym.resolve(strict=True)
            return _parse_version_from_filename(target.name)
        except (FileNotFoundError, OSError):
            pass

    txt = _txt_current_path(name, datasets_dir=datasets_dir)
    if txt.exists():
        try:
            raw = txt.read_text().strip()
            n = int(raw)
            if _dataset_path(name, n, datasets_dir=datasets_dir).exists():
                return n
        except (ValueError, OSError):
            pass

    return None


# --- Load ---


def load_dataset_payload(
    name: str,
    version: int,
    *,
    datasets_dir: Optional[Path] = None,
) -> dict:
    """Load the raw JSON payload for a specific version of a dataset."""
    datasets_dir = _vd(datasets_dir)
    path = _dataset_path(name, version, datasets_dir=datasets_dir)
    if not path.exists():
        raise DatasetNotFoundError(
            f"dataset {name!r} v{version} not found at {path}"
        )
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise DatasetInvalidError(f"cannot parse {path}: {e}") from e
    _validate_schema(payload, path)
    return payload


def load_current(
    name: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> Dataset:
    """Materialize a Dataset from the current pointer for `name`.

    Raises:
        DatasetNotFoundError: when no current pointer exists or the file is missing.
        DatasetInvalidError: when the file fails JSON / schema parse.
    """
    datasets_dir = _vd(datasets_dir)
    version = get_current_version(name, datasets_dir=datasets_dir)
    if version is None:
        raise DatasetNotFoundError(
            f"no current pointer for dataset {name!r} in {datasets_dir}/ — cold-start"
        )
    payload = load_dataset_payload(name, version, datasets_dir=datasets_dir)
    return _payload_to_dataset(payload)


def _validate_schema(payload: dict, path: Path) -> None:
    """Cheap sanity check. Strict schema validation deferred."""
    required = {"version", "name", "samples"}
    missing = required - set(payload.keys())
    if missing:
        raise DatasetInvalidError(
            f"{path} missing required keys: {sorted(missing)}"
        )
    if not isinstance(payload["samples"], list):
        raise DatasetInvalidError(
            f"{path} 'samples' must be a list, got {type(payload['samples']).__name__}"
        )


def _payload_to_dataset(payload: dict) -> Dataset:
    """Convert a raw JSON payload to a Dataset dataclass."""
    metadata = DatasetMetadata(
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
    samples = [
        DatasetSample(
            id=s["id"],
            prompt=s["prompt"],
            ground_truth=s.get("ground_truth", ""),
            tag=s.get("tag"),
            trace_id=s.get("trace_id"),
            added_at=s.get("added_at", ""),
            source=s.get("source", "manual"),
            embedding=list(s.get("embedding", [])),
        )
        for s in payload["samples"]
    ]
    return Dataset(
        metadata=metadata,
        version=int(payload["version"]),
        samples=samples,
        history=list(payload.get("history", [])),
        created_at=payload.get("created_at", ""),
        extra=dict(payload.get("metadata", {})),
    )


# --- Save ---


def save_dataset(
    payload: dict,
    *,
    datasets_dir: Optional[Path] = None,
    update_pointer: bool = True,
    update_registry: bool = True,
) -> Path:
    """Atomically write a dataset_<n>.json artifact + flip the current pointer.

    payload['version'] and payload['name'] are required. Returns the JSON
    path that was written.
    """
    datasets_dir = _vd(datasets_dir)
    if "version" not in payload:
        raise ValueError("payload must include 'version'")
    if "name" not in payload:
        raise ValueError("payload must include 'name'")
    _validate_schema(payload, Path("<inline>"))

    name = payload["name"]
    version = int(payload["version"])
    ds_dir = _dataset_dir(name, datasets_dir=datasets_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)

    json_path = _dataset_path(name, version, datasets_dir=datasets_dir)
    json_payload = json.dumps(payload, indent=2, ensure_ascii=False)
    _atomic_write_text(json_path, json_payload, datasets_dir=datasets_dir)

    if update_pointer:
        _flip_current_pointer(name, version, datasets_dir=datasets_dir)

    if update_registry:
        # Late import to avoid circular dependency at module load.
        from router.data.dataset_registry import sync_from_payload
        sync_from_payload(payload, datasets_dir=datasets_dir)

    return json_path


def _atomic_write_text(target: Path, content: str, *, datasets_dir: Path) -> None:
    staging = datasets_dir / _STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=staging,
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _flip_current_pointer(
    name: str,
    version: int,
    *,
    datasets_dir: Path,
) -> None:
    """Update datasets/<name>/current to point at v<version>.json.

    Tries os.symlink atomically (via tmp + os.replace). On OSError (e.g.
    Windows / overlayfs without privileges), falls back to writing a
    one-line .txt file. Either pointer style is accepted by
    get_current_version.
    """
    ds_dir = _dataset_dir(name, datasets_dir=datasets_dir)
    target_filename = f"v{version}.json"
    sym = _symlink_current_path(name, datasets_dir=datasets_dir)
    txt = _txt_current_path(name, datasets_dir=datasets_dir)

    try:
        tmp_sym = ds_dir / f".{_CURRENT_NAME}.tmp"
        if tmp_sym.exists() or tmp_sym.is_symlink():
            tmp_sym.unlink()
        os.symlink(target_filename, tmp_sym)
        os.replace(tmp_sym, sym)
        if txt.exists():
            txt.unlink()
        return
    except (OSError, NotImplementedError):
        pass

    _atomic_write_text(txt, str(version), datasets_dir=datasets_dir)
    if sym.exists() or sym.is_symlink():
        try:
            sym.unlink()
        except OSError:
            pass


# --- Misc ---


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
