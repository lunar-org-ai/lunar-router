"""router_config_<n>.json load/save + current-pointer resolution.

Disk layout:
    versions/
      router_config_v0.json              (P15.3.1 schema doc)
      router_config_v1.json              (first fitted config — P15.3.7)
      router_config_v1_centroids.npz     (sidecar; centroids in float64 bytes)
      router_config_v2.json              (next promotion)
      router_config_v2_centroids.npz
      router_config_current               (symlink → router_config_vN.json)
        OR
      router_config_current.txt           (single-line "N" — symlink fallback)

The pointer is symlink when the OS supports it; falls back to a one-line
text file otherwise. We probe os.symlink at first save and stick with the
choice for the lifetime of the process.

Atomicity: writes go to versions/.staging/, then os.replace into final
location. Pointer flip uses os.replace on a tmp path. POSIX guarantees
atomicity of os.replace within the same filesystem.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from router.core.clustering import KMeansClusterAssigner
from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError
from router.models.llm_profile import LLMProfile
from router.models.llm_registry import LLMRegistry


VERSIONS_DIR = Path("versions")


def _vd(versions_dir: Optional[Path]) -> Path:
    """Resolve the versions directory at call time so monkeypatching works."""
    if versions_dir is not None:
        return versions_dir
    # Read the module-level constant lazily (tests patch this attribute).
    import router.config_io as _self
    return _self.VERSIONS_DIR
_CURRENT_NAME = "router_config_current"
_CURRENT_TXT = "router_config_current.txt"
_STAGING_DIR = ".staging"


# --- Path helpers ---


def _config_path(version: int, *, versions_dir: Path = VERSIONS_DIR) -> Path:
    return versions_dir / f"router_config_v{version}.json"


def _centroids_path(version: int, *, versions_dir: Path = VERSIONS_DIR) -> Path:
    return versions_dir / f"router_config_v{version}_centroids.npz"


def _symlink_current_path(versions_dir: Path = VERSIONS_DIR) -> Path:
    return versions_dir / _CURRENT_NAME


def _txt_current_path(versions_dir: Path = VERSIONS_DIR) -> Path:
    return versions_dir / _CURRENT_TXT


# --- Current-version resolution ---


def get_current_version(*, versions_dir: Optional[Path] = None) -> Optional[int]:
    """Return the current router_config version, or None if cold-start.

    Tries the symlink first, then the .txt fallback. Returns None if
    neither exists or both point at a missing file.
    """
    versions_dir = _vd(versions_dir)
    sym = _symlink_current_path(versions_dir)
    if sym.is_symlink() or sym.exists():
        # Resolve symlink target and parse version from filename
        try:
            target = sym.resolve(strict=True)
            return _parse_version_from_filename(target.name)
        except (FileNotFoundError, OSError):
            pass

    txt = _txt_current_path(versions_dir)
    if txt.exists():
        try:
            raw = txt.read_text().strip()
            n = int(raw)
            if _config_path(n, versions_dir=versions_dir).exists():
                return n
        except (ValueError, OSError):
            pass

    return None


def _parse_version_from_filename(name: str) -> Optional[int]:
    """`router_config_v3.json` → 3; anything else → None."""
    if not name.startswith("router_config_v") or not name.endswith(".json"):
        return None
    middle = name[len("router_config_v"):-len(".json")]
    try:
        return int(middle)
    except ValueError:
        return None


# --- Load ---


def load_current_config_payload(
    *,
    versions_dir: Optional[Path] = None,
) -> dict:
    """Return the current config's full JSON payload.

    Raises:
        RouterConfigNotFoundError: when no current pointer exists, or when
            the pointer resolves to a missing file.
        RouterConfigInvalidError: when the file fails JSON / schema parse.
    """
    versions_dir = _vd(versions_dir)
    version = get_current_version(versions_dir=versions_dir)
    if version is None:
        raise RouterConfigNotFoundError(
            f"no current router_config in {versions_dir}/ — cold-start"
        )

    path = _config_path(version, versions_dir=versions_dir)
    if not path.exists():
        raise RouterConfigNotFoundError(
            f"current pointer references missing file: {path}"
        )

    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RouterConfigInvalidError(f"cannot parse {path}: {e}") from e

    _validate_schema(payload, path)
    return payload


def load_current_config(
    *,
    versions_dir: Optional[Path] = None,
) -> tuple[KMeansClusterAssigner, LLMRegistry, float]:
    """Materialize (assigner, registry, cost_weight) from the current config.

    Raises the same errors as load_current_config_payload.
    """
    versions_dir = _vd(versions_dir)
    payload = load_current_config_payload(versions_dir=versions_dir)
    return (
        _build_assigner(payload, versions_dir),
        _build_registry(payload),
        float(payload.get("cost_weight", 0.0)),
    )


def _validate_schema(payload: dict, path: Path) -> None:
    """Cheap sanity check. Strict schema validation deferred to a future phase."""
    required = {"version", "k", "model_psi", "cost_weight"}
    missing = required - set(payload.keys())
    if missing:
        raise RouterConfigInvalidError(
            f"{path} missing required keys: {sorted(missing)}"
        )


def _build_assigner(
    payload: dict,
    versions_dir: Path,
) -> KMeansClusterAssigner:
    """Build a KMeansClusterAssigner from the payload + sidecar centroids."""
    version = payload["version"]
    centroids_path = _centroids_path(version, versions_dir=versions_dir)
    if centroids_path.exists():
        return KMeansClusterAssigner.load(centroids_path)
    # Fallback: inline centroids in the JSON (only used by tiny test fixtures
    # — production fits always go to the sidecar).
    inline = payload.get("centroids")
    if inline is None:
        raise RouterConfigInvalidError(
            f"no centroids found: neither {centroids_path} nor inline 'centroids' key"
        )
    return KMeansClusterAssigner(centroids=np.asarray(inline))


def _build_registry(payload: dict) -> LLMRegistry:
    """Reconstruct LLMRegistry from payload['model_psi']."""
    registry = LLMRegistry()
    model_psi: dict = payload.get("model_psi") or {}
    k = int(payload["k"])
    for model_id, blob in model_psi.items():
        if isinstance(blob, list):
            # Compact form: just the Ψ vector. Cost falls back to 0.
            psi = np.asarray(blob, dtype=float)
            cost = 0.0
            counts = np.ones(k, dtype=int)
            metadata: dict = {}
        elif isinstance(blob, dict):
            psi = np.asarray(blob["psi_vector"], dtype=float)
            cost = float(blob.get("cost_per_1k_tokens", 0.0))
            counts = np.asarray(
                blob.get("cluster_sample_counts", [1] * len(psi)), dtype=int
            )
            metadata = dict(blob.get("metadata", {}))
        else:
            raise RouterConfigInvalidError(
                f"model_psi['{model_id}'] must be list or dict, got {type(blob).__name__}"
            )
        registry.register(
            LLMProfile(
                model_id=model_id,
                psi_vector=psi,
                cost_per_1k_tokens=cost,
                num_validation_samples=int(counts.sum()) if counts.size else 0,
                cluster_sample_counts=counts,
                metadata=metadata,
            )
        )
    return registry


# --- Save ---


def save_config(
    payload: dict,
    *,
    centroids: Optional[np.ndarray] = None,
    versions_dir: Optional[Path] = None,
    update_pointer: bool = True,
) -> Path:
    """Atomically write a router_config artifact + flip the current pointer.

    payload['version'] is required and must be int. centroids is written
    to a sibling .npz file when provided (production path). Without centroids,
    the JSON's inline 'centroids' key (if any) is used at load time.

    Returns the JSON path that was written.
    """
    versions_dir = _vd(versions_dir)
    if "version" not in payload:
        raise ValueError("payload must include 'version'")
    version = int(payload["version"])
    versions_dir.mkdir(parents=True, exist_ok=True)

    json_path = _config_path(version, versions_dir=versions_dir)
    json_payload = json.dumps(payload, indent=2)

    _atomic_write_text(json_path, json_payload, versions_dir=versions_dir)

    if centroids is not None:
        npz_path = _centroids_path(version, versions_dir=versions_dir)
        _atomic_write_npz(npz_path, centroids, versions_dir=versions_dir)

    if update_pointer:
        _flip_current_pointer(version, versions_dir=versions_dir)

    return json_path


def save_config_payload(
    payload: dict,
    *,
    versions_dir: Optional[Path] = None,
) -> Path:
    """Convenience for callers (P15.3.8 manual edit) that already have the
    full payload (including inline centroids reference) and just need to
    persist + flip the pointer."""
    return save_config(payload, versions_dir=_vd(versions_dir), update_pointer=True)


def _atomic_write_text(target: Path, content: str, *, versions_dir: Path) -> None:
    staging = versions_dir / _STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=staging,
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _atomic_write_npz(target: Path, centroids: np.ndarray, *, versions_dir: Path) -> None:
    staging = versions_dir / _STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / f"{target.name}.tmp"
    try:
        # np.savez auto-appends '.npz' when given a path/string, but not when
        # given an open file-object. Use the file-object form so our tmp path
        # is the actual on-disk filename.
        with open(tmp, "wb") as f:
            np.savez(f, type="kmeans", centroids=centroids)
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _flip_current_pointer(version: int, *, versions_dir: Path) -> None:
    """Update the current pointer to versions/router_config_v<n>.json.

    Tries os.symlink atomically (via tmp + os.replace). On OSError (e.g.
    Windows / overlayfs without privileges), falls back to writing a one-line
    .txt file. The fallback choice is per-call; either pointer style is
    accepted by get_current_version.
    """
    target_filename = f"router_config_v{version}.json"
    sym = _symlink_current_path(versions_dir)
    txt = _txt_current_path(versions_dir)

    # Try symlink first. If we can't symlink, fall back to .txt.
    try:
        # Atomic replace: create a temporary symlink in versions/, then rename.
        tmp_sym = versions_dir / f".{_CURRENT_NAME}.tmp"
        if tmp_sym.exists() or tmp_sym.is_symlink():
            tmp_sym.unlink()
        os.symlink(target_filename, tmp_sym)
        os.replace(tmp_sym, sym)
        # Clean up any stale .txt fallback to avoid drift between two pointers.
        if txt.exists():
            txt.unlink()
        return
    except (OSError, NotImplementedError):
        pass

    # Symlink path failed — write the .txt marker.
    _atomic_write_text(txt, str(version), versions_dir=versions_dir)
    # Clean up any stale symlink so the two pointers don't disagree.
    if sym.exists() or sym.is_symlink():
        try:
            sym.unlink()
        except OSError:
            pass


# --- Convenience for endpoints + tests ---


def build_view_metadata(payload: dict) -> dict:
    """Extract the small metadata dict the GET /router/config endpoint returns.

    Pure function — no I/O. Doesn't materialize the assigner or registry.
    """
    fitted_from = payload.get("fitted_from") or None
    return {
        "version": int(payload["version"]),
        "k": int(payload.get("k", 0)),
        "model_count": len(payload.get("model_psi") or {}),
        "cost_weight": float(payload.get("cost_weight", 0.0)),
        "embedder_model": payload.get(
            "embedder_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "embedding_dim": int(payload.get("embedding_dim", 384)),
        "last_fit_at": payload.get("created_at"),
        "fitted_from": fitted_from,
        "cold_start": False,
    }


def cold_start_metadata() -> dict:
    """Metadata returned when no router_config exists yet."""
    return {
        "version": None,
        "k": 0,
        "model_count": 0,
        "cost_weight": 0.0,
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "last_fit_at": None,
        "fitted_from": None,
        "cold_start": True,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
