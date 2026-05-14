"""Tenant registry — CRUD on ``tenants/`` (P16.1).

Operations
----------

  - ``ensure_bootstrapped()`` — first-run setup. If ``tenants/_registry.json``
    is missing, create it with a single ``_default`` entry. Does NOT
    move legacy data — that's :mod:`runtime.tenants.bootstrap` job.
    Idempotent.
  - ``list_tenants()`` / ``get_tenant(id)`` — read.
  - ``create_tenant(name, slug=None)`` — make a new ``tenants/<id>/``.
    Slug auto-derived from name with collision suffix.
  - ``delete_tenant(id)`` — soft delete (rename to ``tenants/_deleted/<id>``).
    Refuses to delete ``_default``.
  - ``get_tenant_dir(id)`` — resolved Path under the registry root.

Concurrency: same model as the agent registry — FastAPI single-process,
write + rename for atomic persistence.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runtime.tenants.types import TenantMetadata, TenantRegistry


logger = logging.getLogger("runtime.tenants.registry")


_DEFAULT_ROOT = Path("tenants")
_REGISTRY_FILE = "_registry.json"
_TOKENS_INDEX_FILE = "_tokens_index.json"
_DELETED_BUCKET = "_deleted"
_DEFAULT_ID = "_default"

# Reserved at the top-level of ``tenants/`` so the slug namespace
# doesn't clash with the registry's own filesystem layout.
_RESERVED_IDS = {"_default", "_deleted", "_registry", "_tokens_index"}

# Slug rule: start with alphanum, 2-41 chars total, kebab-case.
# Tighter than the agent slug (which uses a substitution regex);
# tenants are operator-facing so we want a clean rule.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
_SLUG_SUB = re.compile(r"[^a-z0-9-]+")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def ensure_bootstrapped(
    *,
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> TenantRegistry:
    """If ``tenants/_registry.json`` is missing, create it with a single
    ``_default`` entry. Idempotent.

    Does NOT migrate legacy single-tenant data — that's the job of
    :func:`runtime.tenants.bootstrap.migrate_legacy_to_default`, which
    callers run BEFORE this so the on-disk shape is consistent.
    """
    rroot = _resolve_root(root)
    rroot.mkdir(parents=True, exist_ok=True)

    registry_path = rroot / _REGISTRY_FILE
    if registry_path.is_file():
        return _load_registry(rroot)

    logger.info("tenants/_registry.json missing — seeding with _default")
    (rroot / _DEFAULT_ID).mkdir(parents=True, exist_ok=True)

    timestamp = _now_iso(now_iso)
    meta = TenantMetadata(
        id=_DEFAULT_ID,
        name="Default tenant",
        description="Bootstrap tenant. Holds all data from the single-tenant layout that preceded P16.1.",
        created_at=timestamp,
        updated_at=timestamp,
    )
    registry = TenantRegistry(tenants=[meta])
    _save_registry(rroot, registry)
    return registry


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_tenants(*, root: Optional[Path] = None) -> TenantRegistry:
    return _load_registry(_resolve_root(root))


def get_tenant(tenant_id: str, *, root: Optional[Path] = None) -> Optional[TenantMetadata]:
    return _load_registry(_resolve_root(root)).get(tenant_id)


def get_tenant_dir(tenant_id: str, *, root: Optional[Path] = None) -> Path:
    """The on-disk root for ``tenant_id``. Does not verify existence —
    callers that need a guarantee should also call :func:`get_tenant`.

    The directory is created on tenant creation; this helper is read-
    only on the filesystem.
    """
    return _resolve_root(root) / tenant_id


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def create_tenant(
    name: str,
    *,
    slug: Optional[str] = None,
    description: str = "",
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> TenantMetadata:
    """Create a new tenant directory + registry entry.

    ``slug`` defaults to a slugified ``name`` with a collision suffix.
    Reserved IDs are rejected. The new tenant gets empty subdirs for
    ``agents/``, ``ledger/``, ``traces/``, ``corpora/`` so downstream
    writers don't trip on missing parents.
    """
    rroot = _resolve_root(root)
    registry = _load_registry(rroot)

    if slug:
        tenant_id = slug.strip()
        # Strict: case-sensitive match. Operators are expected to pass a
        # well-formed slug; auto-slugs from a free-form name lowercase
        # via _allocate_slug. Mixing both modes silently is confusing.
        if not _SLUG_RE.match(tenant_id):
            raise ValueError(
                f"invalid slug {tenant_id!r}: must match {_SLUG_RE.pattern}"
            )
        if tenant_id in _RESERVED_IDS:
            raise ValueError(f"tenant id {tenant_id!r} is reserved")
        if registry.get(tenant_id) is not None:
            raise ValueError(f"tenant {tenant_id!r} already exists")
    else:
        tenant_id = _allocate_slug(name or "tenant", registry)

    tenant_dir = rroot / tenant_id
    if tenant_dir.exists():
        # Defensive: previous run created the dir but failed to register.
        # Re-register from scratch rather than refusing.
        logger.warning("tenants/%s/ already exists; re-registering", tenant_id)
    else:
        tenant_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("agents", "ledger", "traces", "corpora"):
        (tenant_dir / sub).mkdir(parents=True, exist_ok=True)

    timestamp = _now_iso(now_iso)
    meta = TenantMetadata(
        id=tenant_id,
        name=str(name or tenant_id),
        description=description,
        created_at=timestamp,
        updated_at=timestamp,
    )
    registry.tenants.append(meta)
    _save_registry(rroot, registry)
    return meta


def delete_tenant(
    tenant_id: str,
    *,
    root: Optional[Path] = None,
) -> TenantMetadata:
    """Soft delete: move ``tenants/<id>/`` to ``tenants/_deleted/<id>-<rand>/``.

    Refuses to delete ``_default``. Also drops every token belonging to
    this tenant from the global ``_tokens_index.json`` so resolves
    against revoked tenants 401 immediately.
    """
    rroot = _resolve_root(root)
    registry = _load_registry(rroot)
    meta = registry.get(tenant_id)
    if meta is None:
        raise KeyError(tenant_id)
    if tenant_id == _DEFAULT_ID:
        raise ValueError("cannot delete the _default tenant")

    src = rroot / tenant_id
    if src.is_dir():
        bucket = rroot / _DELETED_BUCKET
        bucket.mkdir(parents=True, exist_ok=True)
        dest = bucket / f"{tenant_id}-{secrets.token_hex(3)}"
        shutil.move(str(src), str(dest))

    registry.tenants = [t for t in registry.tenants if t.id != tenant_id]
    _save_registry(rroot, registry)
    _drop_tokens_for_tenant(rroot, tenant_id)
    return meta


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _allocate_slug(seed: str, registry: TenantRegistry) -> str:
    """Slugify ``seed``; if it collides or hits a reserved id, append a
    4-char hex suffix."""
    base = _SLUG_SUB.sub("-", seed.lower()).strip("-")
    if not base or len(base) < 2:
        base = "tenant"
    # Trim to fit the regex max (41 chars) leaving room for a 5-char suffix.
    base = base[:35]
    if base not in _RESERVED_IDS and registry.get(base) is None and _SLUG_RE.match(base):
        return base
    for _ in range(8):
        candidate = f"{base}-{secrets.token_hex(2)}"
        if (
            candidate not in _RESERVED_IDS
            and registry.get(candidate) is None
            and _SLUG_RE.match(candidate)
        ):
            return candidate
    # Extreme collision — fall back to a fully-random id.
    return f"tenant-{secrets.token_hex(4)}"


def _resolve_root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else _DEFAULT_ROOT


def _load_registry(root: Path) -> TenantRegistry:
    path = root / _REGISTRY_FILE
    if not path.is_file():
        return TenantRegistry()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return TenantRegistry.from_dict(data)


def _save_registry(root: Path, registry: TenantRegistry) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / _REGISTRY_FILE
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(registry.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _drop_tokens_for_tenant(root: Path, tenant_id: str) -> None:
    """Remove every entry in ``_tokens_index.json`` that maps to ``tenant_id``.

    Best-effort: if the index file is missing or unreadable, log and
    continue — the per-tenant ``tokens.json`` is already gone with the
    deleted dir.
    """
    index_path = root / _TOKENS_INDEX_FILE
    if not index_path.is_file():
        return
    try:
        with index_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not read tokens index for revocation: %s", e)
        return
    pruned = {h: tid for h, tid in data.items() if tid != tenant_id}
    if len(pruned) == len(data):
        return
    tmp = index_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(index_path)


def _now_iso(override: Optional[str] = None) -> str:
    if override:
        return override
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
