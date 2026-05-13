"""Agent registry — CRUD on ``agents/`` + the live ``agent/`` dir (P2.0).

Operations
----------

  - ``ensure_bootstrapped()`` — first-run migration. If ``agents/registry.json``
    is missing, snapshot the current ``agent/`` directory into
    ``agents/_default/``, write the registry with ``_default`` as the
    active agent. Idempotent.
  - ``list_agents()`` / ``get_agent(id)`` — read.
  - ``create_agent(payload)`` — make a new ``agents/<id>/``. Seeded with
    the operator's prompt/model/channels (from onboarding) and a default
    pipeline copied from the currently-active agent.
  - ``activate(id)`` — copy ``agents/<id>/*`` → ``agent/*`` and update
    the registry's ``active`` pointer. Triggers ``on_activate`` hook so
    the runtime can recompile its pipeline.
  - ``delete_agent(id)`` — soft delete (rename to ``_deleted/<id>``)
    plus registry mutation. Never touches the live ``agent/`` dir; if
    the active agent is deleted, the caller decides what to activate
    next (UI surface).

Concurrency: FastAPI single-process, no real concurrency between
requests. A simple file write + rename is enough — no locking primitives.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from runtime.agents.types import AgentMetadata, Registry


logger = logging.getLogger("runtime.agents.registry")


_DEFAULT_ROOT = Path("agents")
_LIVE_AGENT_DIR = Path("agent")           # the running agent
_REGISTRY_FILE = "registry.json"
_DELETED_BUCKET = "_deleted"
_DEFAULT_ID = "_default"
_SLUG_RE = re.compile(r"[^a-z0-9-]+")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def ensure_bootstrapped(
    *,
    root: Optional[Path] = None,
    live_dir: Optional[Path] = None,
    now_iso: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Registry:
    """If ``agents/registry.json`` is missing, migrate ``agent/`` to
    ``agents/_default/`` and write a registry. Idempotent.

    P2.1 — also partitions storage. Flat dirs that pre-date multi-agent
    (``ledger/{entries,lessons,decisions,notifications}/...`` +
    ``traces/{raw,feedback,flagged}/...``) get migrated under
    ``<root>/_default/<kind>/...``. Idempotent: if the partition dir
    already exists, the flat path is left as-is for any operator who
    rolled back this code.

    Returns the registry in its post-bootstrap state.
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT
    live = Path(live_dir) if live_dir is not None else _LIVE_AGENT_DIR
    project = Path(project_root) if project_root is not None else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    # Migrate flat storage into _default/ before / regardless of whether
    # the registry exists, so re-running after partial install also
    # finishes the job.
    _migrate_flat_storage(project)

    registry_path = root / _REGISTRY_FILE
    if registry_path.is_file():
        return _load_registry(root)

    logger.info("agents/registry.json missing — migrating agent/ → agents/_default/")
    default_dir = root / _DEFAULT_ID
    if not default_dir.is_dir():
        if live.is_dir():
            _copy_tree(live, default_dir)
        else:
            default_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("no live agent/ dir to migrate — created empty agents/_default/")

    meta = AgentMetadata(
        id=_DEFAULT_ID,
        name="Default agent",
        template=None,
        description="Migrated from the single-agent layout that preceded P2.0.",
        created_at=_now_iso(now_iso),
        updated_at=_now_iso(now_iso),
    )
    registry = Registry(agents=[meta], active=_DEFAULT_ID)
    _save_registry(root, registry)
    return registry


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_agents(*, root: Optional[Path] = None) -> list[AgentMetadata]:
    return _load_registry(_resolve_root(root)).agents


def get_agent(agent_id: str, *, root: Optional[Path] = None) -> Optional[AgentMetadata]:
    return _load_registry(_resolve_root(root)).get(agent_id)


def get_active(*, root: Optional[Path] = None) -> Optional[AgentMetadata]:
    reg = _load_registry(_resolve_root(root))
    return reg.get(reg.active) if reg.active else None


def get_registry(*, root: Optional[Path] = None) -> Registry:
    return _load_registry(_resolve_root(root))


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def create_agent(
    payload: dict[str, Any],
    *,
    root: Optional[Path] = None,
    seed_from: Optional[str] = None,
    now_iso: Optional[str] = None,
) -> AgentMetadata:
    """Create a new agent in ``agents/<id>/``.

    ``payload`` is the onboarding result: ``{name, model, prompt, tools,
    channels, template?}``. We slugify the name into the directory id
    (unique across the registry; appends a hex suffix on collision).

    ``seed_from`` (optional) is the id of an existing agent whose
    pipeline yamls are used as the starting point. Defaults to the
    currently-active agent so the new agent inherits the routing /
    rerank / generate stages. The prompt + model + name are overwritten
    by the payload.
    """
    rroot = _resolve_root(root)
    registry = _load_registry(rroot)

    agent_id = _allocate_slug(payload.get("name", "") or "agent", registry)
    agent_dir = rroot / agent_id
    if agent_dir.exists():
        raise ValueError(f"agents/{agent_id} already exists")

    source_id = seed_from or registry.active or _DEFAULT_ID
    source_dir = rroot / source_id
    if source_dir.is_dir():
        _copy_tree(source_dir, agent_dir)
    else:
        agent_dir.mkdir(parents=True, exist_ok=True)

    _apply_payload_to_dir(agent_dir, payload)

    timestamp = _now_iso(now_iso)
    meta = AgentMetadata(
        id=agent_id,
        name=str(payload.get("name", "") or agent_id),
        template=payload.get("template"),
        model=str(payload.get("model", "claude-sonnet-4-6")),
        description=_describe_from_payload(payload),
        created_at=timestamp,
        updated_at=timestamp,
        onboarding_completed_at=timestamp,
        onboarding=_clone_payload(payload),
    )
    registry.agents.append(meta)
    _save_registry(rroot, registry)
    return meta


def update_agent(
    agent_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    model: Optional[str] = None,
    root: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> AgentMetadata:
    rroot = _resolve_root(root)
    registry = _load_registry(rroot)
    meta = registry.get(agent_id)
    if meta is None:
        raise KeyError(agent_id)
    if name is not None:
        meta.name = name
    if description is not None:
        meta.description = description
    if model is not None:
        meta.model = model
    meta.updated_at = _now_iso(now_iso)
    _save_registry(rroot, registry)
    return meta


def activate(
    agent_id: str,
    *,
    root: Optional[Path] = None,
    live_dir: Optional[Path] = None,
    on_activate: Optional[Callable[[AgentMetadata], None]] = None,
) -> AgentMetadata:
    """Make ``agents/<id>/`` the live agent.

    Copies the catalog entry into ``agent/`` (the runtime reads from
    there). Updates ``registry.active``. The optional ``on_activate``
    hook fires after the copy — the server uses it to recompile its
    pipeline.
    """
    rroot = _resolve_root(root)
    live = Path(live_dir) if live_dir is not None else _LIVE_AGENT_DIR
    registry = _load_registry(rroot)
    meta = registry.get(agent_id)
    if meta is None:
        raise KeyError(agent_id)

    source = rroot / agent_id
    if not source.is_dir():
        raise FileNotFoundError(f"agents/{agent_id}/ missing on disk")

    # Snapshot current live agent back into its catalog entry so any
    # mid-flight prompt/route edits aren't lost on switch.
    prev_active = registry.active
    if prev_active and prev_active != agent_id and live.is_dir():
        prev_dir = rroot / prev_active
        if prev_dir.is_dir() or registry.get(prev_active) is not None:
            _copy_tree(live, prev_dir)

    _replace_tree(source, live)
    registry.active = agent_id
    _save_registry(rroot, registry)

    if on_activate is not None:
        try:
            on_activate(meta)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("on_activate hook failed for %s: %s", agent_id, e)

    return meta


def delete_agent(
    agent_id: str,
    *,
    root: Optional[Path] = None,
) -> AgentMetadata:
    """Soft delete: move ``agents/<id>/`` to ``agents/_deleted/<id>/``.
    Refuses to delete the active agent (caller must activate something
    else first) and refuses to delete ``_default`` (it's the
    bootstrap fallback)."""
    rroot = _resolve_root(root)
    registry = _load_registry(rroot)
    meta = registry.get(agent_id)
    if meta is None:
        raise KeyError(agent_id)
    if agent_id == _DEFAULT_ID:
        raise ValueError("cannot delete the _default agent")
    if registry.active == agent_id:
        raise ValueError("cannot delete the active agent — switch first")

    src = rroot / agent_id
    if src.is_dir():
        bucket = rroot / _DELETED_BUCKET
        bucket.mkdir(parents=True, exist_ok=True)
        dest = bucket / f"{agent_id}-{secrets.token_hex(3)}"
        shutil.move(str(src), str(dest))

    registry.agents = [a for a in registry.agents if a.id != agent_id]
    _save_registry(rroot, registry)
    return meta


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_payload_to_dir(agent_dir: Path, payload: dict[str, Any]) -> None:
    """Materialize the operator's onboarding payload into the agent dir."""
    prompt = (payload.get("prompt") or "").strip()
    if prompt:
        prompt_path = agent_dir / "prompts" / "system.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        body = (
            prompt
            + "\n\nThis prompt is part of the trainable surface. The harness may mutate it.\n"
        )
        prompt_path.write_text(body, encoding="utf-8")

    onboarding_path = agent_dir / "onboarding.json"
    onboarding_path.write_text(
        json.dumps(_clone_payload(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _describe_from_payload(payload: dict[str, Any]) -> str:
    name = payload.get("name") or ""
    template = payload.get("template")
    if template:
        return f"{name} — template {template}"
    return name or "(unnamed)"


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "template": payload.get("template"),
        "name": payload.get("name", ""),
        "company": payload.get("company", ""),
        "prompt": payload.get("prompt", ""),
        "model": payload.get("model", "claude-sonnet-4-6"),
        "tools": list(payload.get("tools") or []),
        "channels": list(payload.get("channels") or []),
    }


def _allocate_slug(seed: str, registry: Registry) -> str:
    """Slugify ``seed``; if it collides, append a 4-char hex suffix."""
    base = _SLUG_RE.sub("-", seed.lower()).strip("-")
    if not base:
        base = "agent"
    if not registry.get(base):
        return base
    for _ in range(8):
        candidate = f"{base}-{secrets.token_hex(2)}"
        if not registry.get(candidate):
            return candidate
    # Extreme collision — fall back to a fully-random id.
    return f"agent-{secrets.token_hex(4)}"


_PARTITIONED_KINDS = {
    "ledger": ("entries", "lessons", "decisions", "notifications"),
    "traces": ("raw", "feedback", "flagged"),
}


def _migrate_flat_storage(project_root: Path) -> None:
    """Move pre-multi-agent flat dirs under ``<root>/_default/<kind>/``.

    Idempotent and best-effort: if any move fails (perm error, mount
    weirdness), we log + skip rather than block startup. The flat path
    stays in place so a future run can retry.
    """
    for parent, kinds in _PARTITIONED_KINDS.items():
        parent_dir = project_root / parent
        if not parent_dir.is_dir():
            continue
        partition_dir = parent_dir / _DEFAULT_ID
        for kind in kinds:
            flat = parent_dir / kind
            if not flat.is_dir():
                continue
            target = partition_dir / kind
            if target.exists():
                # Already partitioned previously — leave the flat dir to
                # the operator's discretion (could be a residual mount).
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(flat), str(target))
                logger.info("migrated %s → %s", flat, target)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("flat-storage migration of %s failed (%s)", flat, e)


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy directory contents. Overwrites destination."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=False)


def _replace_tree(src: Path, dst: Path) -> None:
    """Atomic-ish replace: copy src to a temp dir, then move into place.

    Falls back to direct copy if the temp move would cross devices.
    """
    if not src.is_dir():
        raise FileNotFoundError(src)
    parent = dst.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{dst.name}.staging-{secrets.token_hex(3)}"
    try:
        shutil.copytree(src, staging, symlinks=False)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(staging), str(dst))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _resolve_root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else _DEFAULT_ROOT


def _load_registry(root: Path) -> Registry:
    path = root / _REGISTRY_FILE
    if not path.is_file():
        return Registry()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return Registry.from_dict(data)


def _save_registry(root: Path, registry: Registry) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / _REGISTRY_FILE
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(registry.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _now_iso(override: Optional[str] = None) -> str:
    if override:
        return override
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
