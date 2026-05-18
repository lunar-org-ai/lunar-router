"""Workspace store — per-agent filesystem aligned with the AHE NexAU layout.

Per *Agentic Harness Engineering* (Lin et al. 2604.25850v3 §3.1), the
harness exposes seven orthogonal component types as files at fixed
mount points so each failure category maps to a single editable
component class. Plus a Change Manifest layer (§3.3) that pairs every
harness edit with a falsifiable prediction verified the next round.

Layout (relative to ``<agents_root>/<agent_id>/workspace/``)::

    .opentracy/
        system_prompt.md            # 1. system_prompt
        tools/                      # 2. tool_description + 3. tool_implementation
            <name>.json             #     description (json-schema)
            <name>.sh               #     implementation (executable)
        middleware/                 # 4. middleware
            <name>.{py,sh}
        skills/                     # 5. skill (reusable strategies)
            <name>.md
        subagents/                  # 6. sub_agent_configuration
            <name>.json
        memory/                     # 7. long_term_memory
            plan.md                 #     narrative plan (mandatory rewrite)
            state.json              #     structured next-step / facts
        manifest/                   # AHE Change Manifest (Decision Observability)
            pending.json            #     current round's predictions
            history/                #     verified past manifests
                <iso>.json
    ...                              # arbitrary files the agent creates

Per the paper's *minimal seed* invariant (§3.2), only ``system_prompt``
and ``memory/`` are seeded with defaults — tools/middleware/skills/
subagents start empty so every addition must justify itself through
measured evidence rather than seed bias.

The agents_root resolution mirrors :func:`runtime.agents.registry._resolve_root`
so in multi-tenant mode the path is gcsfuse-mounted to
``gs://opentracy-workspaces/<tenant>/<agent>/workspace/`` — persistence
is automatic. In OSS mode the workspace lives on local disk next to
the agent config.

Sandbox transfer
----------------

E2B sandboxes don't share a filesystem with us, so each turn does
restore-to-sandbox / snapshot-from-sandbox round trips. The ``tar``
helpers exchange a single archive for the whole workspace — much
faster than per-file writes for trees with many small files.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


logger = logging.getLogger("runtime.workspaces")


# ---------------------------------------------------------------------------
# NexAU component paths (§3.1 of the AHE paper)
# ---------------------------------------------------------------------------

OPENTRACY_DIR = ".opentracy"

# 1. system_prompt
SYSTEM_PROMPT_FILE = f"{OPENTRACY_DIR}/system_prompt.md"

# 2 + 3. tool_description + tool_implementation share the same dir; the
# evolution agent decides whether to add a new tool by writing a .json
# description and a sibling .sh implementation.
TOOLS_DIR = f"{OPENTRACY_DIR}/tools"

# 4. middleware
MIDDLEWARE_DIR = f"{OPENTRACY_DIR}/middleware"

# 5. skill
SKILLS_DIR = f"{OPENTRACY_DIR}/skills"

# 6. sub_agent_configuration
SUBAGENTS_DIR = f"{OPENTRACY_DIR}/subagents"

# 7. long_term_memory
MEMORY_DIR = f"{OPENTRACY_DIR}/memory"
PLAN_FILE = f"{MEMORY_DIR}/plan.md"
STATE_FILE = f"{MEMORY_DIR}/state.json"

# AHE Decision Observability (§3.3) — Change Manifest layer
MANIFEST_DIR = f"{OPENTRACY_DIR}/manifest"
MANIFEST_PENDING = f"{MANIFEST_DIR}/pending.json"
MANIFEST_HISTORY_DIR = f"{MANIFEST_DIR}/history"
# Rollback snapshot — pre-edit file contents for the files the pending
# manifest claims to have touched. Used when the next round's verdict
# is ``regressed`` to restore the affected files to their pre-edit
# state (file-level rollback per AHE §3.3).
ROLLBACK_SNAPSHOT = f"{MANIFEST_DIR}/rollback_snapshot.json"


_WORKSPACE_DIR = "workspace"
_VERSIONS_DIR = "versions"
# Per-agent version pointer + snapshot history. Lives in the workspace
# (under .opentracy) rather than `agent.yaml` because AHE workspaces
# don't carry one. Updated on each accepted change_manifest lesson.
VERSION_FILE = f"{OPENTRACY_DIR}/version.json"
_DEFAULT_VERSION = "v0.0.1"


def _bump_patch(version: str) -> str:
    """v0.0.1 → v0.0.2 (or 0.0.1 → 0.0.2 if no `v` prefix).

    Local copy of :func:`harness.executor.promote._bump_patch` —
    importing the harness module to grab one helper would pull in
    its whole dependency graph (ledger writer, snapshot machinery).
    """
    has_v = version.startswith("v")
    core = version[1:] if has_v else version
    parts = core.split(".")
    if not parts[-1].isdigit():
        parts.append("1")
    else:
        parts[-1] = str(int(parts[-1]) + 1)
    return ("v" if has_v else "") + ".".join(parts)

# The empty NexAU component dirs we always materialize so the agent
# always sees the same layout regardless of evolution state. Per the
# minimal-seed invariant we don't put anything *in* them — they exist
# only to give the agent a discoverable place to add components.
_NEXAU_DIRS: tuple[str, ...] = (
    TOOLS_DIR,
    MIDDLEWARE_DIR,
    SKILLS_DIR,
    SUBAGENTS_DIR,
    MANIFEST_HISTORY_DIR,
)

# Seed for system_prompt.md — kept deliberately bare-bones so the
# evolution agent can grow it from evidence rather than inheriting our
# editorial choices. The per-turn engineer prompt assembled in
# ``techniques.prompt_strategies`` wraps this with plan + state context.
_DEFAULT_SYSTEM_PROMPT = """\
You are this agent's autonomous engineer.

Your workspace at `/workspace` holds the agent's persistent state across
turns. Before responding to the user you MUST update
`.opentracy/memory/plan.md` and `.opentracy/memory/state.json` so the
next turn can continue from where you stopped — those files are your
only memory.

Keep replies to the user tight. The file changes carry the long-form
record.
"""

_DEFAULT_PLAN = """\
# Agent plan

_No plan yet — the engineer will write the first one when a task arrives._

## Next step

(empty)

## Done so far

(empty)

## Open questions

(empty)
"""

_DEFAULT_STATE: dict[str, Any] = {
    "next_step": None,
    "facts": [],
    "blockers": [],
    "last_turn_at": None,
}

# Limits: keep workspaces lean. Large blobs belong in object storage,
# not the workspace. Raise deliberately if a real use case shows up.
_MAX_WORKSPACE_BYTES = 256 * 1024 * 1024   # 256 MiB
_MAX_FILES = 10_000


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _agents_root(root: Optional[Path] = None) -> Path:
    """Mirror :func:`runtime.agents.registry._resolve_root` without
    importing the heavy registry module at definition time."""
    if root is not None:
        return Path(root)
    from runtime.tenants.feature import is_multi_tenant_enabled
    if is_multi_tenant_enabled():
        from runtime.tenant_context import get_active
        from runtime.tenants.registry import get_tenant_dir
        return get_tenant_dir(get_active()) / "agents"
    return Path("agents")


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# WorkspaceStore
# ---------------------------------------------------------------------------


class WorkspaceStore:
    """Read/write the workspace dir for a single agent.

    Instantiate via :func:`get_workspace` so root resolution is
    consistent — direct construction is allowed but reserved for tests.
    """

    def __init__(self, agent_id: str, *, root: Optional[Path] = None) -> None:
        self.agent_id = agent_id
        self._root = _agents_root(root)
        self.path = self._root / agent_id / _WORKSPACE_DIR

    # -- lifecycle -----------------------------------------------------

    def ensure(self) -> None:
        """Create the workspace + seed system_prompt and memory.

        Empty component dirs (tools/middleware/skills/subagents/manifest
        history) are materialized but not seeded — that's the AHE
        minimal-seed invariant. The evolution agent fills them later
        based on observed task evidence.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / OPENTRACY_DIR).mkdir(parents=True, exist_ok=True)

        for relpath in _NEXAU_DIRS:
            (self.path / relpath).mkdir(parents=True, exist_ok=True)
        (self.path / MEMORY_DIR).mkdir(parents=True, exist_ok=True)
        (self.path / MANIFEST_DIR).mkdir(parents=True, exist_ok=True)

        system_path = self.path / SYSTEM_PROMPT_FILE
        if not system_path.exists():
            system_path.write_text(_DEFAULT_SYSTEM_PROMPT, encoding="utf-8")

        plan_path = self.path / PLAN_FILE
        if not plan_path.exists():
            plan_path.write_text(_DEFAULT_PLAN, encoding="utf-8")

        state_path = self.path / STATE_FILE
        if not state_path.exists():
            state_path.write_text(
                json.dumps(_DEFAULT_STATE, indent=2) + "\n",
                encoding="utf-8",
            )

    # -- system_prompt -------------------------------------------------

    def read_system_prompt(self) -> str:
        path = self.path / SYSTEM_PROMPT_FILE
        if not path.exists():
            return _DEFAULT_SYSTEM_PROMPT
        return path.read_text(encoding="utf-8")

    # -- memory --------------------------------------------------------

    def read_plan(self) -> str:
        plan = self.path / PLAN_FILE
        if not plan.exists():
            return _DEFAULT_PLAN
        return plan.read_text(encoding="utf-8")

    def read_state(self) -> dict[str, Any]:
        state = self.path / STATE_FILE
        if not state.exists():
            return dict(_DEFAULT_STATE)
        try:
            return json.loads(state.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("workspace state corrupt for %s — resetting", self.agent_id)
            return dict(_DEFAULT_STATE)

    # -- change manifest (AHE §3.3) ------------------------------------

    def write_pending_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist this turn's claimed fixes + at-risk regressions.

        Per AHE Decision Observability, every harness change must come
        with a self-declared prediction. We write it here so the next
        turn can verify the prediction against task deltas and decide
        whether to roll back (file-level).
        """
        self.ensure()
        path = self.path / MANIFEST_PENDING
        body = dict(manifest)
        body.setdefault("created_at", _now_iso())
        path.write_text(
            json.dumps(body, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def read_pending_manifest(self) -> Optional[dict[str, Any]]:
        path = self.path / MANIFEST_PENDING
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("pending manifest corrupt for %s — discarding", self.agent_id)
            return None

    def roll_pending_to_history(
        self,
        *,
        outcome: dict[str, Any],
    ) -> Optional[Path]:
        """Move pending.json into history/ with the verification outcome.

        Returns the path of the archived entry, or None if there was no
        pending manifest. Outcome shape::

            {"verdict": "confirmed" | "regressed" | "rolled_back",
             "evidence": <ref to trace ids / metric deltas>,
             "verified_at": "<iso>"}
        """
        manifest = self.read_pending_manifest()
        if manifest is None:
            return None
        manifest["outcome"] = {**outcome, "verified_at": outcome.get("verified_at") or _now_iso()}

        history_dir = self.path / MANIFEST_HISTORY_DIR
        history_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_iso().replace(":", "-")
        archive = history_dir / f"{stamp}.json"
        archive.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            (self.path / MANIFEST_PENDING).unlink()
        except FileNotFoundError:
            pass
        return archive

    def write_rollback_snapshot(
        self,
        *,
        iteration_id: str,
        files: dict[str, Optional[str]],
    ) -> None:
        """Persist pre-edit file contents so the next round can roll back.

        ``files`` maps relative path → content (str) for files that
        existed before the Evolve Agent ran, or ``None`` for paths the
        Evolve Agent NEWLY created (rollback = unlink for those).

        Stored as JSON next to pending.json so the next iteration's
        verification step can find + apply it. Cleared automatically
        after rollback OR when the next iteration confirms the
        manifest (the file represents "edits awaiting verdict", same
        lifetime contract as pending.json itself).
        """
        self.ensure()
        path = self.path / ROLLBACK_SNAPSHOT
        path.write_text(
            json.dumps({
                "iteration_id": iteration_id,
                "files": files,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def read_rollback_snapshot(self) -> Optional[dict[str, Any]]:
        path = self.path / ROLLBACK_SNAPSHOT
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("rollback snapshot corrupt for %s — discarding", self.agent_id)
            return None

    def clear_rollback_snapshot(self) -> None:
        try:
            (self.path / ROLLBACK_SNAPSHOT).unlink()
        except FileNotFoundError:
            pass

    def apply_rollback(self) -> list[str]:
        """Restore each file from the rollback snapshot. Returns the
        list of paths that were rolled back.

        For paths whose snapshot value is ``None`` (the Evolve Agent
        had created them fresh), rollback = unlink. For paths with
        content, rollback = overwrite with the saved content.

        Always clears the snapshot at the end — single-shot.
        """
        snapshot = self.read_rollback_snapshot()
        if snapshot is None:
            return []
        rolled: list[str] = []
        for rel, content in (snapshot.get("files") or {}).items():
            target = self.path.joinpath(rel)
            try:
                # Guard against path traversal in stored snapshot data.
                target.relative_to(self.path.resolve())
            except (ValueError, OSError):
                logger.warning("rollback: skipping suspicious path %r", rel)
                continue
            try:
                if content is None:
                    if target.is_file():
                        target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                rolled.append(rel)
            except OSError as exc:
                logger.warning("rollback: failed on %s: %s", rel, exc)
        self.clear_rollback_snapshot()
        return rolled

    # -- versioning ----------------------------------------------------

    def read_current_version(self) -> str:
        """Return the workspace's current accepted-state version.

        Defaults to :data:`_DEFAULT_VERSION` when no version file exists
        (a fresh agent that has never had a manifest approved)."""
        vpath = self.path / VERSION_FILE
        if not vpath.exists():
            return _DEFAULT_VERSION
        try:
            doc = json.loads(vpath.read_text(encoding="utf-8"))
            current = doc.get("current")
            return current if isinstance(current, str) and current else _DEFAULT_VERSION
        except (json.JSONDecodeError, OSError):
            return _DEFAULT_VERSION

    def _versions_root(self) -> Path:
        """Sibling of ``workspace/`` — holds snapshots per accepted manifest."""
        return self.path.parent / _VERSIONS_DIR

    def snapshot_workspace_for_version(self, version: str) -> Path:
        """Copy the live workspace tree to
        ``<agents-root>/<agent>/versions/<version>/workspace/``.

        Idempotent — if the snapshot already exists, return its path
        unchanged so re-running the accept twice doesn't lose history.
        """
        target = self._versions_root() / version / _WORKSPACE_DIR
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copytree is the simplest reliable copy; sets `dirs_exist_ok`
        # to handle gcsfuse races where the parent dir already exists.
        import shutil
        shutil.copytree(self.path, target, dirs_exist_ok=True)
        return target

    def bump_and_snapshot(self, *, reason: str) -> str:
        """Bump patch, snapshot current workspace, write the new version
        pointer. Returns the new version string.

        Called when an AHE change_manifest lesson is approved — the
        workspace IS the live agent in this model, so the snapshot is
        of "what just got accepted" so rollback can restore it later.
        """
        current = self.read_current_version()
        new_version = _bump_patch(current)
        snapshot_path = self.snapshot_workspace_for_version(new_version)

        doc = {
            "current": new_version,
            "previous": current,
            "promoted_at": _now_iso(),
            "snapshot_path": str(snapshot_path),
            "reason": reason,
        }
        version_path = self.path / VERSION_FILE
        version_path.parent.mkdir(parents=True, exist_ok=True)
        version_path.write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )
        return new_version

    def list_versions(self) -> list[str]:
        """All accepted version snapshots under this agent, sorted ASC."""
        root = self._versions_root()
        if not root.exists():
            return []
        return sorted(
            d.name for d in root.iterdir()
            if (d / _WORKSPACE_DIR).exists()
        )

    def restore_version(self, version: str) -> None:
        """Replace the live workspace with the snapshot for ``version``.

        Snapshots the *current* state first under its current version
        marker so rollback-of-rollback is possible. Updates the version
        pointer to ``version`` and writes a new ``promoted_at`` so the
        timeline reflects the restoration event."""
        import shutil
        snap = self._versions_root() / version / _WORKSPACE_DIR
        if not snap.exists():
            raise FileNotFoundError(
                f"no snapshot for version {version!r}: {snap}"
            )
        current = self.read_current_version()
        if current != version:
            # Preserve the current state so we can roll-back the rollback.
            self.snapshot_workspace_for_version(current)
        # Replace live workspace contents.
        if self.path.exists():
            shutil.rmtree(self.path)
        shutil.copytree(snap, self.path)
        doc = {
            "current": version,
            "previous": current,
            "promoted_at": _now_iso(),
            "snapshot_path": str(snap),
            "reason": f"restored from version {version}",
        }
        version_path = self.path / VERSION_FILE
        version_path.parent.mkdir(parents=True, exist_ok=True)
        version_path.write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )

    def list_manifest_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent archived manifests, newest first."""
        history_dir = self.path / MANIFEST_HISTORY_DIR
        if not history_dir.exists():
            return []
        items: list[tuple[str, dict[str, Any]]] = []
        for entry in sorted(history_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("history manifest %s unreadable — skipping", entry.name)
                continue
            items.append((entry.name, data))
            if len(items) >= limit:
                break
        return [data for _name, data in items]

    # -- inventory -----------------------------------------------------

    def list_files(self, *, max_files: int = 500) -> list[str]:
        """Relative paths under the workspace, sorted, capped."""
        if not self.path.exists():
            return []
        rel: list[str] = []
        for p in sorted(self.path.rglob("*")):
            if not p.is_file():
                continue
            rel.append(str(p.relative_to(self.path)))
            if len(rel) >= max_files:
                break
        return rel

    def list_nexau_components(self) -> dict[str, list[str]]:
        """Snapshot of which NexAU components have content right now.

        Returns a mapping ``{component_type: [file_names]}`` for the
        evolution agent's introspection. Empty lists are kept — the
        absence of a key would obscure the minimal-seed baseline.
        """
        out: dict[str, list[str]] = {
            "system_prompt": [],
            "tools": [],
            "middleware": [],
            "skills": [],
            "subagents": [],
            "memory": [],
        }
        if (self.path / SYSTEM_PROMPT_FILE).exists():
            out["system_prompt"].append("system_prompt.md")
        out["tools"] = _list_dir_files(self.path / TOOLS_DIR)
        out["middleware"] = _list_dir_files(self.path / MIDDLEWARE_DIR)
        out["skills"] = _list_dir_files(self.path / SKILLS_DIR)
        out["subagents"] = _list_dir_files(self.path / SUBAGENTS_DIR)
        out["memory"] = _list_dir_files(self.path / MEMORY_DIR)
        return out

    # -- sandbox transfer ---------------------------------------------

    def to_tar_bytes(self) -> bytes:
        """Pack the whole workspace into a gzipped tar for sandbox upload.

        Returns the tar bytes (small workspaces — fits in memory).
        Raises :class:`ValueError` if the workspace exceeds limits.
        """
        self.ensure()
        total = 0
        count = 0
        for p in self.path.rglob("*"):
            if p.is_file():
                count += 1
                total += p.stat().st_size
        if count > _MAX_FILES:
            raise ValueError(
                f"workspace has {count} files, exceeds limit {_MAX_FILES}"
            )
        if total > _MAX_WORKSPACE_BYTES:
            raise ValueError(
                f"workspace is {total} bytes, exceeds limit {_MAX_WORKSPACE_BYTES}"
            )

        # Add each top-level entry individually rather than the workspace
        # root as ``.``. Including ``.`` made tar emit a root entry whose
        # extraction tries to ``utime()`` /workspace inside the sandbox,
        # failing with EPERM because the dir is owned by the image's
        # root user but commands run as the non-root sandbox user. Adding
        # per-entry skips the root-dir metadata operation entirely.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in sorted(self.path.iterdir()):
                tar.add(child, arcname=child.name, recursive=True)
        return buf.getvalue()

    def from_tar_bytes(self, data: bytes) -> None:
        """Replace the workspace contents with the given tar.

        The incoming archive is treated as authoritative — anything in
        the existing workspace that isn't in the tar is dropped. The
        ``.opentracy/`` skeleton is re-seeded after extraction so future
        turns still find the NexAU mount points even if the sandbox
        accidentally nuked some.
        """
        if not data:
            raise ValueError("empty tar bytes")

        # Wipe everything first so deletions from the sandbox propagate.
        if self.path.exists():
            import shutil
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # Guard against path-traversal attacks in malicious tars —
            # filter members so nothing escapes the workspace root.
            members = []
            for m in tar.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    logger.warning(
                        "rejecting tar member %r (path traversal)", m.name
                    )
                    continue
                members.append(m)
            tar.extractall(self.path, members=members, filter="data")

        # Re-seed any missing NexAU mount points + memory defaults.
        self.ensure()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_dir_files(d: Path) -> list[str]:
    """Sorted file names directly under ``d`` (non-recursive)."""
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file())


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_workspace(agent_id: str, *, root: Optional[Path] = None) -> WorkspaceStore:
    """Resolve a workspace store for the given agent under the active tenant."""
    ws = WorkspaceStore(agent_id, root=root)
    ws.ensure()
    return ws
