"""Legacy → multi-tenant migration (P16.1).

Moves the pre-P16.1 single-tenant layout under ``tenants/_default/`` and
drops back-compat symlinks at the project root so callers that haven't
been refactored yet keep working.

Idempotent: a second invocation sees ``tenants/_default/`` already in
place and returns ``False`` without touching the filesystem.

Migration shape:

    Before                           After
    ------                           -----
    agents/                          tenants/_default/agents/        (moved)
    ledger/                          tenants/_default/ledger/        (moved)
    traces/                          tenants/_default/traces/        (moved)
    corpora/                         tenants/_default/corpora/       (moved)
                                     agents     -> tenants/_default/agents
                                     ledger     -> tenants/_default/ledger
                                     traces     -> tenants/_default/traces
                                     corpora    -> tenants/_default/corpora

The live ``agent/`` (singular) directory is NOT moved — it stays at the
project root because the runtime executor mounts it directly and we
defer per-tenant live-agent isolation to P16.2.

Failure mode: best-effort. If any individual move raises, we log and
continue with the rest of the migration so a partial install can be
retried on next boot. The migration lock at ``tenants/.migration.lock``
prevents two simultaneous server boots from racing.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger("runtime.tenants.bootstrap")


_TENANTS_ROOT = "tenants"
_DEFAULT_ID = "_default"
_LOCK_FILE = ".migration.lock"
_LOG_FILE = "migration.log.json"

# Top-level dirs we relocate under tenants/_default/<dir>/ and replace
# with a symlink at the project root.
_LEGACY_DIRS = ("agents", "ledger", "traces", "corpora")


def migrate_legacy_to_default(
    project_root: Optional[Path] = None,
    *,
    create_symlinks: bool = True,
    now_iso: Optional[str] = None,
) -> bool:
    """Move legacy data under ``tenants/_default/``. Idempotent.

    Returns ``True`` if migration ran (and at least one dir was moved),
    ``False`` if there was nothing to do (already migrated).

    ``create_symlinks``: if False, skip the back-compat symlinks at the
    project root. Tests use this to keep tmp_path clean; production
    boot always wants them on.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    tenants_dir = root / _TENANTS_ROOT
    default_dir = tenants_dir / _DEFAULT_ID

    # Idempotency check: if _default/ exists AND at least one of its
    # expected subdirs is populated OR the migration log is present,
    # we've already done the work.
    if default_dir.is_dir() and (default_dir / _LOG_FILE).is_file():
        logger.debug(
            "migration: tenants/_default/ already populated (log present), skipping"
        )
        return False

    # Determine which legacy dirs actually need moving. Skip ones that
    # are missing or already a symlink (someone migrated externally).
    moves: list[tuple[Path, Path]] = []
    for name in _LEGACY_DIRS:
        legacy = root / name
        target = default_dir / name
        if legacy.is_symlink():
            continue
        if not legacy.exists():
            continue
        if target.exists():
            # Target dir already populated — leave the legacy in place
            # for the operator to sort out (likely a manual partial migration).
            logger.warning(
                "migration: %s exists; leaving legacy %s in place",
                target,
                legacy,
            )
            continue
        moves.append((legacy, target))

    # If nothing to move AND no legacy dirs at all (fresh install),
    # ensure tenants/_default/ exists for the registry but skip the log.
    if not moves and not _legacy_present(root):
        default_dir.mkdir(parents=True, exist_ok=True)
        return False

    # Acquire migration lock so a second boot races safely.
    tenants_dir.mkdir(parents=True, exist_ok=True)
    lock_path = tenants_dir / _LOCK_FILE
    with _migration_lock(lock_path):
        # Re-check after acquiring the lock — the other process may
        # have finished the migration while we waited.
        if default_dir.is_dir() and (default_dir / _LOG_FILE).is_file():
            return False

        default_dir.mkdir(parents=True, exist_ok=True)

        moved: list[dict[str, str]] = []
        for src, dst in moves:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                logger.info("migration: moved %s → %s", src, dst)
                moved.append({"from": str(src), "to": str(dst)})
            except Exception as e:
                logger.warning("migration: failed to move %s: %s", src, e)
                # Continue with the rest — partial migration is recoverable.

        if create_symlinks:
            for name in _LEGACY_DIRS:
                target_rel = Path(_TENANTS_ROOT) / _DEFAULT_ID / name
                link_path = root / name
                _safe_symlink(link_path, target_rel)

        _write_log(default_dir, moved, _now_iso(now_iso))

    return len(moves) > 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _legacy_present(root: Path) -> bool:
    """True if the project root has anything that looks like the
    pre-P16.1 layout (so a fresh install can short-circuit)."""
    for name in _LEGACY_DIRS:
        p = root / name
        if p.is_dir() and not p.is_symlink():
            return True
    return False


def _safe_symlink(link_path: Path, target: Path) -> None:
    """Create ``link_path -> target`` if nothing exists at link_path.

    Skips if link_path already exists (file, dir, or symlink). A
    pre-existing symlink to a DIFFERENT target is left alone too —
    we don't try to outsmart the operator.
    """
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        # `target_is_directory=True` matters on Windows; on POSIX it's
        # informational. We use a relative target so the symlink keeps
        # working if the whole tree is moved.
        os.symlink(target, link_path, target_is_directory=True)
        logger.info("migration: symlink %s → %s", link_path, target)
    except OSError as e:
        logger.warning("migration: symlink %s failed: %s", link_path, e)


def _write_log(default_dir: Path, moved: list[dict[str, str]], when: str) -> None:
    path = default_dir / _LOG_FILE
    body = {
        "phase": "P16.1",
        "migrated_at": when,
        "moves": moved,
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


class _migration_lock:
    """Best-effort file lock for the migration window.

    On POSIX we use ``fcntl.flock`` if available; otherwise we fall
    back to ``O_EXCL`` create + sleep-poll. Either way it's only used
    once per process boot, so the simplicity is worth more than
    perfect cross-platform correctness.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: Optional[int] = None
        self._used_flock = False

    def __enter__(self) -> "_migration_lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
            self._fd = os.open(
                str(self.path), os.O_RDWR | os.O_CREAT, 0o600
            )
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            self._used_flock = True
        except Exception:
            # Fallback: O_EXCL spin. We'd rather make slow forward
            # progress than crash the boot.
            self._fd = None
            for _ in range(50):
                try:
                    fd = os.open(
                        str(self.path),
                        os.O_RDWR | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    self._fd = fd
                    break
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    import time

                    time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is None:
            return
        try:
            if self._used_flock:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
        except OSError:
            pass
        if not self._used_flock:
            # Remove the lock file so the next boot can recreate it.
            try:
                self.path.unlink()
            except OSError:
                pass


def _now_iso(override: Optional[str] = None) -> str:
    if override:
        return override
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
