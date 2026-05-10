"""Count-threshold wake-up scheduler (P15.3.9).

Hooked into ``runtime/store/traces.py:write_trace()``. Each trace bumps
a persisted counter; when the counter crosses ``threshold`` we fire a
fire-and-forget thread that calls ``run_wakeup()``. The lockfile
prevents concurrent wake-ups across processes; the daemonized thread
keeps ``/run`` non-blocking.

No cron, no fixed thresholds — Claude Code (the brain) decides whether
to actually propose a retrain. The scheduler just *invites* it via the
introspection prompt.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional


logger = logging.getLogger("harness.wakeup.scheduler")


_DEFAULT_THRESHOLD = int(os.getenv("HARNESS_ROUTER_WAKEUP_N", "50"))
_COUNTER_PATH = Path(".harness") / "wakeup_counter.txt"
_LOCK_PATH = Path("/tmp") / "opentracy_router_wakeup.lock"

# Module-level lock guards counter increments within a single process.
# Cross-process exclusion is via the lockfile in _LOCK_PATH.
_counter_lock = threading.Lock()


class _LockHeldError(Exception):
    """Internal: another wake-up is currently running."""


# ---------------------------------------------------------------------------


def increment_trace_counter(*, counter_path: Optional[Path] = None) -> int:
    """Bump the persisted counter atomically; returns the new value."""
    path = counter_path or _COUNTER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _counter_lock:
        n = 0
        if path.exists():
            try:
                raw = path.read_text().strip()
                n = int(raw or "0")
            except (OSError, ValueError):
                n = 0
        n += 1
        path.write_text(str(n))
        return n


def reset_trace_counter(*, counter_path: Optional[Path] = None) -> None:
    path = counter_path or _COUNTER_PATH
    with _counter_lock:
        try:
            path.write_text("0")
        except OSError:
            pass


def maybe_fire(
    *,
    threshold: Optional[int] = None,
    counter_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
    runner: Optional[Callable[[], None]] = None,
) -> None:
    """Increment the counter; if it reached ``threshold``, fire async.

    This must return instantly — it's called from the trace write path.
    Concurrent wake-ups are blocked by the lockfile.

    Args:
        threshold: Override ``HARNESS_ROUTER_WAKEUP_N`` env default.
        counter_path / lock_path: Override defaults (tests use this).
        runner: Override the ``run_wakeup`` callable (tests use this).
    """
    th = threshold if threshold is not None else _DEFAULT_THRESHOLD
    cpath = counter_path or _COUNTER_PATH
    lpath = lock_path or _LOCK_PATH
    n = increment_trace_counter(counter_path=cpath)
    if n < th:
        return

    # Try to acquire the cross-process lockfile.
    try:
        _acquire_lock(lpath)
    except _LockHeldError:
        # Previous wake-up still running — skip; the next trace will retry.
        logger.info("wakeup skipped: another wakeup is already running")
        return

    # Reset counter only after we successfully acquired the lock.
    reset_trace_counter(counter_path=cpath)

    fn = runner or _default_runner

    def _run_with_release() -> None:
        try:
            fn()
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("wakeup run failed: %s", e)
        finally:
            _release_lock(lpath)

    thread = threading.Thread(
        target=_run_with_release,
        daemon=True,
        name="router-wakeup",
    )
    thread.start()


# ---------------------------------------------------------------------------
# Lockfile (cross-process) — fcntl on POSIX, no-op stub elsewhere.
# ---------------------------------------------------------------------------


def _acquire_lock(path: Path) -> None:
    """Create the lockfile with O_EXCL. Atomic on POSIX.

    Stores the current PID. Before raising _LockHeldError on collision,
    we check whether the holder PID is still alive — if not, the lock is
    stale (left over from a crashed wakeup) and we sweep it, then retry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Stale lock detection — was the holder PID killed?
        if _sweep_stale_lock(path):
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as e2:
                raise _LockHeldError(f"lock held (after sweep retry): {path}") from e2
        else:
            raise _LockHeldError(f"lock held: {path}") from None
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)


def _sweep_stale_lock(path: Path) -> bool:
    """If the lockfile's PID isn't a live process, delete it and return True."""
    try:
        raw = path.read_text().strip()
    except OSError:
        return False
    try:
        held_pid = int(raw)
    except ValueError:
        # Corrupt lockfile content — treat as stale.
        try:
            path.unlink()
            logger.warning("stale lockfile (corrupt PID) swept: %s", path)
            return True
        except OSError:
            return False

    if _pid_alive(held_pid):
        return False

    try:
        path.unlink()
        logger.warning("stale lockfile from PID %d swept: %s", held_pid, path)
        return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    """Return True if a process with that PID is alive (POSIX kill -0)."""
    try:
        os.kill(pid, 0)
    except OSError as e:
        # ESRCH = no such process; EPERM = exists but we lack permission.
        if getattr(e, "errno", None) == 1:  # EPERM — process exists, owned by another user
            return True
        return False
    return True


def _release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("lock release failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------


def _default_runner() -> None:
    """Default wakeup body — calls ``run_wakeup`` from the runner module.

    Defined as a module-level function (not a lambda) so tests can swap it
    out via ``runner=`` arg.
    """
    from harness.wakeup.runner import run_wakeup

    run_wakeup()
