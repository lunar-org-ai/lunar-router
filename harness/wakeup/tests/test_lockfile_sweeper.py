"""Tests for the stale-lockfile sweeper in harness/wakeup/scheduler."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.wakeup import scheduler


def test_sweep_removes_lock_with_dead_pid(tmp_path: Path):
    lock = tmp_path / "wakeup.lock"
    # Write a PID that's almost certainly not running. PID 0 is the kernel
    # scheduler placeholder on some systems; use a very high bogus number.
    lock.write_text("999999")
    assert lock.exists()

    swept = scheduler._sweep_stale_lock(lock)
    assert swept is True
    assert not lock.exists()


def test_sweep_keeps_lock_with_live_pid(tmp_path: Path):
    lock = tmp_path / "wakeup.lock"
    # Use the test process's own PID — definitely alive.
    lock.write_text(str(os.getpid()))
    swept = scheduler._sweep_stale_lock(lock)
    assert swept is False
    assert lock.exists()


def test_sweep_removes_corrupt_lock(tmp_path: Path):
    lock = tmp_path / "wakeup.lock"
    lock.write_text("not-a-pid")
    swept = scheduler._sweep_stale_lock(lock)
    assert swept is True
    assert not lock.exists()


def test_sweep_returns_false_for_missing_lock(tmp_path: Path):
    lock = tmp_path / "nonexistent.lock"
    assert not lock.exists()
    swept = scheduler._sweep_stale_lock(lock)
    assert swept is False


def test_acquire_after_dead_pid_lock_succeeds(tmp_path: Path):
    """A stale lock from a dead PID is automatically swept on _acquire_lock."""
    lock = tmp_path / "wakeup.lock"
    lock.write_text("999999")  # dead PID

    # _acquire_lock should sweep + acquire instead of raising.
    scheduler._acquire_lock(lock)
    assert lock.exists()
    assert lock.read_text().strip() == str(os.getpid())
    scheduler._release_lock(lock)


def test_acquire_blocked_when_lock_pid_alive(tmp_path: Path):
    """A lock held by a live PID is NOT swept; _acquire_lock raises."""
    lock = tmp_path / "wakeup.lock"
    lock.write_text(str(os.getpid()))  # we are alive

    with pytest.raises(scheduler._LockHeldError):
        scheduler._acquire_lock(lock)
    # Lock untouched.
    assert lock.exists()
    lock.unlink()
