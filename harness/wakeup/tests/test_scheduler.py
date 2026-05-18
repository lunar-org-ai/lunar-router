"""Tests for harness.wakeup.scheduler."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from harness.wakeup.scheduler import (
    increment_trace_counter,
    maybe_fire,
    reset_trace_counter,
)


def test_increment_persists_across_calls(tmp_path: Path):
    counter = tmp_path / "wakeup.txt"
    assert increment_trace_counter(counter_path=counter) == 1
    assert increment_trace_counter(counter_path=counter) == 2
    assert increment_trace_counter(counter_path=counter) == 3
    assert counter.read_text().strip() == "3"


def test_reset_zeroes_the_counter(tmp_path: Path):
    counter = tmp_path / "wakeup.txt"
    increment_trace_counter(counter_path=counter)
    increment_trace_counter(counter_path=counter)
    reset_trace_counter(counter_path=counter)
    assert counter.read_text().strip() == "0"


def test_maybe_fire_below_threshold_does_not_fire(tmp_path: Path):
    counter = tmp_path / "wakeup.txt"
    lock = tmp_path / "wakeup.lock"
    fired = []
    for _ in range(4):
        maybe_fire(
            threshold=5,
            counter_path=counter,
            lock_path=lock,
            runner=lambda: fired.append(True),
        )
    # 4 increments < 5 → never fired.
    assert fired == []
    assert counter.read_text().strip() == "4"


def test_maybe_fire_at_threshold_fires_once(tmp_path: Path):
    counter = tmp_path / "wakeup.txt"
    lock = tmp_path / "wakeup.lock"
    done = threading.Event()

    def runner():
        done.set()

    for _ in range(5):
        maybe_fire(
            threshold=5,
            counter_path=counter,
            lock_path=lock,
            runner=runner,
        )
    assert done.wait(timeout=2.0)
    # Counter reset on fire.
    assert counter.read_text().strip() == "0"
    # Lock released after the runner finishes.
    time.sleep(0.05)
    assert not lock.exists()


def test_concurrent_wakeups_only_run_once(tmp_path: Path):
    """When the lockfile is held, the second maybe_fire skips."""
    counter = tmp_path / "wakeup.txt"
    lock = tmp_path / "wakeup.lock"
    fired = []
    block_release = threading.Event()

    def slow_runner():
        # Hold the lock until the test releases.
        fired.append(True)
        block_release.wait(timeout=2.0)

    # First fire — hits the threshold and starts running.
    for _ in range(5):
        maybe_fire(
            threshold=5, counter_path=counter, lock_path=lock, runner=slow_runner
        )

    # Wait until the runner started (lock created + fired entry recorded).
    deadline = time.time() + 2.0
    while time.time() < deadline and not fired:
        time.sleep(0.02)
    assert fired == [True]

    # Second wave: increments past threshold but the lock is still held.
    for _ in range(10):
        maybe_fire(
            threshold=5,
            counter_path=counter,
            lock_path=lock,
            runner=lambda: fired.append("second"),
        )

    # Let the first runner finish.
    block_release.set()
    time.sleep(0.1)

    # Only one runner should have been invoked.
    assert fired == [True]


def test_lockfile_released_on_runner_failure(tmp_path: Path):
    counter = tmp_path / "wakeup.txt"
    lock = tmp_path / "wakeup.lock"

    def boom():
        raise RuntimeError("simulated failure")

    for _ in range(3):
        maybe_fire(
            threshold=3, counter_path=counter, lock_path=lock, runner=boom
        )

    # Wait for thread to finish + release.
    time.sleep(0.2)
    assert not lock.exists()
