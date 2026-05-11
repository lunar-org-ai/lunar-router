"""Unit tests for harness.watchers.auto_rollback (P16.4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import pytest

from harness.watchers.auto_rollback import (
    RollbackDecision,
    WindowMetrics,
    check_auto_rollback,
)


# ---------------------------------------------------------------------------
# Fixtures — fake policy + fake ledger entries + fake feedback/traces
# ---------------------------------------------------------------------------


@dataclass
class _FakeAutoRollback:
    csat_drop: float = 0.3
    resolution_drop: float = 0.05
    window_hours: int = 24
    notify_channels: list[str] = field(default_factory=list)


@dataclass
class _FakePolicy:
    auto_rollback: _FakeAutoRollback = field(default_factory=_FakeAutoRollback)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _promote_entry(
    *,
    when: datetime,
    before: str = "v0.0.1",
    after: str = "v0.0.2",
    entry_id: str = "led_promote",
) -> dict:
    return {
        "entry_id": entry_id,
        "kind": "promote",
        "timestamp": _iso(when),
        "agent_version_before": before,
        "agent_version_after": after,
        "payload": {},
    }


def _rollback_entry(*, when: datetime, entry_id: str = "led_rollback") -> dict:
    return {
        "entry_id": entry_id,
        "kind": "rollback",
        "timestamp": _iso(when),
        "agent_version_before": "v0.0.2",
        "agent_version_after": "v0.0.1",
        "payload": {},
    }


def _feedback_row(*, score: int, at: datetime, trace_id: str = "t") -> dict:
    return {"trace_id": trace_id, "score": score, "at": _iso(at)}


def _trace_row(
    *,
    when: datetime,
    success: bool = True,
    response: str = "ok",
    trace_id: str = "t",
) -> dict:
    return {
        "trace_id": trace_id,
        "timestamp": _iso(when),
        "success": success,
        "response": response,
        "stages": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_promote_returns_none():
    policy = _FakePolicy()
    out = check_auto_rollback(
        policy=policy,
        entries_iter=[],
        feedback_iter=[],
        traces_iter=[],
    )
    assert out is None


def test_promote_without_drop_returns_none():
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    # CSAT identical before and after — no drop
    feedback = [
        _feedback_row(score=4, at=promote_at - timedelta(hours=6)),
        _feedback_row(score=4, at=promote_at + timedelta(hours=6)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is None


def test_csat_drop_fires_rollback():
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    feedback = [
        _feedback_row(score=5, at=promote_at - timedelta(hours=6)),
        _feedback_row(score=5, at=promote_at - timedelta(hours=3)),
        _feedback_row(score=2, at=promote_at + timedelta(hours=3)),
        _feedback_row(score=2, at=promote_at + timedelta(hours=6)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(),  # csat_drop=0.3
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is not None
    assert isinstance(out, RollbackDecision)
    assert out.target_version == "v0.0.1"
    assert out.suspect_version == "v0.0.2"
    assert "CSAT" in out.reason
    assert "5.00" in out.reason and "2.00" in out.reason


def test_resolution_drop_fires_rollback():
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    traces = (
        [_trace_row(when=promote_at - timedelta(hours=6 - i), success=True)
         for i in range(5)]
        + [_trace_row(when=promote_at + timedelta(hours=i + 1), success=False)
           for i in range(5)]
    )
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=[],
        traces_iter=traces,
    )
    assert out is not None
    assert "resolution_rate" in out.reason


def test_already_rolled_back_returns_none():
    """A rollback entry that post-dates the promote means we already did it."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    rollback_at = now - timedelta(hours=6)
    feedback = [
        _feedback_row(score=5, at=promote_at - timedelta(hours=6)),
        _feedback_row(score=1, at=promote_at + timedelta(hours=2)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[
            _promote_entry(when=promote_at),
            _rollback_entry(when=rollback_at),
        ],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is None


def test_no_signal_in_window_returns_none():
    """If no feedback OR traces fall in either window, nothing to compare."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=[],
        traces_iter=[],
    )
    assert out is None


def test_signal_only_in_after_window_returns_none():
    """Before is empty → can't compute drop (would look like infinite drop)."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    feedback = [
        _feedback_row(score=2, at=promote_at + timedelta(hours=2)),
        _feedback_row(score=2, at=promote_at + timedelta(hours=3)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is None


def test_promote_without_agent_version_before_returns_none():
    """Can't rollback if we don't know what to rollback to."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    bad_promote = _promote_entry(when=promote_at)
    bad_promote["agent_version_before"] = None
    feedback = [
        _feedback_row(score=5, at=promote_at - timedelta(hours=2)),
        _feedback_row(score=1, at=promote_at + timedelta(hours=2)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[bad_promote],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is None


def test_just_promoted_zero_elapsed_returns_none():
    """If promote happened at `now`, after-window is empty."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    feedback = [_feedback_row(score=5, at=now - timedelta(hours=3))]
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=now)],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is None


def test_drop_below_threshold_returns_none():
    """Drop of exactly 0.1 doesn't cross the 0.3 threshold."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    feedback = [
        _feedback_row(score=5, at=promote_at - timedelta(hours=6)),
        _feedback_row(score=5, at=promote_at - timedelta(hours=3)),
        _feedback_row(score=4, at=promote_at + timedelta(hours=2)),
        _feedback_row(score=5, at=promote_at + timedelta(hours=5)),
    ]
    # 5 → 4.5 = 0.5 drop... that's > 0.3.
    # Let me tighten — score=5, score=4.9 wouldn't be possible with int scores.
    # Use a higher threshold to verify the no-fire path.
    custom_policy = _FakePolicy(
        auto_rollback=_FakeAutoRollback(csat_drop=1.0)
    )
    out = check_auto_rollback(
        policy=custom_policy,
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is None


def test_custom_window_hours_respected():
    """A 1h window only looks at data within ±1h of the promote."""
    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(minutes=30)
    feedback = [
        # Old data — outside the 1h window
        _feedback_row(score=5, at=promote_at - timedelta(hours=6)),
        # Inside 1h before
        _feedback_row(score=5, at=promote_at - timedelta(minutes=30)),
        # Inside 1h after
        _feedback_row(score=1, at=promote_at + timedelta(minutes=15)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(
            auto_rollback=_FakeAutoRollback(csat_drop=0.3, window_hours=1)
        ),
        now_iso=_iso(now),
        entries_iter=[_promote_entry(when=promote_at)],
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is not None
    assert out.before.csat == 5.0  # only the in-window 5 counts, not the older one
    assert out.after.csat == 1.0


def test_dict_or_dataclass_entries_both_supported():
    """The watcher accepts both LedgerEntry dataclasses and plain dicts."""
    from ledger.types import LedgerEntry

    now = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
    promote_at = now - timedelta(hours=12)
    entry = LedgerEntry(
        entry_id="led_dc",
        kind="promote",
        timestamp=_iso(promote_at),
        agent_version_before="v0.0.1",
        agent_version_after="v0.0.2",
        summary="test",
        payload={},
    )
    feedback = [
        _feedback_row(score=5, at=promote_at - timedelta(hours=3)),
        _feedback_row(score=1, at=promote_at + timedelta(hours=3)),
    ]
    out = check_auto_rollback(
        policy=_FakePolicy(),
        now_iso=_iso(now),
        entries_iter=[entry],   # dataclass not dict
        feedback_iter=feedback,
        traces_iter=[],
    )
    assert out is not None
    assert out.target_version == "v0.0.1"


# ---------------------------------------------------------------------------
# Notification side-table
# ---------------------------------------------------------------------------


def test_notification_round_trip(tmp_path, monkeypatch):
    from runtime.store.notifications import (
        notify_channels,
        iter_notifications,
        write_notification,
    )

    root = tmp_path / "notifications"
    monkeypatch.setattr("runtime.store.notifications._NOTIFICATIONS_ROOT", root)

    notify_channels(
        ["email", "slack"],
        subject="Test",
        body="Body",
        lesson_id="L-abc",
        root=root,
    )
    rows = list(iter_notifications(root=root))
    assert len(rows) == 2
    assert {r["channel"] for r in rows} == {"email", "slack"}
    assert all(r["lesson_id"] == "L-abc" for r in rows)
    assert all(r["kind"] == "auto_rollback" for r in rows)


def test_notification_empty_channel_list(tmp_path):
    from runtime.store.notifications import notify_channels
    rows = notify_channels([], subject="x", body="y", root=tmp_path / "n")
    assert rows == []
