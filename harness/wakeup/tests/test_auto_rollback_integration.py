"""Integration tests for P16.4 — run_wakeup() invokes the auto-rollback
watcher BEFORE the brain, executes rollback when triggered, persists a
Lesson + notifications, and short-circuits with action='rolled_back'.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.watchers.auto_rollback import RollbackDecision, WindowMetrics


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """Redirect ledger writes to tmp so artifacts don't pollute the repo."""
    import ledger.writer as lw

    entries = tmp_path / "ledger" / "entries"
    lessons = tmp_path / "ledger" / "lessons"
    decisions = tmp_path / "ledger" / "decisions"
    for d in (entries, lessons, decisions):
        d.mkdir(parents=True)
    monkeypatch.setattr("ledger.writer.ENTRIES_DIR", entries)
    monkeypatch.setattr("ledger.writer.LESSONS_DIR", lessons)
    monkeypatch.setattr("ledger.writer.DECISIONS_DIR", decisions)

    we_kw = dict(lw.write_entry.__kwdefaults__ or {})
    if "entries_dir" in we_kw:
        we_kw["entries_dir"] = entries
        monkeypatch.setattr(lw.write_entry, "__kwdefaults__", we_kw)
    wl_d = list(lw.write_lesson.__defaults__ or ())
    if wl_d:
        wl_d[-1] = lessons
        monkeypatch.setattr(lw.write_lesson, "__defaults__", tuple(wl_d))
    wd_kw = dict(lw.write_decision.__kwdefaults__ or {})
    if "decisions_dir" in wd_kw:
        wd_kw["decisions_dir"] = decisions
        monkeypatch.setattr(lw.write_decision, "__kwdefaults__", wd_kw)
    return tmp_path / "ledger"


@pytest.fixture
def tmp_notifications(tmp_path, monkeypatch):
    root = tmp_path / "notifications"
    monkeypatch.setattr("runtime.store.notifications._NOTIFICATIONS_ROOT", root)
    return root


def _decision() -> RollbackDecision:
    return RollbackDecision(
        target_version="v0.0.1",
        suspect_version="v0.0.2",
        reason="CSAT dropped 4.50 → 2.10 (threshold 0.30, window 24h)",
        before=WindowMetrics(csat=4.5, resolution_rate=0.95, n_traces=20, n_feedback=10),
        after=WindowMetrics(csat=2.1, resolution_rate=0.40, n_traces=20, n_feedback=8),
        promote_entry_id="led_promote_abc",
        promote_timestamp="2026-05-12T00:00:00Z",
    )


def test_wakeup_executes_rollback_when_watcher_fires(
    tmp_ledger, tmp_notifications, monkeypatch,
):
    """End-to-end: watcher fires → rollback_to() runs → Lesson written → notifications persisted."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")  # avoid no-brain block

    # Stub the watcher to return a decision (the watcher itself is
    # tested at the unit level; here we just verify the wiring).
    monkeypatch.setattr(
        "harness.watchers.auto_rollback.check_auto_rollback",
        lambda **kw: _decision(),
    )

    # Stub rollback_to to record the call without mutating the real agent dir.
    rollback_calls: list[tuple[str, str]] = []
    def _fake_rollback(version, *, reason="", **kw):
        rollback_calls.append((version, reason))
        return version
    monkeypatch.setattr("harness.rollback.rollback.rollback_to", _fake_rollback)

    # Stub policy so the notify_channels list is non-empty.
    from harness.approver.policy import AutoRollback, Policy
    monkeypatch.setattr(
        Policy, "from_yaml",
        classmethod(lambda cls, *a, **kw: Policy(
            auto_rollback=AutoRollback(notify_channels=["email", "log"])
        )),
    )

    # Stub the health calls so we don't need a real router config / dataset registry.
    class _H:
        def to_dict(self): return {"cold_start": True}
    monkeypatch.setattr(
        "router.feedback.health.compute_router_health",
        lambda **kw: _H(),
    )
    monkeypatch.setattr(
        "harness.wakeup.runner._safe_dataset_health",
        lambda: {"datasets": []},
    )

    from harness.wakeup.runner import run_wakeup

    outcome = run_wakeup(
        # introspect should NEVER be called when rollback fires
        introspect_fn=lambda _: pytest.fail("brain should not be invoked during rollback"),
    )

    assert outcome.action == "rolled_back"
    assert outcome.target == "auto_rollback"
    assert outcome.lesson_id.startswith("L-")
    assert "CSAT dropped" in outcome.rationale
    assert outcome.rollback_metadata is not None
    assert outcome.rollback_metadata["from_version"] == "v0.0.2"
    assert outcome.rollback_metadata["to_version"] == "v0.0.1"

    # rollback_to was called
    assert rollback_calls == [("v0.0.1", "auto-rollback: CSAT dropped 4.50 → 2.10 (threshold 0.30, window 24h)")]

    # Lesson written with kind=rollback
    lesson_files = list((tmp_ledger / "lessons").glob("*.json"))
    assert len(lesson_files) == 1
    lesson = json.loads(lesson_files[0].read_text())
    assert lesson["kind"] == "rollback"
    assert lesson["status"] == "rolled_back"
    assert lesson["proposal_source"] == "auto"
    assert lesson["delta"]["csat_after"] == 2.1

    # Notifications written for each channel
    notifs = list(tmp_notifications.glob("*.jsonl"))
    assert len(notifs) == 1
    rows = [json.loads(l) for l in notifs[0].read_text().strip().split("\n")]
    assert {r["channel"] for r in rows} == {"email", "log"}
    assert all(r["lesson_id"] == outcome.lesson_id for r in rows)

    # Decision artifact carries the rolled_back action
    decision_files = list((tmp_ledger / "decisions").glob("*.json"))
    assert len(decision_files) == 1
    decision = json.loads(decision_files[0].read_text())
    assert decision["payload"]["action"] == "rolled_back"


def test_wakeup_continues_normally_when_watcher_returns_none(
    tmp_ledger, tmp_notifications, monkeypatch,
):
    """Watcher returns None → wake-up proceeds with brain flow (skipped)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setattr(
        "harness.watchers.auto_rollback.check_auto_rollback",
        lambda **kw: None,
    )
    class _H:
        def to_dict(self): return {"cold_start": True}
    monkeypatch.setattr(
        "router.feedback.health.compute_router_health",
        lambda **kw: _H(),
    )
    monkeypatch.setattr(
        "harness.wakeup.runner._safe_dataset_health",
        lambda: {"datasets": []},
    )

    from harness.wakeup.runner import run_wakeup
    from dataclasses import dataclass, field

    @dataclass
    class _FakeIntro:
        response: str = "skipping for now."
        tool_calls: list = field(default_factory=list)

    outcome = run_wakeup(introspect_fn=lambda _: _FakeIntro())
    assert outcome.action == "skipped"
    assert outcome.rollback_metadata is None
    # No notifications
    assert list(tmp_notifications.glob("*.jsonl")) == []


def test_wakeup_swallows_watcher_exception(
    tmp_ledger, tmp_notifications, monkeypatch,
):
    """Watcher crash must NEVER block the wakeup. Safety must be additive."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    def _boom(**kw): raise RuntimeError("watcher crashed")
    monkeypatch.setattr(
        "harness.watchers.auto_rollback.check_auto_rollback", _boom,
    )
    class _H:
        def to_dict(self): return {"cold_start": True}
    monkeypatch.setattr(
        "router.feedback.health.compute_router_health",
        lambda **kw: _H(),
    )
    monkeypatch.setattr(
        "harness.wakeup.runner._safe_dataset_health",
        lambda: {"datasets": []},
    )

    from harness.wakeup.runner import run_wakeup
    from dataclasses import dataclass, field

    @dataclass
    class _FakeIntro:
        response: str = "all good"
        tool_calls: list = field(default_factory=list)

    outcome = run_wakeup(introspect_fn=lambda _: _FakeIntro())
    # The watcher crashed silently; the wake-up continued normally.
    assert outcome.action == "skipped"
