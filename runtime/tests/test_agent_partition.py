"""Tests for P2.1 storage partitioning by agent_id."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_agent_context():
    """Each test starts with no process-global active agent. Tests that
    need a specific id call set_active themselves."""
    from runtime import agent_context
    agent_context.set_active(None)
    yield
    agent_context.set_active(None)


# ---------------------------------------------------------------------------
# agent_context module
# ---------------------------------------------------------------------------


def test_get_active_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("OPENTRACY_AGENT_ID", raising=False)
    from runtime.agent_context import get_active
    assert get_active() == "_default"


def test_get_active_reads_process_global():
    from runtime.agent_context import get_active, set_active
    set_active("shopify-support")
    assert get_active() == "shopify-support"


def test_get_active_env_override_when_no_process_global(monkeypatch):
    monkeypatch.setenv("OPENTRACY_AGENT_ID", "research")
    from runtime.agent_context import get_active
    assert get_active() == "research"


def test_process_global_beats_env(monkeypatch):
    monkeypatch.setenv("OPENTRACY_AGENT_ID", "from-env")
    from runtime.agent_context import get_active, set_active
    set_active("from-process")
    assert get_active() == "from-process"


# ---------------------------------------------------------------------------
# Writers partition by active agent (no override)
# ---------------------------------------------------------------------------


def test_ledger_writer_partitions_entries_by_active_agent(tmp_path, monkeypatch):
    """write_entry honors the active agent: ledger/<active>/entries/<date>.jsonl."""
    monkeypatch.setattr("ledger.writer._LEDGER_ROOT", tmp_path / "ledger")
    from runtime.agent_context import set_active
    set_active("agent-a")

    from ledger.writer import write_entry
    write_entry("proposal", summary="hi", payload={})

    a_dir = tmp_path / "ledger" / "agent-a" / "entries"
    assert a_dir.is_dir()
    files = list(a_dir.glob("*.jsonl"))
    assert files, "expected one jsonl file"

    # Switching agents writes to a different partition
    set_active("agent-b")
    write_entry("proposal", summary="hello", payload={})
    b_dir = tmp_path / "ledger" / "agent-b" / "entries"
    assert b_dir.is_dir()
    assert list(b_dir.glob("*.jsonl"))
    # The two partitions are isolated
    assert a_dir != b_dir


def test_ledger_writer_partitions_lessons(tmp_path, monkeypatch):
    monkeypatch.setattr("ledger.writer._LEDGER_ROOT", tmp_path / "ledger")
    from runtime.agent_context import set_active
    set_active("agent-x")
    from ledger.types import Lesson
    from ledger.writer import write_lesson

    write_lesson(
        Lesson(
            id="L-xxx", version="v0.0.1", kind="prompt", status="approved",
            title="t", summary="s", voice="v",
            proposal_source="human", delta={}, mutations=[],
            parent_version=None, candidate_id=None,
        ),
    )
    assert (tmp_path / "ledger" / "agent-x" / "lessons" / "L-xxx.json").is_file()


def test_feedback_writer_partitions_by_active_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from runtime.agent_context import set_active
    set_active("agent-y")
    from runtime.store.feedback import write_feedback
    write_feedback("trace-1", score=5)
    assert any(
        p.is_file() for p in (tmp_path / "traces" / "agent-y" / "feedback").glob("*.jsonl")
    )


def test_flags_writer_partitions_by_active_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from runtime.agent_context import set_active
    set_active("agent-z")
    from runtime.store.flags import write_flag
    write_flag("trace-1", source="manual", reason="testing")
    assert any(
        p.is_file() for p in (tmp_path / "traces" / "agent-z" / "flagged").glob("*.jsonl")
    )


def test_notifications_partition_by_active_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from runtime.agent_context import set_active
    set_active("agent-q")
    from runtime.store.notifications import write_notification
    write_notification(channel="log", subject="s", body="b")
    assert any(
        p.is_file() for p in (tmp_path / "ledger" / "agent-q" / "notifications").glob("*.jsonl")
    )


# ---------------------------------------------------------------------------
# Migration in ensure_bootstrapped
# ---------------------------------------------------------------------------


def test_bootstrap_migrates_flat_ledger_and_traces(tmp_path, monkeypatch):
    """Pre-existing flat dirs get moved into <root>/_default/<kind>/."""
    monkeypatch.chdir(tmp_path)

    # Set up flat layout that pre-dates P2.1
    (tmp_path / "ledger" / "entries").mkdir(parents=True)
    (tmp_path / "ledger" / "entries" / "2026-01-01.jsonl").write_text(
        '{"entry_id":"led_old","kind":"start","timestamp":"x"}\n'
    )
    (tmp_path / "ledger" / "lessons").mkdir()
    (tmp_path / "ledger" / "lessons" / "L-old.json").write_text("{}")
    (tmp_path / "traces" / "raw").mkdir(parents=True)
    (tmp_path / "traces" / "raw" / "2026-01-01.jsonl").write_text('{"trace_id":"old"}\n')

    # Live agent dir so bootstrap can seed _default with it
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.yaml").write_text("agent:\n  version: v0\n")

    from runtime.agents.registry import ensure_bootstrapped
    ensure_bootstrapped(
        root=tmp_path / "agents",
        live_dir=tmp_path / "agent",
        project_root=tmp_path,
    )

    # Flat dirs are gone, partitioned dirs hold the data
    assert not (tmp_path / "ledger" / "entries").exists()
    assert (tmp_path / "ledger" / "_default" / "entries" / "2026-01-01.jsonl").is_file()

    assert not (tmp_path / "ledger" / "lessons").exists()
    assert (tmp_path / "ledger" / "_default" / "lessons" / "L-old.json").is_file()

    assert not (tmp_path / "traces" / "raw").exists()
    assert (tmp_path / "traces" / "_default" / "raw" / "2026-01-01.jsonl").is_file()


def test_bootstrap_migration_is_idempotent(tmp_path, monkeypatch):
    """Running twice doesn't corrupt the data."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ledger" / "entries").mkdir(parents=True)
    (tmp_path / "ledger" / "entries" / "x.jsonl").write_text("{}\n")
    (tmp_path / "agent").mkdir()

    from runtime.agents.registry import ensure_bootstrapped
    ensure_bootstrapped(
        root=tmp_path / "agents", live_dir=tmp_path / "agent", project_root=tmp_path,
    )
    # Second call — should be a no-op for the partition
    ensure_bootstrapped(
        root=tmp_path / "agents", live_dir=tmp_path / "agent", project_root=tmp_path,
    )
    assert (tmp_path / "ledger" / "_default" / "entries" / "x.jsonl").is_file()
