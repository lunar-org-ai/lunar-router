"""Tests for runtime.mcp.per_agent_tools — the workspace-scoped tools
the customer's Claude Code calls over MCP.

These tests exercise the *handlers* directly (bypassing the MCP wire
protocol) — that path is covered by tests for the SDK plumbing in
test_mcp_per_agent.py. Here we want to lock in the contract each tool
returns and the cross-tenant isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.mcp.per_agent_tools import (
    HANDLERS,
    call_tool,
    list_manifest_history,
    list_nexau_components,
    list_workspace_files,
    read_pending_manifest,
    read_plan,
    read_state,
    read_system_prompt,
    read_workspace_file,
    send_task,
)


# ---------------------------------------------------------------------------
# Fixtures — pin agent_context + tmp workspace
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Provide an isolated workspace for 'demo' and set agent_context."""
    from runtime.workspaces import store as ws_store

    monkeypatch.setattr(
        ws_store, "_agents_root", lambda root=None: root if root else tmp_path,
    )
    (tmp_path / "demo").mkdir()

    from runtime import agent_context as agent_ctx
    agent_ctx.set_active("demo")
    yield ws_store.get_workspace("demo", root=tmp_path)
    agent_ctx.set_active(None)


# ---------------------------------------------------------------------------
# Workspace read-only tools
# ---------------------------------------------------------------------------


def test_read_plan_returns_seed_content(workspace):
    out = read_plan()
    assert "plan_markdown" in out
    assert "No plan yet" in out["plan_markdown"]


def test_read_plan_returns_evolved_content(workspace):
    (workspace.path / ".opentracy" / "memory" / "plan.md").write_text(
        "# new plan\n\nstep 1\n", encoding="utf-8",
    )
    assert "step 1" in read_plan()["plan_markdown"]


def test_read_state_returns_seed_dict(workspace):
    out = read_state()
    assert out["state"]["next_step"] is None
    assert out["state"]["facts"] == []
    assert out["state"]["blockers"] == []


def test_read_system_prompt_returns_seeded_default(workspace):
    out = read_system_prompt()
    assert "autonomous engineer" in out["system_prompt"].lower()


def test_list_workspace_files_lists_seeded_layout(workspace):
    out = list_workspace_files()
    assert out["count"] >= 3
    # NexAU seeds must show up.
    assert ".opentracy/system_prompt.md" in out["files"]
    assert ".opentracy/memory/plan.md" in out["files"]
    assert ".opentracy/memory/state.json" in out["files"]


def test_read_workspace_file_returns_content(workspace):
    (workspace.path / "notes.md").write_text("hello world", encoding="utf-8")
    out = read_workspace_file(path="notes.md")
    assert out["path"] == "notes.md"
    assert out["content"] == "hello world"
    assert out["size"] == 11


def test_read_workspace_file_rejects_absolute_paths(workspace):
    out = read_workspace_file(path="/etc/passwd")
    assert out["error"] == "path_traversal"


def test_read_workspace_file_rejects_dotdot_traversal(workspace):
    out = read_workspace_file(path="../escape")
    assert out["error"] == "path_traversal"


def test_read_workspace_file_returns_not_found_for_missing(workspace):
    out = read_workspace_file(path="never-existed.txt")
    assert out["error"] == "not_found"


def test_read_workspace_file_handles_binary(workspace):
    (workspace.path / "blob.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
    out = read_workspace_file(path="blob.bin")
    assert out["error"] == "binary"
    assert out["size"] == 5


# ---------------------------------------------------------------------------
# NexAU + Change Manifest tools
# ---------------------------------------------------------------------------


def test_list_nexau_components_baseline(workspace):
    out = list_nexau_components()
    comp = out["components"]
    assert comp["system_prompt"] == ["system_prompt.md"]
    assert comp["tools"] == []
    assert comp["middleware"] == []
    assert comp["skills"] == []
    assert comp["subagents"] == []
    assert sorted(comp["memory"]) == ["plan.md", "state.json"]


def test_read_pending_manifest_returns_null_when_unset(workspace):
    assert read_pending_manifest() == {"pending": None}


def test_read_pending_manifest_returns_written(workspace):
    workspace.write_pending_manifest({
        "changed_files": [".opentracy/skills/plan_first.md"],
        "claimed_fixes": ["agent skips planning on cold tasks"],
    })
    out = read_pending_manifest()
    assert out["pending"]["claimed_fixes"] == ["agent skips planning on cold tasks"]


def test_list_manifest_history_returns_archived_with_verdicts(workspace):
    workspace.write_pending_manifest({"claimed_fixes": ["A"]})
    workspace.roll_pending_to_history(outcome={"verdict": "confirmed"})

    out = list_manifest_history()
    assert len(out["entries"]) == 1
    assert out["entries"][0]["claimed_fixes"] == ["A"]
    assert out["entries"][0]["outcome"]["verdict"] == "confirmed"


# ---------------------------------------------------------------------------
# Write tool — send_task
# ---------------------------------------------------------------------------


def test_send_task_writes_inbox_entry(workspace):
    out = send_task(task="ship the email drip campaign")
    assert "queued_at" in out
    assert out["path"].startswith(".opentracy/inbox/")
    inbox_dir = workspace.path / ".opentracy" / "inbox"
    files = list(inbox_dir.iterdir())
    assert len(files) == 1
    assert "drip campaign" in files[0].read_text(encoding="utf-8")


def test_send_task_rejects_empty(workspace):
    out = send_task(task="   \n\t  ")
    assert out["error"] == "empty_task"
    assert not (workspace.path / ".opentracy" / "inbox").exists() or \
           list((workspace.path / ".opentracy" / "inbox").iterdir()) == []


# ---------------------------------------------------------------------------
# Context safety + dispatch
# ---------------------------------------------------------------------------


def test_tool_falls_back_to_default_when_agent_context_missing(
    tmp_path, monkeypatch, caplog,
):
    """When agent_context isn't set explicitly, the tool quietly falls
    back to ``_default`` (matches the existing OSS behavior) and emits
    a debug log so misuse in multi-tenant surfaces in traces."""
    from runtime import agent_context as agent_ctx
    from runtime.workspaces import store as ws_store

    monkeypatch.setattr(
        ws_store, "_agents_root", lambda root=None: root if root else tmp_path,
    )
    (tmp_path / "_default").mkdir()
    agent_ctx.set_active(None)

    import logging
    with caplog.at_level(logging.DEBUG, logger="runtime.mcp.per_agent_tools"):
        out = read_plan()
    assert "plan_markdown" in out
    assert any(
        "falling back to _default" in record.message
        for record in caplog.records
    )


def test_call_tool_dispatches_known_tool(workspace):
    out = call_tool("read_plan", {})
    assert len(out) == 1
    payload = json.loads(out[0].text)
    assert "plan_markdown" in payload


def test_call_tool_returns_unknown_marker(workspace):
    out = call_tool("does_not_exist", {})
    assert "unknown tool" in out[0].text


def test_call_tool_surfaces_handler_errors_as_text(workspace):
    out = call_tool("read_workspace_file", {"path": "/etc/shadow"})
    # Handler returns its own structured error — call_tool wraps in JSON.
    payload = json.loads(out[0].text)
    assert payload["error"] == "path_traversal"


def test_handlers_registry_matches_declared_tools(workspace):
    from runtime.mcp.per_agent_tools import TOOLS
    declared = {t.name for t in TOOLS}
    assert declared == set(HANDLERS.keys())
