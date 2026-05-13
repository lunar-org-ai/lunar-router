"""Tests for the multi-agent registry (P2.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.agents.registry import (
    activate,
    create_agent,
    delete_agent,
    ensure_bootstrapped,
    get_agent,
    get_active,
    get_registry,
    list_agents,
    update_agent,
)


@pytest.fixture
def workspace(tmp_path):
    """Tmp workspace with a pre-seeded ``agent/`` dir to migrate."""
    agent_dir = tmp_path / "agent"
    (agent_dir / "prompts").mkdir(parents=True)
    (agent_dir / "pipeline").mkdir()
    (agent_dir / "agent.yaml").write_text("agent:\n  version: v0.0.1\n")
    (agent_dir / "prompts" / "system.md").write_text("You are a helpful assistant.")
    (agent_dir / "pipeline" / "route.yaml").write_text("stage: route\n")
    return {
        "tmp": tmp_path,
        "root": tmp_path / "agents",
        "live": agent_dir,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_migrates_legacy_agent_dir(workspace):
    """First run with a legacy ``agent/`` → registry written with _default
    seeded from the live dir."""
    reg = ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])

    assert reg.active == "_default"
    assert len(reg.agents) == 1
    assert reg.agents[0].id == "_default"

    # _default got the live dir's contents
    seeded = workspace["root"] / "_default"
    assert (seeded / "agent.yaml").is_file()
    assert "You are a helpful assistant" in (seeded / "prompts" / "system.md").read_text()


def test_bootstrap_is_idempotent(workspace):
    """Running bootstrap twice doesn't duplicate _default or clobber state."""
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    # Mutate the registry between runs to verify the second call is a no-op
    update_agent("_default", name="renamed", root=workspace["root"])
    reg = ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    assert len(reg.agents) == 1
    assert reg.agents[0].name == "renamed"


def test_bootstrap_with_no_live_dir(tmp_path):
    """Cold install — no ``agent/`` yet. Bootstrap still produces a
    registry with an empty _default placeholder."""
    reg = ensure_bootstrapped(
        root=tmp_path / "agents",
        live_dir=tmp_path / "no-such-agent",
    )
    assert reg.active == "_default"
    assert (tmp_path / "agents" / "_default").is_dir()


# ---------------------------------------------------------------------------
# Create / list / get
# ---------------------------------------------------------------------------


def test_create_agent_writes_dir_and_registry(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])

    meta = create_agent(
        {
            "name": "shopify-support",
            "model": "claude-haiku-4-5",
            "prompt": "You support a Shopify store.",
            "template": "support",
            "tools": [],
            "channels": ["web"],
        },
        root=workspace["root"],
    )

    assert meta.id == "shopify-support"
    assert meta.model == "claude-haiku-4-5"

    agent_dir = workspace["root"] / "shopify-support"
    assert agent_dir.is_dir()
    assert (agent_dir / "prompts" / "system.md").is_file()
    body = (agent_dir / "prompts" / "system.md").read_text()
    assert "Shopify store" in body
    assert "trainable surface" in body
    # Pipeline copied from the seed (_default)
    assert (agent_dir / "agent.yaml").is_file()
    # Onboarding snapshot saved
    snapshot = json.loads((agent_dir / "onboarding.json").read_text())
    assert snapshot["name"] == "shopify-support"

    # Registry now has 2 entries
    reg = get_registry(root=workspace["root"])
    assert {a.id for a in reg.agents} == {"_default", "shopify-support"}
    # Active didn't change — create alone doesn't activate
    assert reg.active == "_default"


def test_create_agent_slug_collision_appends_suffix(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    create_agent({"name": "support", "prompt": "x"}, root=workspace["root"])
    second = create_agent({"name": "support", "prompt": "y"}, root=workspace["root"])
    assert second.id != "support"
    assert second.id.startswith("support-")


def test_list_agents_returns_all(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    create_agent({"name": "a", "prompt": "x"}, root=workspace["root"])
    create_agent({"name": "b", "prompt": "y"}, root=workspace["root"])
    agents = list_agents(root=workspace["root"])
    assert {a.id for a in agents} == {"_default", "a", "b"}


def test_get_agent_missing_returns_none(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    assert get_agent("nonexistent", root=workspace["root"]) is None


# ---------------------------------------------------------------------------
# Activate
# ---------------------------------------------------------------------------


def test_activate_swaps_live_dir_and_updates_active(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    create_agent(
        {"name": "new-agent", "prompt": "New prompt body."},
        root=workspace["root"],
    )

    hook_called: list[str] = []
    meta = activate(
        "new-agent",
        root=workspace["root"],
        live_dir=workspace["live"],
        on_activate=lambda m: hook_called.append(m.id),
    )

    assert meta.id == "new-agent"
    assert get_active(root=workspace["root"]).id == "new-agent"
    assert hook_called == ["new-agent"]
    # The live agent dir now has the new prompt
    body = (workspace["live"] / "prompts" / "system.md").read_text()
    assert "New prompt body" in body


def test_activate_unknown_agent_raises(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    with pytest.raises(KeyError):
        activate("nonexistent", root=workspace["root"], live_dir=workspace["live"])


def test_activate_preserves_live_state_into_previous_active(workspace):
    """Switching from A to B snapshots A's live state back into agents/A/
    so unsaved edits aren't lost."""
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    create_agent({"name": "b", "prompt": "B prompt"}, root=workspace["root"])

    # Operator edits the live prompt while _default is active
    (workspace["live"] / "prompts" / "system.md").write_text("Edited live prompt.")

    # Switch to b
    activate("b", root=workspace["root"], live_dir=workspace["live"])

    # _default's saved state should now contain the edit
    saved = (workspace["root"] / "_default" / "prompts" / "system.md").read_text()
    assert "Edited live prompt" in saved
    # And the live dir now has b's prompt
    assert "B prompt" in (workspace["live"] / "prompts" / "system.md").read_text()


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_soft_deletes_and_drops_from_registry(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    create_agent({"name": "to-go", "prompt": "x"}, root=workspace["root"])
    delete_agent("to-go", root=workspace["root"])
    assert get_agent("to-go", root=workspace["root"]) is None
    # Soft delete moved files into _deleted/
    bucket = workspace["root"] / "_deleted"
    assert bucket.is_dir()
    survivors = list(bucket.iterdir())
    assert any("to-go" in s.name for s in survivors)


def test_cannot_delete_default(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    with pytest.raises(ValueError):
        delete_agent("_default", root=workspace["root"])


def test_cannot_delete_active(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    create_agent({"name": "x", "prompt": "y"}, root=workspace["root"])
    activate("x", root=workspace["root"], live_dir=workspace["live"])
    with pytest.raises(ValueError):
        delete_agent("x", root=workspace["root"])


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_agent_changes_metadata(workspace):
    ensure_bootstrapped(root=workspace["root"], live_dir=workspace["live"])
    update_agent("_default", name="My Agent", description="hello", root=workspace["root"])
    meta = get_agent("_default", root=workspace["root"])
    assert meta.name == "My Agent"
    assert meta.description == "hello"
