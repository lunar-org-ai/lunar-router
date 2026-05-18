"""Tests for the per-agent improvement brain config (P3.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.agents.improvement import (
    ImprovementConfig,
    is_enabled,
    load,
    resolve_for_active_agent,
    save,
)


@pytest.fixture(autouse=True)
def reset_agent_context():
    from runtime import agent_context
    agent_context.set_active(None)
    yield
    agent_context.set_active(None)


# ---------------------------------------------------------------------------
# Defaults + round-trip
# ---------------------------------------------------------------------------


def test_load_missing_returns_defaults(tmp_path):
    cfg = load("ghost", root=tmp_path)
    assert cfg.enabled is True
    assert cfg.transport == "auto"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.cadence_minutes == 30


def test_save_and_load_round_trip(tmp_path):
    src = ImprovementConfig(
        enabled=False,
        transport="claude_code_cli",
        model="claude-opus-4-7",
        cadence_minutes=120,
        notes="paused while we audit",
    )
    save("agent-a", src, root=tmp_path)
    loaded = load("agent-a", root=tmp_path)
    assert loaded.enabled is False
    assert loaded.transport == "claude_code_cli"
    assert loaded.model == "claude-opus-4-7"
    assert loaded.cadence_minutes == 120
    assert loaded.notes == "paused while we audit"


def test_load_normalizes_unknown_transport(tmp_path):
    p = tmp_path / "agent-a" / "improvement.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("transport: galactic\nmodel: claude-haiku-4-5\n")
    cfg = load("agent-a", root=tmp_path)
    # Unknown transport → falls back to 'auto'
    assert cfg.transport == "auto"
    assert cfg.model == "claude-haiku-4-5"


def test_load_handles_malformed_yaml(tmp_path):
    p = tmp_path / "x" / "improvement.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(":::: not valid yaml")
    cfg = load("x", root=tmp_path)
    # Garbage → defaults, not crash
    assert cfg.transport == "auto"


def test_load_handles_non_mapping_yaml(tmp_path):
    p = tmp_path / "x" / "improvement.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("- just\n- a\n- list\n")
    cfg = load("x", root=tmp_path)
    assert cfg.enabled is True  # defaults


# ---------------------------------------------------------------------------
# resolve_for_active_agent
# ---------------------------------------------------------------------------


def test_resolve_uses_active_agent(tmp_path):
    from runtime import agent_context
    save("agent-a", ImprovementConfig(model="claude-opus-4-7"), root=tmp_path)
    agent_context.set_active("agent-a")
    cfg = resolve_for_active_agent(root=tmp_path)
    assert cfg.model == "claude-opus-4-7"


def test_resolve_falls_back_to_defaults_when_no_active(tmp_path):
    cfg = resolve_for_active_agent(root=tmp_path)
    assert cfg.transport == "auto"


def test_resolve_uses_default_transport_when_config_is_auto(tmp_path):
    from runtime import agent_context
    save("agent-b", ImprovementConfig(transport="auto"), root=tmp_path)
    agent_context.set_active("agent-b")
    cfg = resolve_for_active_agent(default_transport="claude_code_cli", root=tmp_path)
    assert cfg.transport == "claude_code_cli"


def test_resolve_keeps_explicit_transport_over_default(tmp_path):
    from runtime import agent_context
    save("agent-c", ImprovementConfig(transport="anthropic_api"), root=tmp_path)
    agent_context.set_active("agent-c")
    cfg = resolve_for_active_agent(default_transport="claude_code_cli", root=tmp_path)
    assert cfg.transport == "anthropic_api"


def test_is_enabled_respects_flag(tmp_path):
    from runtime import agent_context
    save("d", ImprovementConfig(enabled=False), root=tmp_path)
    agent_context.set_active("d")
    assert is_enabled(root=tmp_path) is False


def test_is_enabled_respects_disabled_transport(tmp_path):
    from runtime import agent_context
    save("e", ImprovementConfig(enabled=True, transport="disabled"), root=tmp_path)
    agent_context.set_active("e")
    assert is_enabled(root=tmp_path) is False


def test_is_enabled_true_when_on_and_auto(tmp_path):
    from runtime import agent_context
    save("f", ImprovementConfig(enabled=True, transport="auto"), root=tmp_path)
    agent_context.set_active("f")
    assert is_enabled(root=tmp_path) is True


# ---------------------------------------------------------------------------
# Integration with harness brain transport (P3.2 hook)
# ---------------------------------------------------------------------------


def test_brain_transport_select_respects_disabled(tmp_path, monkeypatch):
    """When the active agent disables improvement, select_transport
    returns 'none' so the wakeup loop bails out cleanly."""
    from runtime import agent_context
    # Force project root path resolution to tmp
    import runtime.agents.improvement as imp_mod
    real_path = imp_mod._path
    monkeypatch.setattr(imp_mod, "_path", lambda agent_id, *, root=None: real_path(agent_id, root=tmp_path))

    save("g", ImprovementConfig(enabled=False), root=tmp_path)
    agent_context.set_active("g")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-global")
    from harness.brain.transport import select_transport
    assert select_transport() == "none"


def test_brain_transport_select_respects_explicit(tmp_path, monkeypatch):
    from runtime import agent_context
    import runtime.agents.improvement as imp_mod
    real_path = imp_mod._path
    monkeypatch.setattr(imp_mod, "_path", lambda agent_id, *, root=None: real_path(agent_id, root=tmp_path))

    save("h", ImprovementConfig(enabled=True, transport="anthropic_api"), root=tmp_path)
    agent_context.set_active("h")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from harness.brain.transport import select_transport
    # Agent's explicit transport wins even though API key is missing.
    assert select_transport() == "anthropic_api"
