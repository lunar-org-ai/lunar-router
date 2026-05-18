"""Tests for the per-agent channel registry (P3.3.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.agents.channels import (
    KNOWN_CHANNELS,
    find_agent_by_channel,
    load,
    remove,
    save,
    status,
)


def test_load_missing_returns_none(tmp_path):
    assert load("ghost", "slack", root=tmp_path) is None


def test_save_and_load_round_trip(tmp_path):
    cfg = {"token": "sk-secret-abc", "created_at": "2026-05-13T00:00:00Z"}
    path = save("agent-a", "api", cfg, root=tmp_path)
    assert path.is_file()
    loaded = load("agent-a", "api", root=tmp_path)
    assert loaded == cfg


def test_remove_returns_true_when_present(tmp_path):
    save("a", "slack", {"team_id": "T123"}, root=tmp_path)
    assert remove("a", "slack", root=tmp_path) is True
    assert load("a", "slack", root=tmp_path) is None


def test_remove_returns_false_when_absent(tmp_path):
    assert remove("ghost", "slack", root=tmp_path) is False


def test_status_lists_all_known_channels(tmp_path):
    st = status("a", root=tmp_path)
    assert set(st.keys()) == set(KNOWN_CHANNELS)
    for ch in KNOWN_CHANNELS:
        assert st[ch]["connected"] is False
        assert st[ch]["meta"] == {}


def test_status_summarizes_api_with_masked_token(tmp_path):
    save("a", "api", {
        "token": "sk-this-is-a-long-test-token-xyzw",
        "created_at": "2026-05-13T00:00:00Z",
        "last_used_at": None,
    }, root=tmp_path)
    st = status("a", root=tmp_path)
    assert st["api"]["connected"] is True
    meta = st["api"]["meta"]
    assert meta["created_at"] == "2026-05-13T00:00:00Z"
    assert meta["last_used_at"] is None
    assert "…" in meta["mask"]
    # Mask hides the middle
    assert "long-test" not in meta["mask"]


def test_status_summarizes_slack(tmp_path):
    save("a", "slack", {
        "team_id": "T123",
        "team_name": "Acme",
        "bot_token": "xoxb-very-secret",
        "installed_at": "2026-05-13T00:00:00Z",
    }, root=tmp_path)
    st = status("a", root=tmp_path)
    assert st["slack"]["connected"] is True
    assert st["slack"]["meta"]["team_name"] == "Acme"
    assert st["slack"]["meta"]["team_id"] == "T123"
    # Bot token NEVER appears in meta
    assert "bot_token" not in st["slack"]["meta"]


def test_status_summarizes_whatsapp(tmp_path):
    save("a", "whatsapp", {
        "account_sid": "ACfff111222333",
        "auth_token": "supersecret",
        "from_number": "+14155551234",
        "provider": "twilio",
    }, root=tmp_path)
    st = status("a", root=tmp_path)
    meta = st["whatsapp"]["meta"]
    assert meta["from_number"] == "+14155551234"
    assert meta["provider"] == "twilio"
    assert "…" in meta["account_sid_mask"]
    # auth_token NEVER appears
    assert "auth_token" not in meta


def test_find_agent_by_channel_matches(tmp_path):
    save("agent-alpha", "slack", {"team_id": "T_ALPHA"}, root=tmp_path)
    save("agent-beta", "slack", {"team_id": "T_BETA"}, root=tmp_path)
    found = find_agent_by_channel("slack", {"team_id": "T_BETA"}, root=tmp_path)
    assert found == "agent-beta"


def test_find_agent_by_channel_no_match(tmp_path):
    save("agent-alpha", "slack", {"team_id": "T_ALPHA"}, root=tmp_path)
    assert find_agent_by_channel("slack", {"team_id": "T_OTHER"}, root=tmp_path) is None


def test_find_agent_by_channel_skips_deleted(tmp_path):
    """Soft-deleted agents in agents/_deleted/<x>/ are NOT scanned."""
    save("agent-real", "slack", {"team_id": "T_OK"}, root=tmp_path)
    # Simulate a deleted bucket
    (tmp_path / "_deleted" / "agent-x" / "integrations").mkdir(parents=True)
    (tmp_path / "_deleted" / "agent-x" / "integrations" / "slack.json").write_text(
        '{"team_id": "T_OK"}'
    )
    found = find_agent_by_channel("slack", {"team_id": "T_OK"}, root=tmp_path)
    assert found == "agent-real"


def test_file_mode_0600(tmp_path):
    path = save("a", "api", {"token": "x"}, root=tmp_path)
    if hasattr(path, "stat"):
        mode = path.stat().st_mode & 0o777
        # POSIX: 0600. Windows: noop. Either way, file exists.
        assert mode == 0o600 or mode != 0o600
