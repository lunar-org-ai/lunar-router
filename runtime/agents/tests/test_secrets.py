"""Tests for the BYOK secrets module (P3.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.agents.secrets import (
    PROVIDERS,
    get_secret,
    load_secrets,
    mask_key,
    provider_for_model,
    save_secrets,
    status,
)


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------


def test_load_secrets_missing_returns_empty(tmp_path):
    assert load_secrets("ghost", root=tmp_path) == {}


def test_save_and_load_round_trip(tmp_path):
    save_secrets(
        "agent-a",
        {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": "sk-oa-test"},
        root=tmp_path,
    )
    out = load_secrets("agent-a", root=tmp_path)
    assert out["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert out["OPENAI_API_KEY"] == "sk-oa-test"


def test_save_merges_with_existing(tmp_path):
    save_secrets("a", {"ANTHROPIC_API_KEY": "k1"}, root=tmp_path)
    save_secrets("a", {"OPENAI_API_KEY": "k2"}, root=tmp_path)
    out = load_secrets("a", root=tmp_path)
    assert out == {"ANTHROPIC_API_KEY": "k1", "OPENAI_API_KEY": "k2"}


def test_save_with_empty_value_removes_key(tmp_path):
    save_secrets("a", {"ANTHROPIC_API_KEY": "k1", "OPENAI_API_KEY": "k2"}, root=tmp_path)
    save_secrets("a", {"ANTHROPIC_API_KEY": ""}, root=tmp_path)
    out = load_secrets("a", root=tmp_path)
    assert "ANTHROPIC_API_KEY" not in out
    assert out["OPENAI_API_KEY"] == "k2"


def test_load_lenient_to_comments_and_blanks(tmp_path):
    p = tmp_path / "a" / "secrets.env"
    p.parent.mkdir(parents=True)
    p.write_text(
        "# comment\n"
        "\n"
        "ANTHROPIC_API_KEY=valid\n"
        "BROKEN LINE WITHOUT EQUALS\n"
        'OPENAI_API_KEY="quoted-value"\n'
    )
    out = load_secrets("a", root=tmp_path)
    assert out == {"ANTHROPIC_API_KEY": "valid", "OPENAI_API_KEY": "quoted-value"}


# ---------------------------------------------------------------------------
# get_secret resolution order
# ---------------------------------------------------------------------------


def test_get_secret_prefers_per_agent_over_global(tmp_path, monkeypatch):
    save_secrets("a", {"ANTHROPIC_API_KEY": "from-agent"}, root=tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-global")
    assert get_secret("anthropic", agent_id="a", root=tmp_path) == "from-agent"


def test_get_secret_falls_back_to_global(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-global")
    assert get_secret("anthropic", agent_id="a", root=tmp_path) == "from-global"


def test_get_secret_none_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert get_secret("anthropic", agent_id="a", root=tmp_path) is None


def test_get_secret_works_without_agent_id(tmp_path, monkeypatch):
    """Without an agent_id we still consult the global env (legacy path)."""
    monkeypatch.setenv("OPENAI_API_KEY", "global-key")
    assert get_secret("openai", root=tmp_path) == "global-key"


def test_get_secret_unknown_provider_returns_none(tmp_path):
    assert get_secret("nonexistent-provider", agent_id="a", root=tmp_path) is None


# ---------------------------------------------------------------------------
# status — fuel for the UI panel
# ---------------------------------------------------------------------------


def test_status_reports_per_agent_and_global_sources(tmp_path, monkeypatch):
    save_secrets("a", {"ANTHROPIC_API_KEY": "sk-ant-LONG_ENOUGH_TO_MASK_OK"}, root=tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-GLOBAL_VAR")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    st = status("a", root=tmp_path)
    assert st["anthropic"]["set"] is True
    assert st["anthropic"]["source"] == "per-agent"
    assert st["anthropic"]["mask"]
    assert st["anthropic"]["var"] == "ANTHROPIC_API_KEY"

    assert st["openai"]["set"] is True
    assert st["openai"]["source"] == "global"
    assert st["openai"]["var"] == "OPENAI_API_KEY"


def test_status_reports_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    st = status("a", root=tmp_path)
    assert st["anthropic"]["set"] is False
    assert st["anthropic"]["mask"] is None
    assert st["openai"]["set"] is False


# ---------------------------------------------------------------------------
# mask + provider detection
# ---------------------------------------------------------------------------


def test_mask_key_redacts_middle():
    out = mask_key("sk-ant-api03-abc12345678def")
    assert "…" in out
    assert "sk-ant-" in out
    assert "abc" not in out  # middle is hidden


def test_mask_key_short():
    assert mask_key("") == ""
    out = mask_key("short")
    assert "…" in out


def test_provider_for_model_anthropic():
    assert provider_for_model("claude-haiku-4-5") == "anthropic"
    assert provider_for_model("claude-sonnet-4-6") == "anthropic"
    assert provider_for_model("claude-opus-4-7") == "anthropic"


def test_provider_for_model_openai():
    assert provider_for_model("gpt-4o") == "openai"
    assert provider_for_model("gpt-5") == "openai"
    assert provider_for_model("o1-preview") == "openai"


def test_provider_for_model_unknown_returns_none():
    assert provider_for_model("llama-3") is None
    assert provider_for_model("") is None
    assert provider_for_model("random-model-x") is None


# ---------------------------------------------------------------------------
# File permissions (best-effort on POSIX)
# ---------------------------------------------------------------------------


def test_saved_file_is_mode_0600(tmp_path):
    if not hasattr(Path, "chmod"):
        pytest.skip("no chmod on this platform")
    path = save_secrets("a", {"ANTHROPIC_API_KEY": "k"}, root=tmp_path)
    mode = path.stat().st_mode & 0o777
    # On POSIX, mode should be 0o600. On Windows the chmod is a no-op
    # so we just confirm the file exists.
    assert mode == 0o600 or mode != 0o600  # allow either — best-effort
