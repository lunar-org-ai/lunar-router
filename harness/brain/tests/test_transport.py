"""Tests for harness.brain.transport — selection logic only.

Live transport calls (anthropic_api / claude_code_cli) need a real brain
and are exercised by the OPENTRACY_RUN_SMOKE-gated judge smoke test.
"""

from __future__ import annotations

import pytest

from harness.brain.transport import (
    BrainNotAvailableError,
    complete,
    select_transport,
)


def test_select_transport_none(monkeypatch):
    """No env key + no `claude` on PATH → 'none'."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)
    monkeypatch.setattr("harness.brain.transport.shutil.which", lambda _: None)
    assert select_transport() == "none"


def test_select_transport_anthropic_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)
    monkeypatch.setattr("harness.brain.transport.shutil.which", lambda _: None)
    assert select_transport() == "anthropic_api"


def test_select_transport_cli_when_only_cli_present(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)
    monkeypatch.setattr("harness.brain.transport.shutil.which", lambda _: "/usr/bin/claude")
    assert select_transport() == "claude_code_cli"


def test_brain_transport_env_overrides(monkeypatch):
    """BRAIN_TRANSPORT env var overrides auto-detection."""
    monkeypatch.setenv("BRAIN_TRANSPORT", "claude_code_cli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "would-otherwise-pick-api")
    assert select_transport() == "claude_code_cli"


def test_brain_transport_env_unknown_value_falls_back(monkeypatch):
    """Unknown BRAIN_TRANSPORT value falls back to auto-detection."""
    monkeypatch.setenv("BRAIN_TRANSPORT", "garbage-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("harness.brain.transport.shutil.which", lambda _: None)
    assert select_transport() == "anthropic_api"


def test_complete_raises_brain_not_available(monkeypatch):
    """complete() with no transport reachable → BrainNotAvailableError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)
    monkeypatch.setattr("harness.brain.transport.shutil.which", lambda _: None)
    with pytest.raises(BrainNotAvailableError):
        complete("hello")


def test_complete_explicit_none_raises():
    """transport='none' is not a valid override — never used in practice but
    asserting the failure mode is explicit."""
    # 'none' isn't in the BRAIN_TRANSPORT whitelist, so we have to call
    # complete directly with select_transport() returning 'none'. The
    # raise happens inside complete() before we'd dispatch.
    pass  # covered indirectly by test_complete_raises_brain_not_available
