"""Tests for runtime.sandbox.e2b — the E2B sandbox wrapper.

These tests run without the real E2B SDK installed by either patching
the SDK import or asserting on the availability gates. End-to-end
sandbox tests live in tests/e2e_sandbox/ and require a real API key.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from runtime.sandbox.e2b import (
    DEFAULT_TEMPLATE,
    SandboxRun,
    SandboxUnavailable,
    is_sandbox_available,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("OPENTRACY_E2B_API_KEY", raising=False)
    monkeypatch.delenv("OPENTRACY_E2B_TEMPLATE", raising=False)


def test_is_sandbox_available_false_without_key():
    assert is_sandbox_available() is False


def test_is_sandbox_available_false_without_sdk(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")
    # Force the e2b import to fail so we exercise the SDK-missing path
    # regardless of whether the optional dep is installed locally.
    import sys
    monkeypatch.setitem(sys.modules, "e2b", None)
    assert is_sandbox_available() is False


def test_sandbox_run_requires_anthropic_key():
    with pytest.raises(ValueError, match="anthropic_key"):
        SandboxRun(anthropic_key="")


def test_sandbox_run_raises_without_api_key():
    sb = SandboxRun(anthropic_key="sk-ant-fake")
    # Stub the SDK so we don't need the real e2b package installed.
    with patch("runtime.sandbox.e2b._require_sdk") as require_sdk:
        require_sdk.return_value = MagicMock()
        with pytest.raises(SandboxUnavailable, match="OPENTRACY_E2B_API_KEY"):
            sb.__enter__()


def test_sandbox_run_spawns_and_kills(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()
    fake_sb.commands.run.return_value = MagicMock(exit_code=0, stderr="")

    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            assert sb._sandbox is fake_sb
            # The mkdir bootstrap call should have happened.
            fake_sb.commands.run.assert_any_call("mkdir -p /workspace")

    fake_sandbox_cls.create.assert_called_once()
    create_kwargs = fake_sandbox_cls.create.call_args.kwargs
    assert create_kwargs["template"] == DEFAULT_TEMPLATE
    assert create_kwargs["envs"]["ANTHROPIC_API_KEY"] == "sk-ant-fake"

    fake_sb.kill.assert_called_once()


def test_sandbox_run_kill_runs_even_on_exception(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()
    fake_sb.commands.run.return_value = MagicMock(exit_code=0)

    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with pytest.raises(RuntimeError, match="boom"):
            with SandboxRun(anthropic_key="sk-ant-fake"):
                raise RuntimeError("boom")

    fake_sb.kill.assert_called_once()


def test_upload_workspace_tar_writes_and_extracts(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()
    fake_sb.commands.run.return_value = MagicMock(exit_code=0, stderr="")
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            sb.upload_workspace_tar(b"<tar bytes>")

    fake_sb.files.write.assert_called_once_with("/tmp/opentracy-in.tar.gz", b"<tar bytes>")
    extract_call = [
        c for c in fake_sb.commands.run.call_args_list
        if "tar xz" in str(c.args[0])
    ]
    assert extract_call, "expected a tar extract command"
    # Permission/metadata flags must be present so extraction doesn't
    # try to utime/chmod entries it can't (sandbox runs as non-root).
    cmd = str(extract_call[0].args[0])
    assert "--no-same-owner" in cmd
    assert "--no-same-permissions" in cmd
    assert "tar xzm" in cmd  # -m disables mtime restoration


def test_upload_workspace_tar_raises_on_extract_failure(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()

    def _run(cmd, **_kw):
        # mkdir succeeds, extract fails.
        if "tar xz" in str(cmd):
            return MagicMock(exit_code=1, stderr="tar: bad archive")
        return MagicMock(exit_code=0, stderr="")

    fake_sb.commands.run.side_effect = _run
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            with pytest.raises(RuntimeError, match="bad archive"):
                sb.upload_workspace_tar(b"<tar bytes>")


def test_snapshot_workspace_tar_returns_bytes(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()
    fake_sb.commands.run.return_value = MagicMock(exit_code=0, stderr="")
    fake_sb.files.read.return_value = b"<archive>"
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            data = sb.snapshot_workspace_tar()

    assert data == b"<archive>"
    fake_sb.files.read.assert_called_once_with("/tmp/opentracy-out.tar.gz", format="bytes")


def test_run_claude_streams_events_and_terminates(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()

    def _commands_run(cmd, on_stdout=None, on_stderr=None, **_kw):
        # When cmd is the shlex-joined claude invocation, fire stream events.
        if isinstance(cmd, str) and cmd.startswith("claude "):
            on_stdout("hello ")
            on_stdout("world")
            on_stderr("warn: nothing serious")
            return MagicMock(exit_code=0)
        return MagicMock(exit_code=0, stderr="")

    fake_sb.commands.run.side_effect = _commands_run
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            events = list(sb.run_claude("say hi", system="Be brief."))

    types = [e["type"] for e in events]
    assert "stdout" in types
    assert "done" in types
    stdout_chunks = [e["data"] for e in events if e["type"] == "stdout"]
    assert "".join(stdout_chunks) == "hello world"
    assert events[-1]["type"] == "done"
    assert events[-1]["exit_code"] == 0


def test_run_claude_argv_includes_system_and_model(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()
    captured_cmd: list[str] = []

    def _commands_run(cmd, on_stdout=None, on_stderr=None, **_kw):
        if isinstance(cmd, str) and cmd.startswith("claude "):
            captured_cmd.append(cmd)
            return MagicMock(exit_code=0)
        return MagicMock(exit_code=0, stderr="")

    fake_sb.commands.run.side_effect = _commands_run
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            list(sb.run_claude(
                "do the thing",
                system="You are X",
                model="claude-sonnet-4-6",
            ))

    cmd = captured_cmd[0]
    # Each flag + value must show up in the shlex-joined string. Quoted
    # args ("You are X", "do the thing") get single-quotes from shlex.
    assert "--append-system-prompt 'You are X'" in cmd
    assert "--model claude-sonnet-4-6" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert cmd.endswith("'do the thing'")


def test_run_claude_yields_error_on_exception(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")

    fake_sb = MagicMock()

    def _commands_run(cmd, on_stdout=None, on_stderr=None, **_kw):
        if isinstance(cmd, str) and cmd.startswith("claude "):
            raise RuntimeError("sandbox disconnected")
        return MagicMock(exit_code=0, stderr="")

    fake_sb.commands.run.side_effect = _commands_run
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake") as sb:
            events = list(sb.run_claude("hi"))

    assert events[-1]["type"] == "error"
    assert "disconnected" in events[-1]["detail"]


def test_template_override_via_env(monkeypatch):
    monkeypatch.setenv("OPENTRACY_E2B_API_KEY", "test-key")
    monkeypatch.setenv("OPENTRACY_E2B_TEMPLATE", "custom-template")

    fake_sb = MagicMock()
    fake_sb.commands.run.return_value = MagicMock(exit_code=0)
    fake_sandbox_cls = MagicMock()
    fake_sandbox_cls.create.return_value = fake_sb

    with patch("runtime.sandbox.e2b._require_sdk", return_value=fake_sandbox_cls):
        with SandboxRun(anthropic_key="sk-ant-fake"):
            pass

    assert fake_sandbox_cls.create.call_args.kwargs["template"] == "custom-template"
