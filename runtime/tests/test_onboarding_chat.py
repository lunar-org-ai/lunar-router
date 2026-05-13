"""Tests for the conversational onboarding turn (P1.12)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# extract_json — tolerant parsing
# ---------------------------------------------------------------------------


def test_extract_json_plain_object():
    from runtime.store.onboarding_chat import extract_json
    out = extract_json('{"reply":"hi","config":{}}')
    assert out == {"reply": "hi", "config": {}}


def test_extract_json_with_fence():
    from runtime.store.onboarding_chat import extract_json
    text = '```json\n{"reply":"hi","config":{}}\n```'
    assert extract_json(text) == {"reply": "hi", "config": {}}


def test_extract_json_with_prose_before():
    from runtime.store.onboarding_chat import extract_json
    text = 'Sure, here is the JSON:\n{"reply":"hi","config":{}}'
    assert extract_json(text) == {"reply": "hi", "config": {}}


def test_extract_json_handles_nested_braces():
    from runtime.store.onboarding_chat import extract_json
    text = '{"reply":"hi","config":{"name":"x","tools":["a"]},"ready":true}'
    out = extract_json(text)
    assert out["config"]["name"] == "x"
    assert out["ready"] is True


def test_extract_json_returns_none_on_garbage():
    from runtime.store.onboarding_chat import extract_json
    assert extract_json("no json here") is None
    assert extract_json("") is None
    assert extract_json("{ unclosed") is None


# ---------------------------------------------------------------------------
# run_turn — scripted fallback path
# ---------------------------------------------------------------------------


def test_run_turn_scripted_fallback_when_transport_none():
    """Explicit transport='none' → returns SCRIPT entries by user-turn count."""
    from runtime.store.onboarding_chat import run_turn

    out1 = run_turn(
        [{"role": "user", "content": "build a support agent"}],
        transport="none",
    )
    assert "refund" in out1["reply"].lower()
    assert out1["config"]["tools"] == ["lookup_order", "search_kb"]
    assert out1["ready"] is False

    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"answer {i}"})
        history.append({"role": "assistant", "content": "thinking"})
    out_last = run_turn(history, transport="none")
    assert out_last["ready"] is True


def test_run_turn_scripted_clamps_beyond_script():
    """More user turns than the script has → keep returning the last entry."""
    from runtime.store.onboarding_chat import run_turn

    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"turn {i}"})
    out = run_turn(history, transport="none")
    assert out["ready"] is True


def test_detect_transport_prefers_cli_over_api(monkeypatch):
    """When both claude CLI and ANTHROPIC_API_KEY are available, CLI wins —
    it has filesystem access + MCP servers, which makes onboarding richer."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)
    from runtime.store.onboarding_chat import detect_transport
    info = detect_transport()
    assert info["transport"] == "claude_code_cli"


def test_detect_transport_falls_through_to_api_then_none(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    from runtime.store.onboarding_chat import detect_transport
    assert detect_transport()["transport"] == "anthropic_api"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert detect_transport()["transport"] == "none"


def test_detect_transport_env_override(monkeypatch):
    """BRAIN_TRANSPORT env var beats auto-detect."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setenv("BRAIN_TRANSPORT", "anthropic_api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    from runtime.store.onboarding_chat import detect_transport
    assert detect_transport()["transport"] == "anthropic_api"


# ---------------------------------------------------------------------------
# run_turn — real-path with mocked Anthropic
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    text: str


class _Resp:
    def __init__(self, text: str):
        self.content = [_Block(text=text)]


class _Messages:
    def __init__(self, text: str):
        self._text = text
        self.captured: dict = {}

    def create(self, **kw):
        self.captured.update(kw)
        return _Resp(self._text)


class _FakeAnthropic:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = _Messages(_FakeAnthropic.next_text)
    next_text = ""


def _inject_anthropic(monkeypatch, text: str):
    _FakeAnthropic.next_text = text
    fake = types.ModuleType("anthropic")
    fake.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def test_run_turn_parses_valid_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    _inject_anthropic(
        monkeypatch,
        '{"reply":"Cool!","config":{"name":"x-agent","model":"claude-haiku-4-5","prompt":"You are X.","tools":["t1"],"channels":["web"]},"justAdded":{"tool":"t1"},"ready":false}',
    )
    from runtime.store.onboarding_chat import run_turn
    out = run_turn([{"role": "user", "content": "what's it for?"}], transport="anthropic_api")
    assert out["reply"] == "Cool!"
    assert out["config"]["name"] == "x-agent"
    assert out["config"]["model"] == "claude-haiku-4-5"
    assert out["config"]["tools"] == ["t1"]
    assert out["justAdded"] == {"tool": "t1"}
    assert out["ready"] is False


def test_run_turn_falls_back_when_json_malformed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    _inject_anthropic(monkeypatch, "I'm not returning JSON, I'm just chatting!")
    from runtime.store.onboarding_chat import run_turn
    out = run_turn([{"role": "user", "content": "hi"}], transport="anthropic_api")
    # Falls back to SCRIPT[0]
    assert "refund" in out["reply"].lower()


def test_run_turn_falls_back_when_anthropic_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    class _Bad:
        def __init__(self, api_key=None):
            self.messages = self
        def create(self, **kw):
            raise RuntimeError("rate limit")

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Bad
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from runtime.store.onboarding_chat import run_turn
    out = run_turn([{"role": "user", "content": "hi"}], transport="anthropic_api")
    # Did NOT crash; got scripted instead
    assert out["reply"]
    assert "config" in out


def test_run_turn_strips_invalid_message_roles(monkeypatch):
    """We send only user/assistant messages to Anthropic — never 'system'
    (that's the top-level system arg) and never empty content."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    captured = {}

    class _Cap:
        def __init__(self, api_key=None):
            self.messages = self
        def create(self, **kw):
            captured.update(kw)
            return _Resp('{"reply":"ok","config":{}}')

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Cap
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from runtime.store.onboarding_chat import run_turn
    run_turn(
        [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "real"},
            {"role": "user", "content": ""},  # empty drops
            {"role": "assistant", "content": "I am here"},
        ],
        transport="anthropic_api",
    )
    msgs = captured["messages"]
    assert all(m["role"] in {"user", "assistant"} for m in msgs)
    assert all(m["content"].strip() for m in msgs)
    assert any(m["content"] == "real" for m in msgs)


# ---------------------------------------------------------------------------
# Server endpoint
# ---------------------------------------------------------------------------


def test_post_onboarding_turn_endpoint(monkeypatch):
    """POST /onboarding/turn returns the parsed turn shape."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Force the offline path so the test doesn't depend on a real CLI/API.
    import runtime.store.onboarding_chat as oc
    monkeypatch.setattr(oc, "detect_transport", lambda: {"transport": "none", "cwd": "/", "claude_version": None})
    from fastapi.testclient import TestClient
    from runtime.server import app

    with TestClient(app) as c:
        r = c.post(
            "/onboarding/turn",
            json={"messages": [{"role": "user", "content": "I want a support agent"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["reply"]
        assert body["config"]["model"] in {
            "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7",
        }
        assert "tools" in body["config"]
        assert "channels" in body["config"]


# ---------------------------------------------------------------------------
# Claude Code CLI path (P1.13)
# ---------------------------------------------------------------------------


def test_run_turn_via_claude_cli(monkeypatch):
    """When transport=claude_code_cli, run_turn invokes `claude --print`
    as a subprocess with the rendered history in the user prompt and
    the SYSTEM_PROMPT appended via --append-system-prompt."""
    import subprocess as _sp
    from runtime.store.onboarding_chat import SYSTEM_PROMPT, run_turn

    captured: dict = {}
    class _Result:
        returncode = 0
        stdout = '{"reply":"From CLI!","config":{"name":"x","model":"claude-sonnet-4-6","prompt":"You are X.","tools":[],"channels":["web"]}}'
        stderr = ""

    def _fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(_sp, "run", _fake_run)

    out = run_turn(
        [{"role": "user", "content": "Build me something useful."}],
        transport="claude_code_cli",
    )
    assert out["reply"] == "From CLI!"
    assert out["config"]["name"] == "x"

    # Args carry the expected flags
    assert captured["args"][0] == "claude"
    assert "--print" in captured["args"]
    assert "--append-system-prompt" in captured["args"]
    sys_idx = captured["args"].index("--append-system-prompt")
    assert captured["args"][sys_idx + 1] == SYSTEM_PROMPT
    # The final positional arg is the rendered prompt
    assert "Build me something useful" in captured["args"][-1]


def test_run_turn_cli_subprocess_failure_falls_back(monkeypatch):
    """CLI returns non-zero → scripted fallback."""
    import subprocess as _sp
    from runtime.store.onboarding_chat import run_turn

    class _Bad:
        returncode = 2
        stdout = ""
        stderr = "claude crashed"

    monkeypatch.setattr(_sp, "run", lambda *a, **kw: _Bad())
    out = run_turn(
        [{"role": "user", "content": "test"}], transport="claude_code_cli",
    )
    # Got scripted SCRIPT[0]
    assert out["reply"]
    assert out["config"]


def test_run_turn_cli_handles_garbage_output(monkeypatch):
    """CLI outputs prose instead of JSON → scripted fallback."""
    import subprocess as _sp
    from runtime.store.onboarding_chat import run_turn

    class _R:
        returncode = 0
        stdout = "I'm just chatting, no JSON for you"
        stderr = ""

    monkeypatch.setattr(_sp, "run", lambda *a, **kw: _R())
    out = run_turn(
        [{"role": "user", "content": "hi"}], transport="claude_code_cli",
    )
    # Falls back to script
    assert "refund" in out["reply"].lower()


def test_get_onboarding_transport_endpoint(monkeypatch):
    """GET /onboarding/transport surfaces whichever brain is connected."""
    import runtime.store.onboarding_chat as oc
    monkeypatch.setattr(
        oc, "detect_transport",
        lambda: {
            "transport": "claude_code_cli",
            "cwd": "/tmp/test-repo",
            "claude_version": "2.1.140",
        },
    )
    from fastapi.testclient import TestClient
    from runtime.server import app

    with TestClient(app) as c:
        r = c.get("/onboarding/transport")
        assert r.status_code == 200
        body = r.json()
        assert body["transport"] == "claude_code_cli"
        assert body["claude_version"] == "2.1.140"
        assert body["cwd"] == "/tmp/test-repo"
