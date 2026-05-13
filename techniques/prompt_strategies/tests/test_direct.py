"""Tests for the P1.9 real-LLM generate stage."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass

import pytest

from runtime.protocols import Context, Document, RoutingDecision
from techniques.prompt_strategies.impl import PromptStrategiesTechnique


def _make_ctx(request="hi", docs=None, model="claude-haiku-4-5"):
    return Context(
        request=request,
        documents=list(docs or []),
        routing=RoutingDecision(model=model) if model else None,
    )


def _compile():
    return PromptStrategiesTechnique().compile(
        "direct",
        knobs={"prompt": "../prompts/system.md", "max_tokens": 64, "temperature": 0.0},
    )


def test_offline_when_no_api_key(monkeypatch):
    """Missing ANTHROPIC_API_KEY → offline marker, no SDK import attempted."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stage = _compile()
    ctx = stage.execute(_make_ctx(request="What's 2+2?"))
    assert ctx.response is not None
    assert ctx.response.startswith("[offline]")
    assert "claude-haiku-4-5" in ctx.response
    assert "llm_usage" not in ctx.state


def test_real_call_populates_response_and_usage(monkeypatch):
    """With ANTHROPIC_API_KEY set, the stage calls Anthropic SDK and
    stashes usage. We inject a fake ``anthropic`` module so the test
    runs offline + deterministically."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    captured = {}

    @dataclass
    class _Block:
        text: str

    @dataclass
    class _Usage:
        input_tokens: int
        output_tokens: int

    class _Response:
        def __init__(self):
            self.content = [_Block(text="The answer is 4.")]
            self.usage = _Usage(input_tokens=12, output_tokens=5)

    class _Messages:
        def create(self, **kw):
            captured.update(kw)
            return _Response()

    class _FakeAnthropic:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.messages = _Messages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    stage = _compile()
    ctx = stage.execute(
        _make_ctx(
            request="What is 2+2?",
            docs=[Document(content="Math: 2+2=4.")],
            model="claude-sonnet-4-6",
        )
    )

    assert ctx.response == "The answer is 4."
    # P3.1 — usage now also carries the provider (anthropic | openai)
    # because the generate stage routes across SDKs.
    assert ctx.state["llm_usage"] == {
        "input_tokens": 12,
        "output_tokens": 5,
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
    }
    # Model came from the routing decision, not the default
    assert captured["model"] == "claude-sonnet-4-6"
    # Documents were rendered into the user message
    assert "Math: 2+2=4." in captured["messages"][0]["content"]
    assert "What is 2+2?" in captured["messages"][0]["content"]
    # Knobs propagated
    assert captured["max_tokens"] == 64
    assert captured["temperature"] == 0.0


def test_sdk_failure_returns_error_marker(monkeypatch):
    """SDK raises → stage swallows + emits error marker. The pipeline must
    not crash because of a transient API failure."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    class _FakeAnthropic:
        def __init__(self, api_key=None):
            self.messages = self
        def create(self, **kw):
            raise RuntimeError("rate limit")

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    stage = _compile()
    ctx = stage.execute(_make_ctx())
    assert ctx.response is not None
    assert ctx.response.startswith("[error]")
    assert "rate limit" in ctx.response


def test_routing_default_when_missing(monkeypatch):
    """No routing decision → falls back to claude-haiku-4-5."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stage = _compile()
    ctx = stage.execute(_make_ctx(model=None))
    assert "claude-haiku-4-5" in ctx.response


def test_executor_uses_real_usage(monkeypatch):
    """End-to-end at the executor level: when generate stashes usage,
    ExecutionRecord.tokens_in/out match the real numbers, not the
    char-based estimate."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    @dataclass
    class _Block:
        text: str

    @dataclass
    class _Usage:
        input_tokens: int
        output_tokens: int

    class _R:
        content = [_Block(text="hello world")]
        usage = _Usage(input_tokens=123, output_tokens=7)

    class _M:
        def create(self, **kw):  # noqa: ARG002
            return _R()

    class _Fake:
        def __init__(self, api_key=None):  # noqa: ARG002
            self.messages = _M()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _Fake
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    from runtime.compiler.builder import compile_agent
    from runtime.compiler.loader import load_agent
    from runtime.executor.pipeline import PipelineExecutor

    cfg = load_agent("agent/agent.yaml")
    pipe = compile_agent(cfg)
    exe = PipelineExecutor(pipe)
    _, record = exe.run("test request")

    # Exact usage numbers carry through (not char-estimated from
    # "test request" + "hello world" which would be 2 + 2).
    assert record.tokens_in == 123
    assert record.tokens_out == 7
    assert record.cost_usd > 0.0
    assert record.response == "hello world"


def test_offline_pipeline_still_populates_cost(monkeypatch):
    """No API key → executor falls back to char-based estimate (the
    pre-P1.9 behavior). Cost must still be non-zero so the metrics
    tiles render."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from runtime.compiler.builder import compile_agent
    from runtime.compiler.loader import load_agent
    from runtime.executor.pipeline import PipelineExecutor

    cfg = load_agent("agent/agent.yaml")
    pipe = compile_agent(cfg)
    exe = PipelineExecutor(pipe)
    _, record = exe.run("offline test request that is long enough to estimate")
    assert record.tokens_in >= 1
    assert record.tokens_out >= 1
    assert record.cost_usd > 0.0
    assert record.response.startswith("[offline]")
