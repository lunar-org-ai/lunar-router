"""Test that compile_agent warms the embedder pool when uniroute is used."""

from __future__ import annotations

import pytest

from runtime.compiler.builder import compile_agent
from runtime.types import (
    AgentConfig,
    BudgetConfig,
    CrossCuttingConfig,
    StageConfig,
)


def _make_cfg(routing_variant: str) -> AgentConfig:
    return AgentConfig(
        version="v0.0.1",
        description="test",
        pipeline=[
            StageConfig(
                stage="route",
                technique="routing",
                variant=routing_variant,
                knobs={"small": "claude-haiku-4-5"},
            ),
        ],
        cross_cutting=CrossCuttingConfig(),
        budget=BudgetConfig(),
    )


def test_warm_called_when_variant_is_uniroute(monkeypatch):
    warmed = []

    class FakePool:
        def warm(self):
            warmed.append(True)

    monkeypatch.setattr("runtime.embedder_pool.get_pool", lambda: FakePool())

    compile_agent(_make_cfg("uniroute"))
    assert warmed == [True]


def test_warm_NOT_called_when_variant_is_small_first(monkeypatch):
    warmed = []

    class FakePool:
        def warm(self):
            warmed.append(True)

    monkeypatch.setattr("runtime.embedder_pool.get_pool", lambda: FakePool())

    compile_agent(_make_cfg("small_first"))
    assert warmed == []


def test_warmup_failure_does_not_break_compile(monkeypatch):
    """When embedder extras aren't installed, compile still succeeds."""

    def boom():
        raise ImportError("torch not installed")

    class FakePool:
        def warm(self):
            boom()

    monkeypatch.setattr("runtime.embedder_pool.get_pool", lambda: FakePool())

    # Should not raise.
    pipe = compile_agent(_make_cfg("uniroute"))
    assert pipe is not None
