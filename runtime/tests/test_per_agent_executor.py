"""Tests for runtime.executor.per_agent — the per-(tenant, agent) cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from runtime.executor import per_agent as pae


@pytest.fixture(autouse=True)
def _clear():
    pae.clear_cache()
    yield
    pae.clear_cache()


def _seed_agent(root: Path, agent_id: str) -> Path:
    """Create a minimal valid agent.yaml under ``root/<agent_id>/``."""
    agent_dir = root / agent_id
    (agent_dir / "pipeline").mkdir(parents=True)
    (agent_dir / "agent.yaml").write_text(
        "agent:\n"
        "  version: v0.0.1\n"
        "  pipeline:\n"
        "  - $ref: pipeline/generate.yaml\n",
        encoding="utf-8",
    )
    (agent_dir / "pipeline" / "generate.yaml").write_text(
        "stage: generate\n"
        "technique: prompt_strategies\n"
        "variant: claude_code\n"
        "knobs:\n"
        "  template: opentracy-sandbox\n",
        encoding="utf-8",
    )
    return agent_dir / "agent.yaml"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_returns_fallback_when_tenant_missing():
    sentinel = object()
    assert pae.get_executor_for_agent(None, "agent-a", fallback_executor=sentinel) is sentinel


def test_returns_fallback_when_agent_missing():
    sentinel = object()
    assert pae.get_executor_for_agent("t1", None, fallback_executor=sentinel) is sentinel


def test_returns_fallback_when_agent_yaml_absent(tmp_path):
    """Per-agent config not on disk → fall through to global executor."""
    sentinel = object()
    out = pae.get_executor_for_agent(
        "t1", "no-such-agent",
        fallback_executor=sentinel,
        agents_root=tmp_path,
    )
    assert out is sentinel


def test_compiles_from_disk_on_cold_call(tmp_path):
    _seed_agent(tmp_path, "agent-a")
    out = pae.get_executor_for_agent(
        "t1", "agent-a",
        fallback_executor=None,
        agents_root=tmp_path,
    )
    assert out is not None
    assert out.pipeline.config.pipeline[0].variant == "claude_code"


def test_cache_hit_returns_same_instance(tmp_path):
    _seed_agent(tmp_path, "agent-a")
    a = pae.get_executor_for_agent(
        "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
    )
    b = pae.get_executor_for_agent(
        "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
    )
    assert a is b


def test_compile_failure_falls_back_without_caching(tmp_path, monkeypatch):
    _seed_agent(tmp_path, "agent-broken")

    def _boom(_path):
        raise RuntimeError("yaml is haunted")
    monkeypatch.setattr(pae, "_compile", _boom)

    sentinel = object()
    out = pae.get_executor_for_agent(
        "t1", "agent-broken", fallback_executor=sentinel, agents_root=tmp_path,
    )
    assert out is sentinel
    assert pae.cache_keys() == []  # didn't poison the cache


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_same_agent_id_different_tenants_isolated(tmp_path, monkeypatch):
    # Two roots, same agent slug under each.
    root_a = tmp_path / "tenantA" / "agents"
    root_b = tmp_path / "tenantB" / "agents"
    _seed_agent(root_a, "shared-id")
    _seed_agent(root_b, "shared-id")

    exec_a = pae.get_executor_for_agent(
        "tenantA", "shared-id", fallback_executor=None, agents_root=root_a,
    )
    exec_b = pae.get_executor_for_agent(
        "tenantB", "shared-id", fallback_executor=None, agents_root=root_b,
    )
    assert exec_a is not None
    assert exec_b is not None
    assert exec_a is not exec_b
    # Cache holds both keys distinctly.
    assert set(pae.cache_keys()) == {("tenantA", "shared-id"), ("tenantB", "shared-id")}


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_invalidate_drops_cached_executor(tmp_path):
    _seed_agent(tmp_path, "agent-a")
    pae.get_executor_for_agent(
        "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
    )
    assert pae.cache_keys() == [("t1", "agent-a")]

    removed = pae.invalidate("t1", "agent-a")
    assert removed is True
    assert pae.cache_keys() == []


def test_invalidate_returns_false_when_no_entry():
    assert pae.invalidate("t1", "absent") is False


def test_invalidate_after_edit_picks_up_new_config(tmp_path):
    """Edit-then-invalidate-then-resolve sees the new YAML."""
    _seed_agent(tmp_path, "agent-a")
    first = pae.get_executor_for_agent(
        "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
    )
    assert first.pipeline.config.pipeline[0].variant == "claude_code"

    # Edit underlying YAML.
    (tmp_path / "agent-a" / "pipeline" / "generate.yaml").write_text(
        "stage: generate\n"
        "technique: prompt_strategies\n"
        "variant: direct\n"
        "knobs:\n"
        "  prompt: ../prompts/system.md\n",
        encoding="utf-8",
    )
    # Without invalidation, the cache still returns the old compile.
    cached = pae.get_executor_for_agent(
        "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
    )
    assert cached.pipeline.config.pipeline[0].variant == "claude_code"

    # After invalidation, the next call recompiles + sees the edit.
    pae.invalidate("t1", "agent-a")
    fresh = pae.get_executor_for_agent(
        "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
    )
    assert fresh.pipeline.config.pipeline[0].variant == "direct"


def test_clear_cache_drops_everything(tmp_path):
    _seed_agent(tmp_path, "agent-a")
    _seed_agent(tmp_path, "agent-b")
    pae.get_executor_for_agent("t1", "agent-a", fallback_executor=None, agents_root=tmp_path)
    pae.get_executor_for_agent("t1", "agent-b", fallback_executor=None, agents_root=tmp_path)
    assert len(pae.cache_keys()) == 2

    dropped = pae.clear_cache()
    assert dropped == 2
    assert pae.cache_keys() == []


# ---------------------------------------------------------------------------
# Concurrency — double-checked locking keeps single instance per key
# ---------------------------------------------------------------------------


def test_concurrent_misses_yield_single_cached_executor(tmp_path, monkeypatch):
    """The resolver doesn't hold the lock during compile (cheap op,
    not worth blocking on), so racing callers may all do the work.
    The invariant that matters: exactly one of those compiles ends up
    in the cache, and every caller returns that one — no two requests
    ever see different executor instances for the same key."""
    _seed_agent(tmp_path, "agent-a")
    real_compile = pae._compile

    def _slow_compile(path):
        import time
        time.sleep(0.05)
        return real_compile(path)

    monkeypatch.setattr(pae, "_compile", _slow_compile)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(
            lambda _: pae.get_executor_for_agent(
                "t1", "agent-a", fallback_executor=None, agents_root=tmp_path,
            ),
            range(6),
        ))

    first = results[0]
    assert all(r is first for r in results)
    # Cache holds exactly one entry, not 6.
    assert pae.cache_keys() == [("t1", "agent-a")]
