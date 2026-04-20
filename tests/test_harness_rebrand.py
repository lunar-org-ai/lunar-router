"""Regression tests for the harness / operator loop after the rebrand.

The harness module (agents, memory store, runner, operator loop) was
rebrand-touched in three non-obvious places:

1. The on-disk memory directory moved from ``lunar_router/harness/memory/``
   to ``opentracy/harness/memory/`` (both the path constant and ``.gitignore``).
2. The ``X-Lunar-Internal`` HTTP header the runner sets on internal LLM calls
   became ``X-OpenTracy-Internal``. The Go engine reads this header to skip
   trace collection; if the client header name drifts from the server's
   expectation, harness runs get recorded as user traffic and pollute the
   distillation corpus.
3. The ``ENGINE_URL`` read was moved onto ``opentracy._env.env`` so the
   ``LUNAR_ENGINE_URL`` fallback works.

Pre-rebrand there were zero tests in this module. These are pure-static
checks (no HTTP, no DB, no GPU) that lock the three invariants in place
so a future refactor can't silently flip them.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Import smoke — every harness submodule loads cleanly.
# ---------------------------------------------------------------------------


_HARNESS_MODULES = [
    "opentracy.harness",
    "opentracy.harness.memory_store",
    "opentracy.harness.runner",
    "opentracy.harness.tools",
    "opentracy.harness.operator",
    "opentracy.harness.operator_tools",
    "opentracy.harness.registry",
    "opentracy.harness.scheduler",
    "opentracy.harness.trace_scanner",
    "opentracy.harness.training_advisor",
]


@pytest.mark.parametrize("name", _HARNESS_MODULES)
def test_harness_submodule_imports_cleanly(name: str) -> None:
    """Load the module. Catches any SyntaxError / missing symbol that a
    global sed pass would have introduced (the exact class of bug that
    broke ``datasets/router.py`` at commit 9512b5e)."""
    importlib.import_module(name)


# ---------------------------------------------------------------------------
# 2. Memory directory — path points at the renamed package.
# ---------------------------------------------------------------------------


def test_default_memory_dir_resolves_under_opentracy() -> None:
    """``DEFAULT_MEMORY_DIR`` must sit inside ``opentracy/harness/memory``.

    The directory path is derived from ``Path(__file__).parent / "memory"``,
    so it naturally follows the package rename — but a copy-paste into some
    agent prompt, or a hardcoded string in a config, could drift. Guard it.
    """
    from opentracy.harness.memory_store import DEFAULT_MEMORY_DIR

    parts = DEFAULT_MEMORY_DIR.parts
    assert parts[-3:] == ("opentracy", "harness", "memory"), (
        f"DEFAULT_MEMORY_DIR = {DEFAULT_MEMORY_DIR!r}, expected path to end in "
        "opentracy/harness/memory — rebrand would have moved this from the "
        "old lunar_router/ tree."
    )
    assert "lunar_router" not in str(DEFAULT_MEMORY_DIR), (
        "DEFAULT_MEMORY_DIR still references lunar_router — the rebrand missed this"
    )


# ---------------------------------------------------------------------------
# 3. MemoryStore CRUD roundtrip — store is the on-disk contract for agents.
# ---------------------------------------------------------------------------


def _make_entry(memory_store_module, *, agent: str = "planner", category: str = "run_result"):
    from datetime import datetime, timezone

    return memory_store_module.MemoryEntry(
        id="entry-1",
        agent=agent,
        category=category,
        created_at=datetime.now(timezone.utc).isoformat(),
        body="## Test\nhello world",
        model="openai/gpt-4o-mini",
        duration_ms=123.4,
        tokens_in=10,
        tokens_out=20,
        tool_calls=0,
        retried=False,
        tags=["test"],
        context_hash="abc123",
        evaluation={"success": True, "confidence": 0.9},
    )


def test_memory_store_save_and_get_roundtrip(tmp_path: Path) -> None:
    """Save an entry, read it back, fields survive the YAML frontmatter
    serialization + reparse."""
    from opentracy.harness import memory_store as ms

    store = ms.MemoryStore(memory_dir=tmp_path)
    entry = _make_entry(ms)

    written_path = store.save(entry)
    assert written_path.exists(), "save() did not create the .md file"
    assert written_path.parent == tmp_path, "file was written outside the memory dir"

    # Fresh store reloads from disk — proves the markdown/YAML roundtrip.
    reloaded_store = ms.MemoryStore(memory_dir=tmp_path)
    reloaded = reloaded_store.get("entry-1")
    assert reloaded is not None, "entry lost across store instances"
    assert reloaded.agent == entry.agent
    assert reloaded.category == entry.category
    assert reloaded.tokens_in == entry.tokens_in
    assert reloaded.tokens_out == entry.tokens_out
    assert "hello world" in reloaded.body
    assert reloaded.evaluation.get("confidence") == 0.9


def test_memory_store_delete_removes_disk_and_cache(tmp_path: Path) -> None:
    from opentracy.harness import memory_store as ms

    store = ms.MemoryStore(memory_dir=tmp_path)
    store.save(_make_entry(ms))
    assert store.get("entry-1") is not None

    ok = store.delete("entry-1")
    assert ok is True, "delete() returned False for existing entry"
    assert store.get("entry-1") is None
    assert list(tmp_path.glob("*.md")) == [], "delete() left the .md file on disk"


def test_memory_store_query_filters_by_agent(tmp_path: Path) -> None:
    from opentracy.harness import memory_store as ms

    store = ms.MemoryStore(memory_dir=tmp_path)
    e1 = _make_entry(ms, agent="planner")
    e1.id = "e1"
    e2 = _make_entry(ms, agent="operator")
    e2.id = "e2"
    store.save(e1)
    store.save(e2)

    planners = store.query(agent="planner")
    assert [r.id for r in planners] == ["e1"]


# ---------------------------------------------------------------------------
# 4. HTTP header — X-OpenTracy-Internal is what the runner/engine agreed on.
# ---------------------------------------------------------------------------


def test_runner_sets_opentracy_internal_header_not_lunar() -> None:
    """The runner must send ``X-OpenTracy-Internal: true`` on every internal
    LLM call. The Go engine's handlers read this exact header to skip trace
    recording; if it drifts (e.g. a merge undoing the rename), harness runs
    start showing up in user-facing trace dashboards."""
    from opentracy.harness import runner

    src = inspect.getsource(runner)
    assert '"X-OpenTracy-Internal": "true"' in src, (
        "runner.py no longer sets X-OpenTracy-Internal — Go engine will "
        "record harness calls as user traffic"
    )
    assert "X-Lunar-Internal" not in src, (
        "runner.py still references the pre-rebrand X-Lunar-Internal header"
    )


def test_evals_common_invoker_also_uses_opentracy_header() -> None:
    """The eval model invoker sends the same header as the runner — both
    must stay aligned with the Go server's handler filter (handlers.go:797)."""
    from opentracy.evals_common import model_invoker

    src = inspect.getsource(model_invoker)
    assert '"X-OpenTracy-Internal": "true"' in src
    assert "X-Lunar-Internal" not in src


# ---------------------------------------------------------------------------
# 5. Env-var reads go through the fallback helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    ["opentracy.harness.runner", "opentracy.harness.tools", "opentracy.harness.operator_tools"],
)
def test_harness_modules_use_env_helper_for_engine_url(module_name: str) -> None:
    """Each module must read ``ENGINE_URL`` through ``opentracy._env.env()``
    so the ``LUNAR_ENGINE_URL`` fallback stays in effect. If someone sneaks
    in a raw ``os.environ.get("LUNAR_ENGINE_URL", …)``, the deprecation
    warning vanishes and users never learn to migrate."""
    mod = importlib.import_module(module_name)
    src = inspect.getsource(mod)
    assert 'env("ENGINE_URL"' in src, (
        f"{module_name} must read ENGINE_URL via opentracy._env.env()"
    )
    assert 'os.environ.get("LUNAR_ENGINE_URL"' not in src, (
        f"{module_name} reads LUNAR_ENGINE_URL directly — bypass the helper "
        "and we lose the one-shot deprecation warning"
    )
    assert 'os.getenv("LUNAR_ENGINE_URL"' not in src, (
        f"{module_name} reads LUNAR_ENGINE_URL via os.getenv directly"
    )
