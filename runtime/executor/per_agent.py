"""Per-(tenant, agent) executor cache.

The runtime kept a single process-global executor compiled once at
startup from ``agent/agent.yaml`` (P2). Multi-tenant + per-agent
pipelines (P16.1 + P16.4) made that wrong: a chat request for agent
A on tenant X needs to run A's pipeline, not whatever last booted.

This module resolves the right executor per request and caches the
compiled pipeline so we don't pay the YAML-load + stage-compile cost
on every chat hit.

Resolution
----------

  1. Cache hit on ``(tenant_id, agent_id)`` → return immediately.
  2. Cache miss → load ``tenants/<tenant_id>/agents/<agent_id>/agent.yaml``,
     compile, cache, return.
  3. No per-agent config on disk → fall back to the global
     ``_state['executor']`` so endpoints that predate per-agent
     pipelines (the harness CLI, OSS local mode) keep working.

Invalidation happens at three points:

  - :func:`runtime.server._reload_live_pipeline` after activate(),
    so the just-promoted pipeline is recompiled on its next call.
  - The pipeline-reload endpoint (POST /agents/<id>/pipeline/reload)
    for cases where the operator edits YAML directly in the bucket.
  - At test boundaries via :func:`clear_cache`.

The cache lives in-process. Cloud Run autoscale starts a fresh
instance with a cold cache; each instance re-warms on its first hit.
Memory is bounded by the agent count per tenant — bounded by humans
in the loop, not request volume.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("runtime.executor.per_agent")


# (tenant_id, agent_id) → PipelineExecutor. Single dict serialized by
# a coarse lock — cache miss is rare (per-agent, per-process) so a
# fine-grained per-key lock isn't worth the complexity.
_executors: dict[tuple[str, str], Any] = {}
_lock = threading.Lock()


def get_executor_for_agent(
    tenant_id: Optional[str],
    agent_id: Optional[str],
    *,
    fallback_executor: Any = None,
    agents_root: Optional[Path] = None,
) -> Any:
    """Return the executor that should run ``agent_id``'s pipeline.

    ``tenant_id`` and ``agent_id`` come from the active ContextVars
    (set by the tenant middleware + the API-chat authorizer). When
    either is None or the per-agent agent.yaml is missing, we return
    ``fallback_executor`` (the caller passes ``_state['executor']``
    from the lifespan-compiled global pipeline).

    Pass ``agents_root`` only from tests — production callers let the
    resolver derive it from the active tenant.
    """
    if not tenant_id or not agent_id:
        return fallback_executor

    key = (tenant_id, agent_id)
    with _lock:
        cached = _executors.get(key)
    if cached is not None:
        return cached

    cfg_path = _agent_yaml_path(tenant_id, agent_id, root=agents_root)
    if cfg_path is None or not cfg_path.is_file():
        # Per-agent config doesn't exist on disk; fall back to the
        # global executor so the caller still gets a working pipeline.
        # Common during onboarding (agent created but not yet seeded)
        # and in OSS mode.
        return fallback_executor

    try:
        executor = _compile(cfg_path)
    except Exception as exc:
        logger.warning(
            "per-agent compile failed for (%s, %s): %s — using fallback",
            tenant_id, agent_id, exc, exc_info=True,
        )
        return fallback_executor

    with _lock:
        # Double-check the cache in case a concurrent request raced us
        # — keep the first compiled executor so stage-level state
        # (none today, but room for it) doesn't fork.
        cached = _executors.get(key)
        if cached is not None:
            return cached
        _executors[key] = executor

    logger.info(
        "per-agent executor compiled: tenant=%s agent=%s stages=%d",
        tenant_id, agent_id, len(executor.pipeline.stages),
    )
    return executor


def invalidate(tenant_id: Optional[str], agent_id: Optional[str]) -> bool:
    """Drop the cached executor for ``(tenant_id, agent_id)``.

    Returns True if an entry was removed. Call after editing the
    agent's pipeline YAML so the next request recompiles from disk.
    """
    if not tenant_id or not agent_id:
        return False
    key = (tenant_id, agent_id)
    with _lock:
        return _executors.pop(key, None) is not None


def clear_cache() -> int:
    """Wipe every cached executor. Returns the number dropped.

    Used by tests and by the pipeline-wide reload endpoint (when an
    operator edits global config that affects every agent — e.g.
    techniques code reload).
    """
    with _lock:
        n = len(_executors)
        _executors.clear()
    return n


def cache_keys() -> list[tuple[str, str]]:
    """Snapshot the cached (tenant, agent) pairs. For debug/metrics."""
    with _lock:
        return list(_executors.keys())


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _agent_yaml_path(
    tenant_id: str,
    agent_id: str,
    *,
    root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve ``tenants/<t>/agents/<a>/agent.yaml`` under the right root."""
    if root is not None:
        return Path(root) / agent_id / "agent.yaml"
    try:
        from runtime.tenants.registry import get_tenant_dir
    except Exception:
        return None
    try:
        tenant_dir = get_tenant_dir(tenant_id)
    except Exception:
        return None
    if tenant_dir is None:
        return None
    return tenant_dir / "agents" / agent_id / "agent.yaml"


def _compile(agent_yaml: Path) -> Any:
    """Load + compile + wrap in an executor. Isolated for monkeypatch in tests."""
    from runtime.compiler.builder import compile_agent
    from runtime.compiler.loader import load_agent
    from runtime.executor.pipeline import PipelineExecutor

    cfg = load_agent(agent_yaml)
    pipeline = compile_agent(cfg)
    return PipelineExecutor(pipeline)
