"""AHE Algorithm 1 — one iteration (v1).

Order (paraphrasing §3.4):

  0. **Attribute prior round.** If a pending Change Manifest exists
     from the previous iteration, compare its claimed_fixes /
     at_risk_regressions against the CURRENT rollout's pass/fail and
     roll it to history with a verdict. If verdict is ``regressed``,
     restore the files the prior Evolve Agent edited (file-level
     rollback per §3.3).
  1. **Rollout.** Replay each eval task ``k`` times against the
     current harness (v1: k=2 default; v0 was k=1).
  2. **Distill.** Pack the rollout into evidence:
        - raw pass/fail corpus (v0 layer)
        - root-cause clusters via Agent Debugger Lite (v1 layer)
  3. **Edit.** Spawn the Evolve Agent sandbox; it reads the corpus +
     NexAU snapshot + manifest history, edits one or more harness
     files, and writes a fresh pending manifest.
  4. **Snapshot for rollback.** Save the pre-edit content of every
     file the agent claims to have touched, so the NEXT iteration's
     verdict can revert if predictions miss.
  5. **Commit.** Snapshot the workspace back (already done inside
     :func:`evolve.run_evolve`); invalidate the per-agent executor
     cache so the next chat request recompiles with the edits.

v1 limitations (vs the paper, still):
  - Verdict is coarse pass/fail per task — no per-claim grading
  - No Clean step yet (rollouts don't carry base64 / tool dumps)
  - No git tag per iteration; the manifest archive timestamp is the
    only iteration marker on disk
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Optional

from runtime.evolution.distill import cluster_failures, summarize_rollout
from runtime.evolution.evolve import run_evolve
from runtime.evolution.rollout import run_rollout
from runtime.evolution.types import (
    EvolveOutcome,
    IterationResult,
    RolloutResult,
    VerificationResult,
)


logger = logging.getLogger("runtime.evolution.loop")


DEFAULT_K = 2


def run_one_iteration(
    *,
    agent_id: str,
    tasks: list[str],
    tenant_id: Optional[str] = None,
    k: int = DEFAULT_K,
    sandbox_factory: Optional[Any] = None,
) -> IterationResult:
    """Run one AHE iteration against ``agent_id``.

    ``k`` controls replay count per task in the rollout phase (v1
    default 2; pass 1 to reproduce v0 behavior).
    """
    iteration_id = _new_iteration_id()
    started = time.time()
    logger.info(
        "evolve: iteration %s starting for agent=%s tasks=%d k=%d",
        iteration_id, agent_id, len(tasks), k,
    )

    # Resolve dependencies lazily so the module imports cheaply and
    # tests can stub each layer independently.
    from runtime.agents.secrets import get_secret
    from runtime.executor import per_agent as _per_agent
    from runtime.tenant_context import get_active as get_tenant
    from runtime.workspaces import get_workspace

    if tenant_id is None:
        tenant_id = get_tenant(default="")

    anthropic_key = get_secret("anthropic", agent_id=agent_id)
    if not anthropic_key:
        raise RuntimeError(
            "evolve: no Anthropic key for agent — set BYOK first"
        )

    workspace = get_workspace(agent_id)

    # 0a. Attribute prior round (if any pending manifest).
    pending_before = workspace.read_pending_manifest()

    # 1. Rollout (k>=1). Pass agent_id so the rollout pins
    # agent_context — without that, stages that lazily read the
    # active agent (claude_code strategy reads the workspace via
    # ``get_workspace(get_active())``) fall back to ``_default`` and
    # use the wrong harness state, which silently invalidates every
    # downstream signal (evidence, evolve, verdict).
    # ``write_trace`` makes every rollout call persist a trace under
    # ``traces/<agent>/raw/<date>.jsonl`` — same path the chat
    # endpoint uses — so Technical → Traces shows the rollout runs.
    from runtime.executor.tracing import write_trace as _write_trace

    def _persist_trace(record: Any) -> Optional[str]:
        return _write_trace(record, agent_id=agent_id)

    executor = _resolve_executor_for_evolution(agent_id, tenant_id)
    rollout: RolloutResult = run_rollout(
        executor=executor,
        tasks=tasks,
        k=k,
        agent_id=agent_id,
        write_trace=_persist_trace,
    )

    # 0b. Verdict on the prior round, using THIS round's rollout.
    #     If regressed, also apply file-level rollback.
    verification = _verify_previous(
        workspace=workspace,
        pending=pending_before,
        rollout=rollout,
    )

    # 2. Distill — raw summary + LLM-clustered failures.
    evidence = summarize_rollout(rollout)
    evidence = cluster_failures(evidence, anthropic_key=anthropic_key)

    # 3 + 4. Pre-snapshot, edit (sandboxed), persist rollback snapshot.
    pre_edit_files = _snapshot_files(workspace)
    try:
        evolve_outcome: EvolveOutcome = run_evolve(
            workspace=workspace,
            anthropic_key=anthropic_key,
            agent_id=agent_id,
            evidence_summary=_evidence_for_evolve(evidence),
            sandbox_factory=sandbox_factory,
        )
    except Exception as exc:
        logger.warning("evolve: edit phase failed: %s", exc, exc_info=True)
        evolve_outcome = EvolveOutcome(
            files_edited=[],
            pending_manifest=None,
            raw_response=f"[error] {type(exc).__name__}: {exc}",
        )

    _persist_rollback_snapshot(
        workspace=workspace,
        iteration_id=iteration_id,
        pre_edit_files=pre_edit_files,
        pending_manifest=evolve_outcome.pending_manifest,
    )

    # 5. Drop the per-agent executor cache so the next chat picks up
    # whatever the Evolve Agent just wrote.
    _per_agent.invalidate(tenant_id, agent_id)

    duration_s = time.time() - started
    logger.info(
        "evolve: iteration %s done in %.1fs verdict=%s edited=%d",
        iteration_id, duration_s, verification.verdict,
        len(evolve_outcome.files_edited),
    )

    result = IterationResult(
        iteration_id=iteration_id,
        agent_id=agent_id,
        tenant_id=tenant_id or None,
        verification=verification,
        rollout=rollout,
        evidence=evidence,
        evolve=evolve_outcome,
    )

    # 7. Bridge — publish a Lesson + eval report so the iteration
    # shows up as a review card in the Evolution timeline. Best-effort:
    # publication failures must not mask the iteration outcome.
    # The trajectory verifier (AHE §3.3) needs the agent's contract +
    # the prior iteration's pending manifest so each case can be graded
    # against claimed fixes / at-risk regressions.
    try:
        from runtime.evolution.bridge import publish_iteration
        system_prompt = workspace.read_system_prompt()
        publish_iteration(
            result,
            system_prompt=system_prompt,
            pending_before=pending_before or {},
            anthropic_key=anthropic_key,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("evolve: bridge publish failed: %s", exc, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _new_iteration_id() -> str:
    """Short, sortable, unique. Used in logs + future UI."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"evo-{stamp}-{secrets.token_hex(3)}"


def _resolve_executor_for_evolution(agent_id: str, tenant_id: str):
    """Same resolver the chat path uses, but pulling fallback inline.

    We can't import ``runtime.server._resolve_executor`` here without
    a circular dep, so the bits are inlined.
    """
    from runtime.executor import per_agent as _per_agent
    from runtime.tenants.feature import is_multi_tenant_enabled

    fallback = None  # populated below if needed
    try:
        from runtime.server import _state as _server_state
        fallback = _server_state.get("executor")
    except Exception:
        pass

    if not is_multi_tenant_enabled():
        if fallback is None:
            raise RuntimeError("evolve: no executor available (OSS mode + lifespan not booted)")
        return fallback

    executor = _per_agent.get_executor_for_agent(
        tenant_id, agent_id, fallback_executor=fallback,
    )
    if executor is None:
        raise RuntimeError(
            f"evolve: no executor available for ({tenant_id!r}, {agent_id!r})"
        )
    return executor


def _verify_previous(
    *,
    workspace: Any,
    pending: Optional[dict[str, Any]],
    rollout: RolloutResult,
) -> VerificationResult:
    """Compute verdict on the prior round's pending manifest + roll it
    to history. On ``regressed``, apply file-level rollback.

    Verdict heuristic (v0/v1 same):
      - ``confirmed`` if rollout has ZERO majority-fail tasks
      - ``regressed`` if rollout has fails AND at_risk_regressions
        were predicted
      - ``mixed`` if there are fails but no predictions for them
      - ``no_signal`` if there's no pending manifest at all
    """
    if pending is None:
        return VerificationResult(verdict="no_signal")

    failed = rollout.failed
    at_risk = pending.get("at_risk_regressions") or []
    if failed == 0:
        verdict = "confirmed"
    elif at_risk:
        verdict = "regressed"
    else:
        verdict = "mixed"

    delta = {
        "passed": rollout.passed,
        "failed": rollout.failed,
        "total": rollout.total_tasks,
        "flaky": len(rollout.flaky_tasks),
    }

    rollback_applied: list[str] = []
    if verdict == "regressed":
        try:
            rollback_applied = workspace.apply_rollback()
            if rollback_applied:
                logger.info(
                    "evolve: file-level rollback applied to %d files: %s",
                    len(rollback_applied), rollback_applied,
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("evolve: rollback failed: %s", exc, exc_info=True)
    else:
        # confirmed / mixed: the edits stand, snapshot is stale.
        try:
            workspace.clear_rollback_snapshot()
        except Exception:  # pragma: no cover
            pass

    archive = workspace.roll_pending_to_history(outcome={
        "verdict": verdict,
        "delta": delta,
        "rollback_applied": rollback_applied,
    })
    return VerificationResult(
        pending_archived_to=str(archive) if archive else None,
        verdict=verdict,
        delta={**delta, "rollback_applied": rollback_applied},
    )


def _snapshot_files(workspace: Any) -> dict[str, Optional[str]]:
    """Snapshot every UTF-8 file in the workspace into a path→content map.

    Used to populate the rollback snapshot AFTER we learn which files
    the Evolve Agent claims to have changed. We snapshot everything
    up front because we don't know in advance which files will be in
    the pending manifest; the post-evolve step picks the right entries.

    Binary files are skipped — Claude Code workspaces are text-heavy,
    and binary blobs (model weights, tarballs) shouldn't live here.
    """
    out: dict[str, Optional[str]] = {}
    try:
        files = workspace.list_files(max_files=10_000)
    except Exception:
        return out
    for rel in files:
        path = workspace.path.joinpath(rel)
        try:
            out[rel] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Skip binary or unreadable — rollback can't help here.
            continue
    return out


def _persist_rollback_snapshot(
    *,
    workspace: Any,
    iteration_id: str,
    pre_edit_files: dict[str, Optional[str]],
    pending_manifest: Optional[dict[str, Any]],
) -> None:
    """Write the rollback snapshot scoped to the files the pending
    manifest declared as changed.

    For paths the Evolve Agent CREATED (not in ``pre_edit_files``),
    record ``None`` — the rollback action becomes ``unlink``. For
    paths the agent EDITED, record the pre-edit content.

    No-op when the pending manifest is missing or declares no
    ``changed_files`` — nothing to roll back means nothing to save.
    """
    if not pending_manifest:
        return
    declared = pending_manifest.get("changed_files") or []
    if not declared:
        return

    snapshot: dict[str, Optional[str]] = {}
    for rel in declared:
        if not isinstance(rel, str) or not rel.strip():
            continue
        # Only record entries the agent actually shipped — if a path
        # was claimed but the workspace doesn't show it (lying agent
        # / tar miss), skip rather than poison the snapshot.
        post_edit = workspace.path.joinpath(rel)
        if not post_edit.exists():
            continue
        snapshot[rel] = pre_edit_files.get(rel)  # None if file was newly created

    if not snapshot:
        return
    try:
        workspace.write_rollback_snapshot(
            iteration_id=iteration_id, files=snapshot,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("evolve: failed to persist rollback snapshot: %s", exc)


def _evidence_for_evolve(evidence: Any) -> str:
    """Render the evidence (raw summary + clusters) for the Evolve Agent."""
    parts = [evidence.summary]
    if evidence.clusters:
        parts.append("")
        parts.append("--- Agent Debugger clusters ---")
        for c in sorted(evidence.clusters, key=lambda x: -x.severity):
            parts.append(
                f"[severity {c.severity}] {c.root_cause} "
                f"({len(c.tasks)} task(s))"
            )
            for t in c.tasks:
                parts.append(f"    - {t!r}")
            if c.notes:
                parts.append(f"    notes: {c.notes}")
    return "\n".join(parts)
