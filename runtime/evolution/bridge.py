"""Bridge each AHE iteration to the UI Lesson surface.

Without this module the evolution loop only writes a pending Change
Manifest in the workspace — operators never see a card in the
Evolution timeline. This makes every iteration produce two persistent
artifacts the UI already knows how to render:

  1. ``cand_<iteration_id>.json`` — eval report with one ``case`` per
     rollout outcome. Drives the Traces / Evals tabs in LessonDetail.
  2. A :class:`ledger.types.Lesson` with ``kind="change_manifest"``.
     Status is derived from the iteration outcome:
       - ``rolled_back`` if verdict=regressed (file-level rollback applied)
       - ``awaiting_review`` if Evolve Agent proposed edits
       - ``auto_promoted`` if confirmed and no edits (no-op iteration)

Publication is best-effort: a failure here logs and returns
``(None, None)`` rather than tanking the iteration result.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ledger.types import Lesson
from ledger.writer import write_lesson
from runtime.evolution.types import IterationResult


logger = logging.getLogger("runtime.evolution.bridge")


def publish_iteration(
    result: IterationResult,
    *,
    system_prompt: Optional[str] = None,
    pending_before: Optional[dict[str, Any]] = None,
    anthropic_key: Optional[str] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """Write the eval report + Lesson for ``result``. Returns
    ``(report_path, lesson_path)`` — either may be ``None`` on failure.

    When ``anthropic_key`` + ``system_prompt`` are supplied the per-case
    rubrics are upgraded from ``pipeline_succeeded`` only to a full
    trajectory verdict (follows_contract / claim_<i> / risk_<i>) per
    AHE §3.3. Otherwise we keep the mechanical default.
    """
    report_path: Optional[Path] = None
    rubric_aggregates: dict[str, float] = {}
    try:
        report_path, rubric_aggregates = _write_eval_report(
            result,
            system_prompt=system_prompt,
            pending_before=pending_before or {},
            anthropic_key=anthropic_key,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("bridge: eval report write failed: %s", exc, exc_info=True)

    lesson_path: Optional[Path] = None
    try:
        lesson_path = _write_lesson(
            result,
            has_report=report_path is not None,
            rubric_aggregates=rubric_aggregates,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("bridge: lesson write failed: %s", exc, exc_info=True)

    if lesson_path is not None:
        logger.info(
            "bridge: iteration %s published — lesson=%s report=%s",
            result.iteration_id, lesson_path, report_path,
        )
    return report_path, lesson_path


# ---------------------------------------------------------------------------
# Eval report
# ---------------------------------------------------------------------------


def eval_reports_dir() -> Path:
    """The canonical directory eval reports go in.

    Multi-tenant (staging/prod) → ``tenants/<tid>/evals/reports/`` so
    reports persist on gcsfuse alongside the rest of the tenant's data
    and don't collide across tenants.

    OSS / single-tenant → flat ``<project_root>/evals/reports/`` to
    match the legacy layout (baked-in candidate reports from the
    experiments runner live there)."""
    from runtime.tenants.feature import is_multi_tenant_enabled

    if is_multi_tenant_enabled():
        from runtime.tenant_context import get_active
        from runtime.tenants.registry import get_tenant_dir

        return get_tenant_dir(get_active()) / "evals" / "reports"
    return Path(__file__).resolve().parent.parent.parent / "evals" / "reports"


def _write_eval_report(
    result: IterationResult,
    *,
    system_prompt: Optional[str],
    pending_before: dict[str, Any],
    anthropic_key: Optional[str],
) -> tuple[Path, dict[str, float]]:
    """One case per rollout outcome. Shape mirrors the existing
    experiments-runner reports so the UI's Traces tab can read it
    without a new code path.

    Each case gets:
      - ``pipeline_succeeded`` rubric (mechanical, always)
      - trajectory verdicts (``follows_contract``, ``claim_<i>``,
        ``risk_<i>``) when ``anthropic_key`` is supplied — per AHE
        §3.3 trajectory-level evaluation.
    """
    claimed_fixes = list(pending_before.get("claimed_fixes") or [])
    at_risk_regressions = list(pending_before.get("at_risk_regressions") or [])

    from runtime.evolution.verifier import verify_trajectory

    cases: list[dict[str, Any]] = []
    for i, o in enumerate(result.rollout.outcomes):
        rubrics: list[dict[str, Any]] = [{
            "rubric": "pipeline_succeeded",
            "type": "pipeline_success",
            "score": 1.0 if o.success else 0.0,
            "passed": o.success,
            "detail": o.error,
        }]
        traj_rubrics = verify_trajectory(
            task=o.task,
            response=o.response,
            success=o.success,
            error=o.error,
            system_prompt=system_prompt,
            claimed_fixes=claimed_fixes,
            at_risk_regressions=at_risk_regressions,
            anthropic_key=anthropic_key,
        )
        if traj_rubrics:
            rubrics.extend(traj_rubrics)
        cases.append({
            "golden_id": f"rollout-{i:03d}-run{o.run_index}",
            "request": o.task,
            "response": o.response,
            "duration_ms": o.duration_ms,
            "success": o.success,
            "error": o.error,
            "trace_id": o.trace_id,
            "rubric_results": rubrics,
        })
    report = {
        "suite": f"ahe-rollout-{result.iteration_id}",
        "agent_version": None,
        "started_at": _now_iso(),
        "finished_at": _now_iso(),
        "iteration_id": result.iteration_id,
        "agent_id": result.agent_id,
        "verdict": result.verification.verdict,
        "cases": cases,
    }
    out = eval_reports_dir() / f"cand_{result.iteration_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    aggregates = _aggregate_rubrics(cases)
    return out, aggregates


def _aggregate_rubrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Per-rubric pass rate across all cases.

    Returns ``{rubric_name: pass_rate}`` with pass_rate ∈ [0, 1].
    Powers the Lesson's ``delta.per_rubric`` so the UI Evals tab and
    the Story sidebar can render numbers without re-reading every case.
    """
    totals: dict[str, int] = {}
    passes: dict[str, int] = {}
    for c in cases:
        for r in c.get("rubric_results") or []:
            name = r.get("rubric")
            if not isinstance(name, str):
                continue
            totals[name] = totals.get(name, 0) + 1
            if r.get("passed"):
                passes[name] = passes.get(name, 0) + 1
    return {
        name: round(passes.get(name, 0) / n, 4)
        for name, n in totals.items()
        if n > 0
    }


# ---------------------------------------------------------------------------
# Lesson
# ---------------------------------------------------------------------------


def _write_lesson(
    result: IterationResult,
    *,
    has_report: bool,
    rubric_aggregates: dict[str, float],
) -> Optional[Path]:
    files_edited = list(result.evolve.files_edited or [])
    pending = result.evolve.pending_manifest or {}
    verdict = result.verification.verdict
    overall_score = (
        rubric_aggregates.get("trajectory_quality")
        or rubric_aggregates.get("pipeline_succeeded")
        or 0.0
    )

    # Consult the approval policy. Policy modes (paper §3.4 governance):
    #   off    → don't surface this iteration as a review at all
    #   auto   → if Δ clears auto_min_lift, promote without human (snapshot
    #            + version bump same as the manual approve path)
    #   review → land as awaiting_review (default; current behavior)
    policy_decision = _decide_status_via_policy(
        verdict=verdict,
        has_edits=bool(files_edited),
        overall_score=overall_score,
    )
    if policy_decision.skip:
        logger.info(
            "bridge: policy mode=off — skipping lesson publish for iter %s",
            result.iteration_id,
        )
        return None
    status = policy_decision.status

    title = _title_for(result, files_edited, verdict)
    summary = _summary_for(result, files_edited, has_report=has_report)
    voice = _voice_for(result, pending)

    delta: dict[str, Any] = {
        "passed": result.rollout.passed,
        "failed": result.rollout.failed,
        "total": result.rollout.total_tasks,
        "flaky": len(result.rollout.flaky_tasks),
        "k": result.rollout.k,
        "verdict": verdict,
    }
    if pending:
        delta["claimed_fixes"] = list(pending.get("claimed_fixes") or [])
        delta["at_risk_regressions"] = list(
            pending.get("at_risk_regressions") or []
        )
    if result.evidence.clusters:
        delta["clusters"] = [c.to_dict() for c in result.evidence.clusters]
    # UI-facing aggregates: ``per_rubric`` powers the Evals tab,
    # ``overall_score`` + ``pass_rate`` power the Story sidebar.
    # Use trajectory_quality when available (it's the verifier's
    # all-up verdict); fall back to the mechanical pipeline rubric.
    if rubric_aggregates:
        delta["per_rubric"] = rubric_aggregates
        delta["overall_score"] = (
            rubric_aggregates.get("trajectory_quality")
            or rubric_aggregates.get("pipeline_succeeded")
            or 0.0
        )
        if result.rollout.total_tasks > 0:
            delta["pass_rate"] = round(
                result.rollout.passed / result.rollout.total_tasks, 4
            )

    # ``promoted_at`` here follows the convention set by the
    # ``agent_created`` lesson in ``runtime/store/onboarding.py``:
    # the field marks "when this lesson was finalized on disk", not
    # exclusively "when it went live". Setting it on every iteration
    # so the Evolution timeline can order/display the row.
    now = _now_iso()

    # When auto-promote fires, snapshot the workspace as a new version
    # (same code path the manual approve uses) so Versions tab + the
    # ledger see this iteration land as a real release. Without this,
    # "auto_promoted" would be only a status sticker with no rollback
    # target.
    version: Optional[str] = None
    if status == "auto_promoted" and files_edited:
        try:
            from runtime.workspaces import get_workspace
            workspace = get_workspace(result.agent_id)
            version = workspace.bump_and_snapshot(
                reason=f"auto-promote iter {result.iteration_id}"
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "bridge: auto-promote snapshot failed: %s", exc, exc_info=True
            )

    lesson = Lesson(
        id=_new_lesson_id(),
        kind="change_manifest",
        status=status,
        title=title,
        summary=summary,
        voice=voice,
        delta=delta,
        mutations=files_edited,
        parent_version="",
        candidate_id=result.iteration_id,
        version=version,
        promoted_at=now,
        ledger_entry_id="",
        proposal_source="claude_code",
    )
    lesson_path = write_lesson(lesson, agent_id=result.agent_id)

    # Email the operator when this iteration needs a human decision.
    # Auto-promoted ones land silently — they're already live; the
    # operator can review the lesson card without being paged.
    if status == "awaiting_review" and result.tenant_id:
        _notify_lesson_awaiting_review(
            tenant_id=result.tenant_id, lesson=lesson, files_edited=files_edited,
        )

    # Mirror the manual approve flow: write a `promote` ledger entry
    # for auto-promoted iterations so introspection ("what changed
    # today?") and metrics (trust score) pick it up.
    if status == "auto_promoted" and version:
        try:
            from ledger.writer import write_entry
            write_entry(
                kind="promote",
                candidate_id=result.iteration_id,
                agent_id=result.agent_id,
                agent_version_after=version,
                summary=f"auto-approved AHE change_manifest (lesson {lesson.id})",
                payload={
                    "lesson_id": lesson.id,
                    "kind": "change_manifest",
                    "mutations": files_edited,
                    "delta": delta,
                    "human_approved": False,
                    "auto_promote": True,
                    "agent_version_after": version,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("bridge: auto-promote entry failed: %s", exc)

    return lesson_path


def _status_for(*, verdict: str, has_edits: bool) -> str:
    """Status when no policy applies — keeps backward-compat for tests
    that don't have a policy/workspace setup. Real decisions go through
    :func:`_decide_status_via_policy`."""
    if verdict == "regressed":
        return "rolled_back"
    if has_edits:
        return "awaiting_review"
    return "auto_promoted"


class _PolicyDecision:
    """Internal struct: status + whether to skip publication entirely."""

    __slots__ = ("status", "skip")

    def __init__(self, status: str, skip: bool = False) -> None:
        self.status = status
        self.skip = skip


_POLICY_KIND = "change_manifest"


def _decide_status_via_policy(
    *, verdict: str, has_edits: bool, overall_score: float
) -> _PolicyDecision:
    """Map policy mode + iteration outcome → lesson status.

    Rules (paper §3.4 governance + our auto_min_lift gate):
      - verdict ``regressed`` → ``rolled_back`` (rollback already done
        in the loop; status mirrors that for the timeline)
      - no edits → ``auto_promoted`` regardless of mode (nothing to
        gate — the iteration was a no-op)
      - mode ``off`` → skip publish entirely
      - mode ``auto`` + ``overall_score >= auto_min_lift`` → ``auto_promoted``
      - mode ``auto`` + score below threshold → ``awaiting_review``
        (fall back to human review rather than silently dropping)
      - mode ``review`` (default) → ``awaiting_review``
    """
    if verdict == "regressed":
        return _PolicyDecision("rolled_back")
    if not has_edits:
        return _PolicyDecision("auto_promoted")
    try:
        from runtime.server import _policy_path
        from harness.approver import Policy
        policy = Policy.from_yaml(_policy_path())
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("bridge: policy load failed (%s) — defaulting to review", exc)
        return _PolicyDecision("awaiting_review")

    mode = policy.mode_for(_POLICY_KIND)
    if mode == "off":
        return _PolicyDecision("awaiting_review", skip=True)
    if mode == "auto":
        if overall_score >= policy.auto_min_lift:
            return _PolicyDecision("auto_promoted")
        logger.info(
            "bridge: policy=auto but score %.4f < threshold %.4f — review",
            overall_score, policy.auto_min_lift,
        )
        return _PolicyDecision("awaiting_review")
    return _PolicyDecision("awaiting_review")


def _title_for(result: IterationResult, files_edited: list[str], verdict: str) -> str:
    if files_edited:
        n = len(files_edited)
        suffix = "edit" if n == 1 else "edits"
        return f"Evolve proposed {n} {suffix} — {verdict}"
    return f"Evolve iteration — {verdict} (no edits)"


def _summary_for(
    result: IterationResult, files_edited: list[str], *, has_report: bool
) -> str:
    rollout = result.rollout
    parts = [
        f"Rollout {rollout.passed}/{rollout.total_tasks} passed "
        f"(k={rollout.k}, {len(rollout.flaky_tasks)} flaky).",
    ]
    if files_edited:
        preview = ", ".join(files_edited[:5])
        if len(files_edited) > 5:
            preview += f" (+{len(files_edited) - 5} more)"
        parts.append(f"Files: {preview}.")
    else:
        parts.append("No edits proposed by Evolve Agent.")
    if not has_report:
        parts.append("(eval report missing — diagnostics only)")
    return " ".join(parts)


def _voice_for(result: IterationResult, pending: dict[str, Any]) -> str:
    rationale = pending.get("rationale") if pending else None
    if isinstance(rationale, str) and rationale.strip():
        return rationale.strip()[:280]
    raw = result.evolve.raw_response or ""
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    if first_line:
        return first_line.strip()[:280]
    return (
        f"Ran one AHE iteration on {result.agent_id}: "
        f"{result.rollout.passed}/{result.rollout.total_tasks} passed, "
        f"verdict={result.verification.verdict}."
    )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _new_lesson_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"L-{stamp}-{secrets.token_hex(2)}"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _notify_lesson_awaiting_review(
    *, tenant_id: str, lesson: Lesson, files_edited: list[str]
) -> None:
    """Best-effort email the operator when a change_manifest needs
    human review. Silently drops when transport / recipient are
    missing — auditable copy lives on the lesson itself."""
    try:
        from runtime.notifications.sender import operator_email_for_tenant, send_email
        recipient = operator_email_for_tenant(tenant_id)
        if not recipient:
            logger.info("bridge: no operator email for tenant %s — skip notify", tenant_id)
            return
        ui_origin = os.environ.get("OPENTRACY_UI_ORIGIN", "https://app.dev.opentracy.cloud")
        link = f"{ui_origin}/lesson/{lesson.id}"
        files_line = (
            ", ".join(files_edited[:5])
            + (f" (+{len(files_edited) - 5} more)" if len(files_edited) > 5 else "")
            if files_edited else "(no harness file changes)"
        )
        text = (
            f"Your agent proposed an iteration that needs review.\n\n"
            f"Lesson: {lesson.title}\n"
            f"Files: {files_line}\n"
            f"Voice: {lesson.voice or '(no voice)'}\n\n"
            f"Review it: {link}\n\n"
            f"— Opentracy"
        )
        send_email(
            to=recipient,
            subject=f"[Opentracy] {lesson.title}",
            text=text,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("bridge: notify failed (%s)", exc)


import os  # noqa: E402 — used by _notify_lesson_awaiting_review only
