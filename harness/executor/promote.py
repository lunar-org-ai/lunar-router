"""Executor — promote an approved candidate to live agent/.

Steps:
  1. Snapshot current live agent → ledger/versions/<old_version>/
  2. Copy candidate's agent tree → live agent/
  3. Bump version in live agent.yaml (patch +1)
  4. Append LedgerEntry (kind=promote) tying old/new versions to the candidate.
  5. Write a Lesson (UI card) summarizing the change + delta.

Two entry points:
  - promote(outcome): used by the auto loop with an in-memory LoopOutcome.
  - promote_queued(lesson_id): used by the approve endpoint to finalize a
    previously-queued review lesson (status=awaiting_review → approved).
"""

from __future__ import annotations

import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from experiments.branching import candidate_agent_path
from harness.types import LoopOutcome, kind_from_mutations
from ledger.types import Lesson
from ledger.versioning import LIVE_AGENT, read_version, snapshot_agent
from ledger.writer import read_lesson, update_lesson, write_entry, write_lesson


def _bump_patch(version: str) -> str:
    """v0.0.1 → v0.0.2 (or 0.0.1 → 0.0.2 if no `v` prefix)."""
    has_v = version.startswith("v")
    core = version[1:] if has_v else version
    parts = core.split(".")
    if not parts[-1].isdigit():
        # fallback: append .1
        parts.append("1")
    else:
        parts[-1] = str(int(parts[-1]) + 1)
    return ("v" if has_v else "") + ".".join(parts)


def _set_version(agent_yaml: Path, new_version: str) -> None:
    with agent_yaml.open() as f:
        doc = yaml.safe_load(f)
    doc["agent"]["version"] = new_version
    with agent_yaml.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


_kind_from_mutations = kind_from_mutations  # kept for callers below


def _parse_mutation(describe: str) -> tuple[str, str, str]:
    """`pipeline/retrieve.yaml:knobs.k=12` → ('retrieve', 'k', '12')."""
    file_part, rest = describe.split(":", 1)
    path_part, value = rest.split("=", 1)
    leaf = path_part.split(".")[-1]
    stage = file_part.split("/")[-1].split(".")[0]
    return stage, leaf, value


_VOICES_PATH = Path(__file__).resolve().parent / "voices.yaml"
_voices_cache: Optional[dict[str, Any]] = None


def _voices() -> dict[str, Any]:
    """Lazy-load voices.yaml. Templates live as data so non-engineers can edit."""
    global _voices_cache
    if _voices_cache is None:
        with _VOICES_PATH.open() as f:
            _voices_cache = yaml.safe_load(f) or {}
    return _voices_cache


def _outcome_phrase(delta: float) -> str:
    outcomes = _voices().get("outcomes", {})
    if delta > 0.005:
        return outcomes.get("improvement", "")
    if delta < -0.005:
        return outcomes.get("regression", "")
    return outcomes.get("neutral", "")


def _voice_for(
    kind: str,
    mutations: list[str],
    delta_overall: float,
    prediction_rationale: Optional[str],
) -> tuple[str, str]:
    """Return (title, voice). All template content lives in voices.yaml.

    Resolution order:
      1. Prediction.rationale, if the proposer wrote one.
      2. (kind, knob) lookup in templates.
      3. (kind, "_default") fallback for the kind.
      4. "other" generic.
      Plus a "multi" path for multi-mutation lessons.
    """
    cfg = _voices()
    outcome = _outcome_phrase(delta_overall)

    # 1. Prediction-driven
    if prediction_rationale:
        rationale = prediction_rationale.strip().rstrip(".")
        voice_tpl = cfg.get("prediction", {}).get("voice", "I figured {rationale}. {outcome}")
        title = rationale.split(",")[0].split("→")[0].strip().capitalize()
        if len(title) > 60:
            title = title[:57].rstrip() + "…"
        return title, voice_tpl.format(rationale=rationale, outcome=outcome).strip()

    # 2. Multi-mutation
    if len(mutations) > 1:
        multi = cfg.get("multi", {})
        title = multi.get("title", "Made {n} changes to my {kind}").format(
            n=len(mutations), kind=kind
        )
        voice = multi.get(
            "voice", "I tried out {n} adjustments to my {kind}. {outcome}"
        ).format(n=len(mutations), kind=kind, outcome=outcome)
        return title, voice.strip()

    # 3. Single-mutation lookup
    knob = ""
    if mutations:
        try:
            _stage, knob, _value = _parse_mutation(mutations[0])
        except ValueError:
            knob = ""

    by_kind = cfg.get("templates", {}).get(kind, {})
    template = by_kind.get(knob) or by_kind.get("_default") or cfg.get("other", {})

    title = template.get("title", "Made a change")
    voice_body = template.get("voice", "I made a small adjustment.")
    voice = f"{voice_body.strip()} {outcome}".strip()
    return title, voice


def _summary_for(status: str, parent_version: str, delta_overall: float) -> str:
    moved = abs(delta_overall) > 0.0005
    if status == "auto_promoted":
        if moved:
            return f"Auto-promoted from {parent_version}. Δoverall on the eval suite: {delta_overall:+.3f}."
        return (
            f"Auto-promoted from {parent_version}. Eval didn't move much, but the change was "
            "non-regressing and the policy auto-approved it."
        )
    if status == "approved":
        if moved:
            return f"Approved from {parent_version} after human review. Δoverall: {delta_overall:+.3f}."
        return f"Approved from {parent_version} after human review. Eval barely moved."
    if status == "awaiting_review":
        if moved:
            return (
                f"Queued for your review. Branched from {parent_version}; "
                f"candidate Δoverall: {delta_overall:+.3f}."
            )
        return (
            f"Queued for your review. Branched from {parent_version}; "
            "candidate didn't move evals much but didn't regress either."
        )
    return f"Change rooted at {parent_version} (status={status})."


def build_lesson(
    outcome: LoopOutcome,
    *,
    status: str,
    parent_version: str,
    new_version: Optional[str] = None,
    entry_id: Optional[str] = None,
    promoted_at: Optional[str] = None,
) -> Lesson:
    """Pure builder — no filesystem side effects. Used by both auto-promote
    and queue-for-review paths.
    """
    mutations = [m.describe() for m in outcome.proposal.mutations]
    kind = _kind_from_mutations(mutations)
    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    )
    delta = outcome.candidate_result.delta if outcome.candidate_result else {}
    delta_overall = delta.get("overall_score", 0.0) if delta else 0.0

    prediction_rationale = (
        outcome.proposal.prediction.rationale if outcome.proposal.prediction else None
    )
    title, voice = _voice_for(kind, mutations, delta_overall, prediction_rationale)
    summary = _summary_for(status, parent_version, delta_overall)

    return Lesson(
        id=lesson_id,
        version=new_version,
        kind=kind,
        status=status,
        title=title,
        summary=summary,
        proposal_source=outcome.proposal.source,
        delta=delta,
        mutations=mutations,
        parent_version=parent_version,
        candidate_id=outcome.candidate_id or "",
        promoted_at=promoted_at,
        ledger_entry_id=entry_id,
        voice=voice,
    )


def promote(
    outcome: LoopOutcome,
    *,
    agent_dir: Path | str = LIVE_AGENT,
) -> tuple[str, str]:
    """Promote `outcome.candidate` to live agent/. Returns (new_version, lesson_id)."""
    if not outcome.approved or outcome.candidate_id is None:
        raise ValueError("promote() requires an approved outcome with a candidate_id")

    agent_dir = Path(agent_dir)

    # 1. Snapshot current
    old_version, _ = snapshot_agent(agent_dir)

    # 2. Replace live with candidate
    cand_agent_dir = candidate_agent_path(outcome.candidate_id).parent
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    shutil.copytree(cand_agent_dir, agent_dir)

    # 3. Bump version
    new_version = _bump_patch(old_version)
    _set_version(agent_dir / "agent.yaml", new_version)

    # 4. Ledger entry
    delta = outcome.candidate_result.delta if outcome.candidate_result else {}
    delta_overall = delta.get("overall_score", 0.0) if delta else 0.0

    payload: dict = {
        "mutations": [m.describe() for m in outcome.proposal.mutations],
        "delta": delta,
        "verdicts": [
            {"critic": v.critic, "approved": v.approved, "reason": v.reason}
            for v in outcome.verdicts
        ],
    }
    # AHE pillar 3 — record prediction + verification when present
    if outcome.proposal.prediction is not None:
        payload["prediction"] = {
            "rubric": outcome.proposal.prediction.rubric,
            "expected_delta": outcome.proposal.prediction.expected_delta,
            "rationale": outcome.proposal.prediction.rationale,
            "confidence": outcome.proposal.prediction.confidence,
        }
    if outcome.verification is not None:
        payload["verification"] = {
            "rubric": outcome.verification.rubric,
            "expected_delta": outcome.verification.expected_delta,
            "actual_delta": outcome.verification.actual_delta,
            "direction_correct": outcome.verification.direction_correct,
            "magnitude_met": outcome.verification.magnitude_met,
            "verdict": outcome.verification.verdict,
        }

    entry = write_entry(
        kind="promote",
        candidate_id=outcome.candidate_id,
        agent_version_before=old_version,
        agent_version_after=new_version,
        summary=f"promoted with Δoverall={delta_overall:+.4f}",
        payload=payload,
    )

    # 5. Lesson
    lesson = build_lesson(
        outcome,
        status="auto_promoted",
        parent_version=old_version,
        new_version=new_version,
        entry_id=entry.entry_id,
        promoted_at=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    )
    write_lesson(lesson)

    return new_version, lesson.id


def promote_queued(
    lesson_id: str,
    *,
    agent_dir: Path | str = LIVE_AGENT,
    reviewer: Optional[str] = None,
) -> Lesson:
    """Finalize a previously queued review lesson.

    The candidate is still on disk under experiments/candidates/<cand_id>/.
    We snapshot live, copy candidate over, bump version, write a promote
    ledger entry that references the queued_review entry as parent, and
    mutate the lesson in place (status=approved, version, promoted_at,
    ledger_entry_id pointing to the new promote entry).
    """
    lesson = read_lesson(lesson_id)
    if lesson is None:
        raise FileNotFoundError(f"lesson {lesson_id!r} not found")
    if lesson.status != "awaiting_review":
        raise ValueError(
            f"lesson {lesson_id!r} has status {lesson.status!r}; only awaiting_review can be approved"
        )
    if not lesson.candidate_id:
        raise ValueError(f"lesson {lesson_id!r} carries no candidate_id; cannot promote")

    agent_dir = Path(agent_dir)
    cand_agent_dir = candidate_agent_path(lesson.candidate_id).parent
    if not cand_agent_dir.exists():
        raise FileNotFoundError(
            f"candidate dir {cand_agent_dir} missing — cannot promote queued lesson"
        )

    old_version, _ = snapshot_agent(agent_dir)

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    shutil.copytree(cand_agent_dir, agent_dir)

    new_version = _bump_patch(old_version)
    _set_version(agent_dir / "agent.yaml", new_version)

    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    payload: dict[str, Any] = {
        "mutations": lesson.mutations,
        "delta": lesson.delta,
        "lesson_id": lesson.id,
        "human_approved": True,
    }
    if reviewer:
        payload["reviewer"] = reviewer

    entry = write_entry(
        kind="promote",
        candidate_id=lesson.candidate_id,
        agent_version_before=old_version,
        agent_version_after=new_version,
        parent_entry_id=lesson.ledger_entry_id,
        summary=f"approved + promoted from review queue (lesson {lesson.id})",
        payload=payload,
    )

    return update_lesson(
        lesson.id,
        status="approved",
        version=new_version,
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
    )


def record_manual_change(
    apply_edit: Any,
    *,
    kind: str,
    summary: str,
    mutations_desc: list[str],
    voice: Optional[str] = None,
    agent_dir: Path | str = LIVE_AGENT,
) -> Lesson:
    """Apply a human-driven change to the agent surface and record it like a
    proposer-driven promotion.

    AutoHarness paper (arxiv 2603.03329) treats the agent's editable surface
    as a single line of history regardless of who proposed each edit.
    Operator edits via AgentSheet go through the same snapshot + version
    bump + ledger + Lesson dance the auto loop uses for candidates, so:

      - rolling back a manual edit works exactly like rolling back a
        candidate-promoted lesson
      - the Evolution timeline shows manual edits alongside auto edits
      - lesson.proposal_source = "human" makes the source legible
      - lesson.candidate_id is empty (no eval suite ran)

    `apply_edit` is invoked between the old-version snapshot and the
    version bump so the snapshot captures pre-edit state.
    """
    agent_dir = Path(agent_dir)

    old_version, _ = snapshot_agent(agent_dir)

    apply_edit()  # caller writes whatever file(s) the edit touches

    new_version = _bump_patch(old_version)
    _set_version(agent_dir / "agent.yaml", new_version)

    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    entry = write_entry(
        kind="promote",
        agent_version_before=old_version,
        agent_version_after=new_version,
        summary=f"manual edit: {summary}",
        payload={
            "source": "human",
            "kind": kind,
            "mutations": mutations_desc,
        },
    )

    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    )
    lesson = Lesson(
        id=lesson_id,
        version=new_version,
        kind=kind,
        status="approved",
        title=summary,
        summary=(
            f"Manual operator edit from {old_version}. "
            "No eval ran — applied directly to the live surface."
        ),
        proposal_source="human",
        delta={},
        mutations=mutations_desc,
        parent_version=old_version,
        candidate_id="",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice=voice or f"I updated my {kind} directly.",
    )
    write_lesson(lesson)
    return lesson


def record_manual_router_change(
    new_payload: dict,
    *,
    summary: str,
    voice: Optional[str] = None,
    versions_dir: Optional[Path] = None,
) -> Lesson:
    """Manual operator edit of router_config — AHE-aligned variant.

    Mirrors ``record_manual_change`` but specialized for router_config:
      - **No agent_dir snapshot** (the agent code is unchanged; only
        versions/router_config_<n>.json changed).
      - **No agent.yaml version bump** (router_config has its own version
        baked into the artifact).
      - Uses ``apply_router_candidate`` for the atomic write + pointer flip.

    P15.3.8's ``PUT /v1/router/config`` calls this so manual λ overrides
    surface in Evolution as ``Lesson(kind="router_config",
    proposal_source="human")``, rolling back via the standard
    ``/v1/versions/{v}/rollback`` machinery.
    """
    json_path, _ = apply_router_candidate(new_payload, versions_dir=versions_dir)
    new_version = int(new_payload["version"])
    parent_version = max(0, new_version - 1)

    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    mutations_desc = [f"versions/router_config_v{new_version}.json"]
    entry = write_entry(
        kind="promote",
        summary=f"manual router_config edit: {summary}",
        payload={
            "source": "human",
            "kind": "router_config",
            "mutations": mutations_desc,
            "json_path": str(json_path),
        },
    )

    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    )
    lesson = Lesson(
        id=lesson_id,
        version=str(new_version),
        kind="router_config",
        status="approved",
        title=summary,
        summary=(
            f"Manual operator edit of router_config (v{parent_version} → v{new_version}). "
            "No eval ran — applied directly to the live surface."
        ),
        proposal_source="human",
        delta={},
        mutations=mutations_desc,
        parent_version=str(parent_version),
        candidate_id="",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice=voice or "I tweaked my router config directly.",
    )
    write_lesson(lesson)
    return lesson


def record_manual_dataset_change(
    *,
    name: str,
    new_version: int,
    summary: str,
    apply_edit: Any,
    voice: Optional[str] = None,
) -> Lesson:
    """Manual operator edit of a dataset — AHE-aligned variant for P15.4.

    Mirrors ``record_manual_router_change`` but specialized for datasets:
      - **No agent_dir snapshot** (datasets live outside the agent's
        editable surface; only ``datasets/<name>/v<n>.json`` changes).
      - **No agent.yaml version bump** (datasets have their own version
        chain baked into the artifact).
      - Caller passes ``apply_edit`` which performs the actual
        ``save_dataset(payload)`` write — same shape as
        ``record_manual_change``.

    P15.4.2's POST/PUT ``/v1/datasets`` calls this so manual create/edit
    surfaces in Evolution as ``Lesson(kind="dataset",
    proposal_source="human")``.
    """
    apply_edit()
    parent_version = max(0, new_version - 1)

    promoted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    mutations_desc = [f"datasets/{name}/v{new_version}.json"]
    entry = write_entry(
        kind="promote",
        summary=f"manual dataset edit: {summary}",
        payload={
            "source": "human",
            "kind": "dataset",
            "name": name,
            "mutations": mutations_desc,
        },
    )

    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    )
    lesson = Lesson(
        id=lesson_id,
        version=str(new_version),
        kind="dataset",
        status="approved",
        title=summary,
        summary=(
            f"Manual operator edit of dataset {name!r} (v{parent_version} → v{new_version}). "
            "No eval ran — applied directly to the live surface."
        ),
        proposal_source="human",
        delta={},
        mutations=mutations_desc,
        parent_version=str(parent_version),
        candidate_id="",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice=voice or f"I edited the '{name}' dataset directly.",
    )
    write_lesson(lesson)
    return lesson


def reject_queued(lesson_id: str, *, reason: Optional[str] = None) -> Lesson:
    """Mark a queued review lesson as human_rejected; no live agent change."""
    lesson = read_lesson(lesson_id)
    if lesson is None:
        raise FileNotFoundError(f"lesson {lesson_id!r} not found")
    if lesson.status != "awaiting_review":
        raise ValueError(
            f"lesson {lesson_id!r} has status {lesson.status!r}; only awaiting_review can be rejected"
        )

    entry = write_entry(
        kind="rejected",
        candidate_id=lesson.candidate_id or None,
        parent_entry_id=lesson.ledger_entry_id,
        summary=f"human rejected lesson {lesson.id}" + (f": {reason}" if reason else ""),
        payload={"lesson_id": lesson.id, "reason": reason},
    )

    return update_lesson(
        lesson.id, status="human_rejected", ledger_entry_id=entry.entry_id
    )


def requeue(lesson_id: str, *, agent_dir: Path | str = LIVE_AGENT) -> Lesson:
    """Undo an approve or reject — put the lesson back into the review queue.

    For human_rejected: flip status, write a fresh queued_review entry.
    For approved: rollback the live agent to parent_version first, then flip
    status and clear version/promoted_at. Useful for the Undo button right
    after a hasty click.
    """
    lesson = read_lesson(lesson_id)
    if lesson is None:
        raise FileNotFoundError(f"lesson {lesson_id!r} not found")
    if lesson.status not in ("human_rejected", "approved"):
        raise ValueError(
            f"lesson {lesson_id!r} has status {lesson.status!r}; only approved or human_rejected can be requeued"
        )

    # Approved → rollback live first so on-disk state matches the queue state.
    if lesson.status == "approved":
        if not lesson.parent_version:
            raise ValueError(
                f"approved lesson {lesson.id} missing parent_version; cannot rollback"
            )
        from harness.rollback import rollback_to

        rollback_to(lesson.parent_version, reason=f"undo approve of lesson {lesson.id}")

    entry = write_entry(
        kind="queued_review",
        candidate_id=lesson.candidate_id or None,
        agent_version_before=read_version(Path(agent_dir)),
        parent_entry_id=lesson.ledger_entry_id,
        summary=f"requeued lesson {lesson.id} (was {lesson.status})",
        payload={"lesson_id": lesson.id, "previous_status": lesson.status},
    )

    return update_lesson(
        lesson.id,
        status="awaiting_review",
        version=None,
        promoted_at=None,
        ledger_entry_id=entry.entry_id,
    )


# ---------------------------------------------------------------------------
# P15.3.7 — router_config promotion path
# ---------------------------------------------------------------------------
#
# router_config Lessons don't go through the agent_dir snapshot/copy/version
# bump used above — there's no agent code change to promote, just a JSON
# artifact + sidecar centroids. apply_router_candidate writes the artifact
# atomically; promote_router_config wraps that with the same ledger + Lesson
# emission ``promote()`` does for prompt/RAG/route changes.
#
# Cross-source uniformity (AHE alignment): manual operator edits via
# ``record_manual_change(kind="router_config", apply_edit=...)`` go through
# ``apply_router_candidate`` too — the only difference is proposal_source.


def apply_router_candidate(
    candidate_payload: dict,
    *,
    versions_dir: Optional[Path] = None,
) -> tuple[Path, Optional[Path]]:
    """Atomically write versions/router_config_v<n>.json + .npz sidecar.

    The current pointer flips after both files are durable.

    Args:
        candidate_payload: The proposer's inline payload (from
            Mutation.value). Must contain 'version', 'k', 'model_psi'.
            The 'centroids' value, if a list/array, gets pulled out into
            the sidecar .npz so the JSON stays small.
        versions_dir: Override the default ``versions/`` directory
            (tests use this).

    Returns:
        ``(json_path, centroids_path_or_None)``.
    """
    import numpy as np

    from router.config_io import save_config

    centroids = candidate_payload.get("centroids")
    centroids_arr: Optional[Any] = None
    if centroids is not None:
        centroids_arr = np.asarray(centroids, dtype=float)
        # Drop the inline copy so the JSON stays compact; sidecar is the
        # source of truth.
        candidate_payload = {**candidate_payload, "centroids": None}

    json_path = save_config(
        candidate_payload,
        centroids=centroids_arr,
        versions_dir=versions_dir,
        update_pointer=True,
    )

    npz_path: Optional[Path] = None
    if centroids_arr is not None:
        from router.config_io import _centroids_path, _vd

        npz_path = _centroids_path(
            int(candidate_payload["version"]), versions_dir=_vd(versions_dir)
        )

    return json_path, npz_path


def promote_router_config(
    outcome: LoopOutcome,
    *,
    versions_dir: Optional[Path] = None,
) -> tuple[int, str]:
    """Promote a router_config candidate. Returns ``(new_version, lesson_id)``.

    Mirrors ``promote(outcome)`` but for ``kind="router_config"`` proposals.
    The candidate payload lives inline on ``outcome.proposal.mutations[0].value``;
    no agent_dir snapshot is taken (there's no agent code change).
    """
    if not outcome.proposal.mutations:
        raise ValueError("promote_router_config requires a proposal with mutations")
    payload = outcome.proposal.mutations[0].value
    if not isinstance(payload, dict):
        raise ValueError(
            "promote_router_config expects dict payload on Mutation.value, "
            f"got {type(payload).__name__}"
        )

    # 1. Apply the candidate (atomic write + pointer flip).
    json_path, npz_path = apply_router_candidate(payload, versions_dir=versions_dir)
    new_version = int(payload["version"])

    promoted_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    # 2. Ledger entry — record the candidate metadata for later inspection.
    metadata = payload.get("metadata") or {}
    delta_overall = float(metadata.get("silhouette", 0.0))
    ledger_payload: dict = {
        "kind": "router_config",
        "source": outcome.proposal.source,
        "json_path": str(json_path),
        "npz_path": str(npz_path) if npz_path else None,
        "k": payload.get("k"),
        "n_models": len(payload.get("model_psi") or {}),
        "fitted_from": payload.get("fitted_from"),
        "metadata": metadata,
        "verdicts": [
            {"critic": v.critic, "approved": v.approved, "reason": v.reason}
            for v in outcome.verdicts
        ],
    }
    if outcome.proposal.prediction is not None:
        ledger_payload["prediction"] = {
            "rubric": outcome.proposal.prediction.rubric,
            "expected_delta": outcome.proposal.prediction.expected_delta,
            "rationale": outcome.proposal.prediction.rationale,
            "confidence": outcome.proposal.prediction.confidence,
        }

    entry = write_entry(
        kind="promote",
        candidate_id=outcome.candidate_id or "",
        agent_version_before=None,           # router_config doesn't affect agent.yaml version
        agent_version_after=None,
        summary=f"promoted router_config v{new_version} (silhouette={delta_overall:.3f})",
        payload=ledger_payload,
    )

    # 3. Lesson — uses the existing build_lesson scaffold so the Evolution
    # timeline picks it up uniformly.
    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}"
    )
    proposal_source = (
        "claude_code"
        if outcome.proposal.source == "claude_code"
        else outcome.proposal.source or "auto"
    )
    pred_rationale = (
        outcome.proposal.prediction.rationale
        if outcome.proposal.prediction is not None
        else None
    )
    title, voice = _voice_for(
        kind="router_config",
        mutations=[m.describe() for m in outcome.proposal.mutations],
        delta_overall=delta_overall,
        prediction_rationale=pred_rationale,
    )
    lesson = Lesson(
        id=lesson_id,
        version=str(new_version),
        kind="router_config",
        status="auto_promoted",
        title=title,
        summary=(
            f"Promoted router_config v{new_version} "
            f"(K={payload.get('k')}, "
            f"models={len(payload.get('model_psi') or {})}, "
            f"silhouette={delta_overall:.3f})."
        ),
        proposal_source=proposal_source,
        delta={"silhouette": delta_overall},
        mutations=[m.describe() for m in outcome.proposal.mutations],
        parent_version=str((new_version - 1) if new_version > 0 else 0),
        candidate_id=outcome.candidate_id or "",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice=voice,
    )
    write_lesson(lesson)

    return new_version, lesson.id


# ---------------------------------------------------------------------------
# P15.4.4 — Dataset candidate promotion
# ---------------------------------------------------------------------------


def apply_dataset_candidate(
    candidate_payload: dict,
    *,
    datasets_dir: Optional[Path] = None,
) -> Path:
    """Atomically write datasets/<name>/v<n>.json + flip the current pointer.

    Args:
        candidate_payload: Proposer's inline payload (from Mutation.value).
            Must contain ``version``, ``name``, ``samples``.
        datasets_dir: Override the default ``datasets/`` directory.

    Returns:
        Path to the v<n>.json that was written.
    """
    from router.data.dataset_io import save_dataset

    return save_dataset(
        candidate_payload,
        datasets_dir=datasets_dir,
        update_pointer=True,
    )


def promote_dataset(
    outcome: LoopOutcome,
    *,
    datasets_dir: Optional[Path] = None,
) -> tuple[int, str]:
    """Promote a dataset candidate. Returns ``(new_version, lesson_id)``.

    Mirrors ``promote_router_config`` for ``kind="dataset"`` proposals.
    The candidate payload lives inline on
    ``outcome.proposal.mutations[0].value``; no agent_dir snapshot is
    taken (datasets sit outside the agent's editable surface, same as
    router_config).
    """
    if not outcome.proposal.mutations:
        raise ValueError("promote_dataset requires a proposal with mutations")
    payload = outcome.proposal.mutations[0].value
    if not isinstance(payload, dict):
        raise ValueError(
            "promote_dataset expects dict payload on Mutation.value, "
            f"got {type(payload).__name__}"
        )

    json_path = apply_dataset_candidate(payload, datasets_dir=datasets_dir)
    new_version = int(payload["version"])
    name = str(payload["name"])

    promoted_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    metadata = payload.get("metadata") or {}
    added = int(metadata.get("added", 0))
    gap_score_before = metadata.get("gap_score_before")
    gap_score_after = metadata.get("gap_score_after")

    ledger_payload: dict = {
        "kind": "dataset",
        "name": name,
        "source": outcome.proposal.source,
        "json_path": str(json_path),
        "added": added,
        "total_size": len(payload.get("samples") or []),
        "gap_score_before": gap_score_before,
        "gap_score_after": gap_score_after,
        "metadata": metadata,
        "verdicts": [
            {"critic": v.critic, "approved": v.approved, "reason": v.reason}
            for v in outcome.verdicts
        ],
    }
    if outcome.proposal.prediction is not None:
        ledger_payload["prediction"] = {
            "rubric": outcome.proposal.prediction.rubric,
            "expected_delta": outcome.proposal.prediction.expected_delta,
            "rationale": outcome.proposal.prediction.rationale,
            "confidence": outcome.proposal.prediction.confidence,
        }

    entry = write_entry(
        kind="promote",
        candidate_id=outcome.candidate_id or "",
        agent_version_before=None,
        agent_version_after=None,
        summary=f"promoted dataset {name!r} v{new_version} (added={added})",
        payload=ledger_payload,
    )

    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}"
    )
    proposal_source = (
        "claude_code"
        if outcome.proposal.source == "claude_code"
        else outcome.proposal.source or "auto"
    )
    pred_rationale = (
        outcome.proposal.prediction.rationale
        if outcome.proposal.prediction is not None
        else None
    )
    title, voice = _voice_for(
        kind="dataset",
        mutations=[m.describe() for m in outcome.proposal.mutations],
        delta_overall=float(added),
        prediction_rationale=pred_rationale,
    )

    gap_text = ""
    if gap_score_before is not None and gap_score_after is not None:
        gap_text = f" (gap_score {float(gap_score_before):.2f} → {float(gap_score_after):.2f})"
    lesson_summary = (
        f"Promoted dataset {name!r} v{new_version} — "
        f"added {added} sample(s){gap_text}."
    )

    lesson = Lesson(
        id=lesson_id,
        version=str(new_version),
        kind="dataset",
        status="auto_promoted",
        title=title,
        summary=lesson_summary,
        proposal_source=proposal_source,
        delta={
            "added": float(added),
            "gap_score_before": float(gap_score_before) if gap_score_before is not None else 0.0,
            "gap_score_after": float(gap_score_after) if gap_score_after is not None else 0.0,
        },
        mutations=[m.describe() for m in outcome.proposal.mutations],
        parent_version=str((new_version - 1) if new_version > 0 else 0),
        candidate_id=outcome.candidate_id or "",
        promoted_at=promoted_at,
        ledger_entry_id=entry.entry_id,
        voice=voice,
    )
    write_lesson(lesson)

    return new_version, lesson.id
