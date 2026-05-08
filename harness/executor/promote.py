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
from harness.types import LoopOutcome
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


def _kind_from_mutations(mutations: list[str]) -> str:
    files = {m.split(":")[0] for m in mutations}
    if any("retrieve" in f for f in files):
        return "rag"
    if any("rerank" in f for f in files):
        return "rerank"
    if any("route" in f for f in files):
        return "router"
    if any("generate" in f or "prompts/" in f for f in files):
        return "prompt"
    if any("memory" in f for f in files):
        return "memory"
    return "other"


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
