"""Executor — promote an approved candidate to live agent/.

Steps:
  1. Snapshot current live agent → ledger/versions/<old_version>/
  2. Copy candidate's agent tree → live agent/
  3. Bump version in live agent.yaml (patch +1)
  4. Append LedgerEntry (kind=promote) tying old/new versions to the candidate.
  5. Write a Lesson (UI card) summarizing the change + delta.
"""

from __future__ import annotations

import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from experiments.branching import candidate_agent_path
from harness.types import LoopOutcome
from ledger.types import Lesson
from ledger.versioning import LIVE_AGENT, read_version, snapshot_agent
from ledger.writer import write_entry, write_lesson


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


def _make_lesson(
    outcome: LoopOutcome,
    new_version: str,
    parent_version: str,
    entry_id: str,
) -> Lesson:
    mutations = [m.describe() for m in outcome.proposal.mutations]
    kind = _kind_from_mutations(mutations)
    lesson_id = (
        f"L-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    )
    delta = outcome.candidate_result.delta if outcome.candidate_result else {}
    delta_overall = delta.get("overall_score", 0.0) if delta else 0.0
    title = mutations[0] if len(mutations) == 1 else f"{len(mutations)} {kind} mutations"
    summary = (
        f"Candidate scored Δoverall={delta_overall:+.4f} vs baseline ({parent_version}); "
        f"all critics passed."
    )
    voice = (
        f"I tweaked {kind}: {', '.join(mutations)}. "
        f"Δoverall on the suite came out to {delta_overall:+.4f}, so I promoted it."
    )

    return Lesson(
        id=lesson_id,
        version=new_version,
        kind=kind,
        status="auto_promoted",
        title=title,
        summary=summary,
        proposal_source=outcome.proposal.source,
        delta=delta,
        mutations=mutations,
        parent_version=parent_version,
        candidate_id=outcome.candidate_id or "",
        promoted_at=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
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
    entry = write_entry(
        kind="promote",
        candidate_id=outcome.candidate_id,
        agent_version_before=old_version,
        agent_version_after=new_version,
        summary=f"promoted with Δoverall={delta_overall:+.4f}",
        payload={
            "mutations": [m.describe() for m in outcome.proposal.mutations],
            "delta": delta,
            "verdicts": [
                {"critic": v.critic, "approved": v.approved, "reason": v.reason}
                for v in outcome.verdicts
            ],
        },
    )

    # 5. Lesson
    lesson = _make_lesson(outcome, new_version, old_version, entry.entry_id)
    write_lesson(lesson)

    return new_version, lesson.id
