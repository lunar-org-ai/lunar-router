"""Edit phase — spawn an Evolve Agent sandbox that mutates the harness.

The Evolve Agent is just ``claude --print`` again, but with a
system prompt that puts it in evolution mode: read the NexAU
snapshot, read the distilled evidence + manifest history, then edit
one or more harness files and write a fresh pending Change Manifest.

We reuse :class:`runtime.sandbox.SandboxRun` so the sandbox layer
stays single-purpose — the only thing that differs from the per-turn
engineer call is the system prompt + a different stdout consumer
(we don't stream the evolve output to a user; we collect it for the
caller).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from runtime.evolution.types import EvolveOutcome


logger = logging.getLogger("runtime.evolution.evolve")


_EVOLVE_PROMPT = """Improve this agent based on rollout evidence."""


def _build_system_prompt(
    *,
    agent_id: str,
    base_system: str,
    nexau: dict[str, list[str]],
    manifest_history: list[dict[str, Any]],
    evidence_summary: str,
) -> str:
    """Compose the evolve-mode system prompt.

    Differs from the per-turn engineer prompt by:
      - Reframing the role as "evolution agent for this harness"
      - Forbidding chat-shaped replies (no user is waiting)
      - Mandating a pending manifest with prediction + at-risk regressions,
        so Decision Observability (AHE §3.3) actually works next round
    """
    nexau_lines = []
    for key in ("system_prompt", "tools", "middleware", "skills", "subagents", "memory"):
        items = nexau.get(key, [])
        nexau_lines.append(
            f"- {key}: {', '.join(items) if items else '(empty — minimal-seed)'}"
        )
    history_lines = []
    if manifest_history:
        for entry in manifest_history[:5]:
            outcome = (entry.get("outcome") or {}).get("verdict", "(no verdict)")
            fixes = ", ".join(entry.get("claimed_fixes", []) or ["?"])
            history_lines.append(f"- {outcome}: {fixes}")
    else:
        history_lines.append("(no manifest history yet)")

    return (
        f"You are the EVOLUTION agent for agent `{agent_id}` (not the\n"
        f"customer-facing engineer). No human is waiting on you — your\n"
        f"job is to improve this harness for the NEXT rollout.\n\n"
        f"{base_system.rstrip()}\n\n"
        f"--- harness snapshot -----------------------------------------\n\n"
        f"## NexAU components (file at .opentracy/<type>/...)\n"
        f"{chr(10).join(nexau_lines)}\n\n"
        f"## Recent Change Manifest verdicts\n"
        f"{chr(10).join(history_lines)}\n\n"
        f"--- rollout evidence -----------------------------------------\n\n"
        f"{evidence_summary}\n\n"
        f"--- your task ------------------------------------------------\n\n"
        f"Based on the failures above (and what worked), edit ONE OR\n"
        f"MORE NexAU files to improve future rollouts. Typical edits:\n"
        f"  - tighten `.opentracy/system_prompt.md`\n"
        f"  - add a reusable strategy to `.opentracy/skills/<name>.md`\n"
        f"  - add a tool description+impl to `.opentracy/tools/<name>.{{json,sh}}`\n"
        f"  - add middleware in `.opentracy/middleware/<name>.{{py,sh}}`\n\n"
        f"BEFORE finishing, you MUST write a Change Manifest to\n"
        f"`.opentracy/manifest/pending.json` with these keys:\n"
        f"  - changed_files: [relative paths you edited]\n"
        f"  - rationale: short prose, why these edits\n"
        f"  - claimed_fixes: [task patterns you expect to flip pass]\n"
        f"  - at_risk_regressions: [task patterns that might newly fail]\n\n"
        f"Be minimal. Each edit must be justifiable by specific evidence.\n"
        f"If the rollout already passes everything cleanly, write a manifest\n"
        f"with empty `changed_files` and `rationale='nothing to improve'`.\n\n"
        f"Reply with one short paragraph summarizing what you changed."
    )


def run_evolve(
    *,
    workspace: Any,           # WorkspaceStore
    anthropic_key: str,
    agent_id: str,
    evidence_summary: str,
    sandbox_factory: Optional[Any] = None,
    timeout_s: int = 300,
    model: Optional[str] = None,
) -> EvolveOutcome:
    """Spawn an Evolve sandbox, let claude edit the workspace, snapshot back.

    ``sandbox_factory`` is the ``SandboxRun`` class (overridable for
    tests). Returns an :class:`EvolveOutcome` summarizing what changed
    + the pending manifest that the agent committed (if any).
    """
    from runtime.sandbox import SandboxRun as _DefaultSandbox

    SandboxRun = sandbox_factory or _DefaultSandbox

    base_system = workspace.read_system_prompt()
    nexau = workspace.list_nexau_components()
    history = workspace.list_manifest_history(limit=5)
    files_before = set(workspace.list_files(max_files=10_000))

    system_prompt = _build_system_prompt(
        agent_id=agent_id,
        base_system=base_system,
        nexau=nexau,
        manifest_history=history,
        evidence_summary=evidence_summary,
    )

    tar_in = workspace.to_tar_bytes()
    response_chunks: list[str] = []

    with SandboxRun(
        anthropic_key=anthropic_key,
        timeout_s=timeout_s,
    ) as sb:
        sb.upload_workspace_tar(tar_in)
        for evt in sb.run_claude(
            _EVOLVE_PROMPT,
            system=system_prompt,
            model=model,
        ):
            kind = evt.get("type")
            if kind == "stdout":
                response_chunks.append(evt.get("data") or "")
            elif kind == "stderr":
                logger.info("evolve sandbox stderr: %s", evt.get("data"))
            elif kind == "error":
                logger.warning("evolve sandbox error: %s", evt.get("detail"))
                break
            elif kind == "done":
                break

        try:
            tar_out = sb.snapshot_workspace_tar()
            workspace.from_tar_bytes(tar_out)
        except Exception as exc:
            logger.warning("evolve workspace snapshot back failed: %s", exc)

    raw_response = "".join(response_chunks).strip()
    pending = workspace.read_pending_manifest()
    files_after = set(workspace.list_files(max_files=10_000))
    edited = sorted(
        (files_after - files_before)
        | _changed_files(workspace, files_before & files_after)
    )

    return EvolveOutcome(
        files_edited=edited,
        pending_manifest=pending,
        raw_response=raw_response,
    )


def _changed_files(workspace: Any, candidates: set[str]) -> set[str]:
    """Best-effort detection of edited files among ones that existed
    both before and after. v0 just returns the candidates — we don't
    snapshot content hashes pre-edit. Future work: cache pre-edit
    hashes and diff. The pending manifest's ``changed_files`` (written
    by the agent) is the authoritative list for AHE accountability."""
    return set()
