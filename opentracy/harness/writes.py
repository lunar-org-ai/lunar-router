"""Critic-gated write paths for the harness.

Single shared core called by both the FastAPI routes
(`/v1/harness/run/...`, `/v1/harness/proposals/...`) and the MCP tools
in `mcp_server.py`. Every non-inspector write goes through
`critic_gate.critic_check` first.

Design notes:
- Inspector agents skip the critic. They are read-only by convention
  (see `agents/inspectors/`) — gating them would burn LLM calls without
  changing any outcome. Proposers and critics are gated.
- Proposals are stored as `type="proposal"` ledger rows. "Pending"
  status is derived by checking for a child decision entry tagged
  `proposal_resolved`. There is no separate proposals table — status is
  always read off the ledger.
- `record_outcome` does NOT call the critic. It records side-effects
  that already happened.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from .critic_gate import CriticVerdict, critic_check
from .ledger import LedgerEntry, LedgerStore, Outcome, get_ledger_store
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


ProposalStatus = Literal["pending", "approved", "rejected", "rejected_by_critic", "executed", "failed"]


class _RunnerProtocol(Protocol):
    async def run(self, agent_name: str, user_input: str) -> Any: ...
    async def run_with_tools(self, agent_name: str, user_input: str) -> Any: ...


def _agent_role(agent_name: str, registry: Optional[AgentRegistry] = None) -> str:
    """Return 'inspectors' | 'proposers' | 'critics' | 'unknown' based on
    the file path of the agent .md. Unknown agents are treated as
    non-inspector (gated) — failing closed is the safe default."""
    reg = registry or AgentRegistry()
    config = reg.get(agent_name)
    if config is None or not config.file_path:
        return "unknown"
    parts = Path(config.file_path).parts
    for role in ("inspectors", "proposers", "critics", "narrators"):
        if role in parts:
            return role
    return "unknown"


def _ledger_or_default(ledger: Optional[LedgerStore]) -> LedgerStore:
    return ledger if ledger is not None else get_ledger_store()


# ---------------------------------------------------------------------------
# run_agent_gated
# ---------------------------------------------------------------------------


async def run_agent_gated(
    agent_name: str,
    user_input: str,
    *,
    objective_id: Optional[str] = None,
    use_tools: bool = False,
    context: Optional[dict[str, Any]] = None,
    runner: Optional[_RunnerProtocol] = None,
    registry: Optional[AgentRegistry] = None,
    ledger: Optional[LedgerStore] = None,
) -> dict[str, Any]:
    """Run an agent, gated by the budget critic for non-inspector roles.

    Returns one of:
        {"decision": "approved", "result": <agent data>, "run_entry_id": id, "decision_entry_id": id}
        {"decision": "rejected", "rationale": ..., "decision_entry_id": id}
        {"decision": "ungated", "result": <agent data>, "run_entry_id": id}  # inspectors

    Inspectors bypass the critic; their results still get written as a
    `run` ledger row so the audit trail covers every agent invocation.
    """
    store = _ledger_or_default(ledger)
    role = _agent_role(agent_name, registry=registry)

    if runner is None:
        from .runner import AgentRunner

        runner = AgentRunner()

    parent_id: Optional[str] = None
    decision_entry_id: Optional[str] = None
    decision: str = "ungated"

    if role != "inspectors":
        verdict = await critic_check(
            action_kind="run_agent",
            payload={"name": agent_name, "input": user_input, "use_tools": use_tools},
            objective_id=objective_id,
            ledger=store,
            runner=runner,
        )
        decision_entry_id = verdict.decision_entry_id
        decision = "approved" if verdict.approved else "rejected"
        parent_id = verdict.decision_entry_id
        if not verdict.approved:
            return {
                "decision": "rejected",
                "rationale": verdict.rationale,
                "estimated_cost_usd": verdict.estimated_cost_usd,
                "decision_entry_id": decision_entry_id,
            }

    # Approved (or inspector) — run the agent.
    if use_tools:
        result = await runner.run_with_tools(agent_name, user_input)
    else:
        result = await runner.run(agent_name, user_input, context) if context else await runner.run(agent_name, user_input)

    result_data = getattr(result, "data", result if isinstance(result, dict) else {"value": result})

    run_entry = LedgerEntry(
        type="run",
        objective_id=objective_id,
        agent=agent_name,
        parent_id=parent_id,
        data=result_data if isinstance(result_data, dict) else {"value": result_data},
        tags=["agent_run", agent_name, role],
        cost_usd=getattr(result, "cost_usd", None),
        duration_ms=int(getattr(result, "duration_ms", 0)) or None,
        outcome="ok",
    )
    store.append(run_entry)

    out: dict[str, Any] = {
        "decision": decision,
        "result": result_data,
        "run_entry_id": run_entry.id,
    }
    if decision_entry_id:
        out["decision_entry_id"] = decision_entry_id
    return out


# ---------------------------------------------------------------------------
# Proposal lifecycle
# ---------------------------------------------------------------------------


async def propose_action(
    *,
    kind: str,
    payload: dict[str, Any],
    objective_id: Optional[str],
    summary: str,
    runner: Optional[_RunnerProtocol] = None,
    ledger: Optional[LedgerStore] = None,
) -> dict[str, Any]:
    """Create a proposal. The critic runs FIRST; the proposal carries
    the verdict so the operator UI can show it without re-fetching."""
    store = _ledger_or_default(ledger)

    verdict = await critic_check(
        action_kind=kind,
        payload=payload,
        objective_id=objective_id,
        ledger=store,
        runner=runner,
    )

    proposal = LedgerEntry(
        type="proposal",
        objective_id=objective_id,
        agent="harness_caller",
        parent_id=verdict.decision_entry_id,
        data={
            "kind": kind,
            "payload": payload,
            "summary": summary,
            "estimated_cost_usd": verdict.estimated_cost_usd,
            "estimated_benefit": verdict.estimated_benefit,
            "verdict": verdict.to_dict(),
        },
        tags=[
            "proposal",
            kind,
            f"critic_{verdict.decision}",
        ],
        cost_usd=verdict.estimated_cost_usd,
        outcome=None,
    )
    store.append(proposal)

    return {
        "proposal_id": proposal.id,
        "decision_entry_id": verdict.decision_entry_id,
        "verdict": verdict.to_dict(),
        "summary": summary,
        "kind": kind,
    }


def _resolution_for(proposal_id: str, store: LedgerStore) -> Optional[LedgerEntry]:
    """Find the `proposal_resolved` decision child of a proposal, if any.

    Walks one level — proposals are resolved by exactly one direct child
    decision in the current model.
    """
    chain = store.chain(proposal_id)
    for entry in chain:
        if entry.id == proposal_id:
            continue
        if entry.parent_id == proposal_id and entry.type == "decision" and "proposal_resolved" in entry.tags:
            return entry
    return None


def _outcome_for(proposal_id: str, store: LedgerStore) -> Optional[LedgerEntry]:
    """Find the most recent `proposal_outcome` action under a proposal."""
    chain = store.chain(proposal_id)
    outcome: Optional[LedgerEntry] = None
    for entry in chain:
        if entry.type == "action" and "proposal_outcome" in entry.tags:
            outcome = entry  # latest wins (chain is ts-ordered)
    return outcome


def _proposal_status(proposal: LedgerEntry, store: LedgerStore) -> ProposalStatus:
    """Derive the lifecycle status of a proposal from its descendants."""
    verdict = (proposal.data or {}).get("verdict") or {}
    if verdict.get("decision") == "reject":
        return "rejected_by_critic"

    resolution = _resolution_for(proposal.id, store)
    if resolution is not None:
        if "rejected" in resolution.tags:
            return "rejected"
        if "approved" in resolution.tags:
            outcome = _outcome_for(proposal.id, store)
            if outcome is None:
                return "approved"
            return "executed" if outcome.outcome == "ok" else "failed"

    return "pending"


async def approve_proposal(
    proposal_id: str,
    *,
    runner: Optional[_RunnerProtocol] = None,
    ledger: Optional[LedgerStore] = None,
) -> dict[str, Any]:
    """Approve a proposal. Re-runs the critic — guards against the case
    where the objective has recovered between propose and approve, so
    the action is no longer worth its cost. Cache normally hits."""
    store = _ledger_or_default(ledger)

    proposal = store.get(proposal_id)
    if proposal is None or proposal.type != "proposal":
        raise ValueError(f"Proposal '{proposal_id}' not found")

    if _resolution_for(proposal_id, store) is not None:
        raise ValueError(f"Proposal '{proposal_id}' already resolved")

    payload = proposal.data.get("payload") or {}
    kind = proposal.data.get("kind") or "unknown"

    verdict = await critic_check(
        action_kind=kind,
        payload=payload,
        objective_id=proposal.objective_id,
        ledger=store,
        runner=runner,
    )

    if not verdict.approved:
        decision = LedgerEntry(
            type="decision",
            objective_id=proposal.objective_id,
            parent_id=proposal_id,
            agent="approver",
            data={
                "decision": "rejected",
                "reason": "critic_recheck_rejected",
                "verdict": verdict.to_dict(),
            },
            tags=["proposal_resolved", "rejected", "critic_recheck"],
            outcome="ok",
        )
        store.append(decision)
        return {
            "proposal_id": proposal_id,
            "decision": "rejected",
            "verdict": verdict.to_dict(),
            "decision_entry_id": decision.id,
        }

    decision = LedgerEntry(
        type="decision",
        objective_id=proposal.objective_id,
        parent_id=proposal_id,
        agent="approver",
        data={"decision": "approved", "verdict": verdict.to_dict()},
        tags=["proposal_resolved", "approved"],
        outcome="ok",
    )
    store.append(decision)

    return {
        "proposal_id": proposal_id,
        "decision": "approved",
        "verdict": verdict.to_dict(),
        "decision_entry_id": decision.id,
    }


def reject_proposal(
    proposal_id: str,
    *,
    reason: str = "",
    ledger: Optional[LedgerStore] = None,
) -> dict[str, Any]:
    """Reject a proposal. No critic re-check — manual rejection is
    always permitted."""
    store = _ledger_or_default(ledger)

    proposal = store.get(proposal_id)
    if proposal is None or proposal.type != "proposal":
        raise ValueError(f"Proposal '{proposal_id}' not found")

    if _resolution_for(proposal_id, store) is not None:
        raise ValueError(f"Proposal '{proposal_id}' already resolved")

    decision = LedgerEntry(
        type="decision",
        objective_id=proposal.objective_id,
        parent_id=proposal_id,
        agent="approver",
        data={"decision": "rejected", "reason": reason},
        tags=["proposal_resolved", "rejected", "manual"],
        outcome="ok",
    )
    store.append(decision)

    return {
        "proposal_id": proposal_id,
        "decision": "rejected",
        "decision_entry_id": decision.id,
    }


def record_outcome(
    proposal_id: str,
    *,
    result: dict[str, Any],
    outcome: Outcome,
    ledger: Optional[LedgerStore] = None,
) -> dict[str, Any]:
    """Record the outcome of executing a proposal. Recording observed
    reality — no critic check."""
    store = _ledger_or_default(ledger)

    proposal = store.get(proposal_id)
    if proposal is None or proposal.type != "proposal":
        raise ValueError(f"Proposal '{proposal_id}' not found")

    cost = result.get("cost_usd")
    try:
        cost_val = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_val = None

    action = LedgerEntry(
        type="action",
        objective_id=proposal.objective_id,
        parent_id=proposal_id,
        agent="executor",
        data={"result": result},
        tags=["proposal_outcome", outcome],
        outcome=outcome,
        cost_usd=cost_val,
    )
    store.append(action)

    return {
        "proposal_id": proposal_id,
        "outcome": outcome,
        "action_entry_id": action.id,
    }


# ---------------------------------------------------------------------------
# Listing / status
# ---------------------------------------------------------------------------


def list_proposals(
    *,
    status: Optional[ProposalStatus] = None,
    objective_id: Optional[str] = None,
    limit: int = 100,
    ledger: Optional[LedgerStore] = None,
) -> list[dict[str, Any]]:
    """List proposals newest-first with optional status filter.

    Status is computed per-row from the ledger; no separate index. For
    high volumes this is O(N×fanout), acceptable at Phase 1.5 scale.
    """
    store = _ledger_or_default(ledger)
    capped = max(1, min(limit, 1000))

    raw = store.recent(limit=capped * 10)
    out: list[dict[str, Any]] = []
    for entry in raw:
        if entry.type != "proposal":
            continue
        if objective_id and entry.objective_id != objective_id:
            continue
        computed_status = _proposal_status(entry, store)
        if status and computed_status != status:
            continue
        out.append(_proposal_to_dict(entry, computed_status))
        if len(out) >= capped:
            break
    return out


def get_proposal(
    proposal_id: str,
    *,
    ledger: Optional[LedgerStore] = None,
) -> Optional[dict[str, Any]]:
    """Fetch a single proposal with computed status. Returns None for
    missing or non-proposal ids."""
    store = _ledger_or_default(ledger)
    entry = store.get(proposal_id)
    if entry is None or entry.type != "proposal":
        return None
    return _proposal_to_dict(entry, _proposal_status(entry, store))


def _proposal_to_dict(entry: LedgerEntry, status: ProposalStatus) -> dict[str, Any]:
    base = entry.model_dump()
    base["status"] = status
    return base
