"""OpenTracy harness MCP server — Phase 1.5 of the harness redesign.

Exposes the harness's read surface to any MCP client (Claude Code
being the primary target). Two transports from one codebase:

  - **stdio** (`python -m opentracy.harness.mcp_server`): for
    contributors running locally. Claude Code spawns this as a
    subprocess via `claude mcp add opentracy-harness python -m
    opentracy.harness.mcp_server`.

  - **HTTP** (mounted at `/mcp` on the FastAPI app): for production
    operations, where the operator connects Claude Code to a deployed
    instance via `claude mcp add --transport http http://host/mcp/`.
    The same `build_server()` output drives both.

Design rule: every tool is a thin wrapper over existing harness
primitives. No behavior lives here. When Phase 2 auto-research arrives,
it consumes the same tool surface — so any logic added here would have
to be duplicated there.

Scope: read-only. Write operations (enable/disable policy, trigger
recipe manually, acknowledge signal) require runtime-mutable state in
the harness that Phase 1.5 deliberately defers. See `docs/harness_redesign.md`.
"""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "opentracy-harness"
SERVER_INSTRUCTIONS = (
    "Read + write access to the OpenTracy harness: objectives, ledger, "
    "policies, recipes, actions, agents. Use `get_ledger_chain` for "
    "causal drill-down from any signal id. Use "
    "`get_objective_time_series` to see how an objective has moved.\n\n"
    "Write tools (`run_agent`, `propose_action`, `approve_proposal`, "
    "`reject_proposal`, `record_outcome`) are gated through the "
    "`budget_justifier` critic — every approved write also writes a "
    "decision row to the ledger so the audit trail is intact. Inspector "
    "agents skip the gate (read-only by convention)."
)


def build_server() -> FastMCP:
    """Construct the MCP server with all tools (read + write) registered.

    Kept as a function (not module-level execution) so tests and both
    transport entry points use the exact same server configuration.
    """
    # `streamable_http_path="/"` is important: FastMCP defaults this to
    # `/mcp`, which combined with the FastAPI `app.mount("/mcp", ...)`
    # would produce an external URL of `/mcp/mcp`. Override to `/` so
    # clients can connect at the expected `http://host/mcp/`.
    server = FastMCP(
        SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
        streamable_http_path="/",
    )
    _register_objective_tools(server)
    _register_ledger_tools(server)
    _register_catalog_tools(server)
    _register_write_tools(server)
    return server


# ---------------------------------------------------------------------------
# Objective tools
# ---------------------------------------------------------------------------


def _register_objective_tools(server: FastMCP) -> None:
    @server.tool()
    def list_objectives() -> list[dict]:
        """List all user-declared objectives with YAML definitions."""
        from opentracy.harness.objectives.loader import load_all

        return [o.model_dump() for o in load_all()]

    @server.tool()
    def get_objective_time_series(
        objective_id: str,
        hours: int = 168,
    ) -> dict:
        """Return the measurement time-series + action markers for an
        objective within a trailing window. Identical shape to the
        `/v1/harness/objectives/{id}/time-series` HTTP endpoint so
        auto-research can consume either path.
        """
        from datetime import datetime, timedelta, timezone

        from opentracy.harness.ledger import get_ledger_store

        hours_capped = max(1, min(int(hours), 24 * 30))
        start_iso = (
            datetime.now(timezone.utc) - timedelta(hours=hours_capped)
        ).isoformat()

        store = get_ledger_store()
        entries = store.time_series(objective_id, start=start_iso, limit=5000)

        measurements = []
        markers = []
        for entry in entries:
            if entry.type == "observation" and "measurement" in entry.tags:
                value = entry.data.get("value")
                if value is None:
                    continue
                measurements.append({
                    "ts": entry.ts,
                    "value": value,
                    "sample_size": entry.data.get("sample_size"),
                    "id": entry.id,
                })
            elif entry.type in {"signal", "decision", "action"}:
                markers.append(entry.model_dump())

        return {
            "objective_id": objective_id,
            "window_hours": hours_capped,
            "measurements": measurements,
            "markers": markers,
        }


# ---------------------------------------------------------------------------
# Ledger tools
# ---------------------------------------------------------------------------


def _register_ledger_tools(server: FastMCP) -> None:
    @server.tool()
    def list_ledger_entries(
        type: Optional[str] = None,
        objective_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        """List recent ledger entries newest-first, with AND-combined
        filters. `limit` is capped at 1000."""
        from opentracy.harness.ledger import get_ledger_store

        capped = max(1, min(int(limit), 1000))
        store = get_ledger_store()
        raw = store.recent(limit=capped * 10)
        filtered = []
        for e in raw:
            if type and e.type != type:
                continue
            if objective_id and e.objective_id != objective_id:
                continue
            if agent and e.agent != agent:
                continue
            filtered.append(e)
            if len(filtered) >= capped:
                break
        return {
            "entries": [e.model_dump() for e in filtered],
            "count": len(filtered),
        }

    @server.tool()
    def get_ledger_entry(entry_id: str) -> Optional[dict]:
        """Fetch a single ledger entry by id. Returns null when the id
        is unknown — caller should treat as a missing entry, not an
        error."""
        from opentracy.harness.ledger import get_ledger_store

        store = get_ledger_store()
        entry = store.get(entry_id)
        return entry.model_dump() if entry else None

    @server.tool()
    def get_ledger_chain(root_entry_id: str) -> dict:
        """Return the full BFS causal chain rooted at `root_entry_id`.
        Primary drill-down tool: given a signal, returns every run,
        observation, decision, and action that descended from it."""
        from opentracy.harness.ledger import get_ledger_store

        store = get_ledger_store()
        root = store.get(root_entry_id)
        if root is None:
            return {"root_id": root_entry_id, "entries": [], "count": 0}
        chain = store.chain(root_entry_id)
        return {
            "root_id": root_entry_id,
            "entries": [e.model_dump() for e in chain],
            "count": len(chain),
        }


# ---------------------------------------------------------------------------
# Catalog tools (policies / recipes / actions / agents)
# ---------------------------------------------------------------------------


def _register_catalog_tools(server: FastMCP) -> None:
    @server.tool()
    def list_policies() -> list[dict]:
        """List every policy YAML loaded from
        `opentracy/harness/triggers/definitions/`."""
        from opentracy.harness.triggers.policies import load_policies

        return [p.model_dump() for p in load_policies()]

    @server.tool()
    def describe_policy(policy_id: str) -> Optional[dict]:
        """Full Policy model for one id. Returns null when unknown."""
        from opentracy.harness.triggers.policies import load_policy

        p = load_policy(policy_id)
        return p.model_dump() if p else None

    @server.tool()
    def list_recipes() -> list[dict]:
        """List every recipe YAML from
        `opentracy/harness/tasks/recipes/`."""
        from opentracy.harness.tasks import load_recipes

        return [r.model_dump() for r in load_recipes()]

    @server.tool()
    def describe_recipe(recipe_id: str) -> Optional[dict]:
        """Full Recipe model for one id. Returns null when unknown."""
        from opentracy.harness.tasks import load_recipe

        r = load_recipe(recipe_id)
        return r.model_dump() if r else None

    @server.tool()
    def list_actions() -> list[str]:
        """Names of every code action registered via `@register_action`.
        Recipes reference actions by these names."""
        from opentracy.harness.actions import list_actions as _la

        return _la()

    @server.tool()
    def list_agents() -> list[dict]:
        """Every harness agent .md file, with name/description/model/
        temperature. The heavy fields (system_prompt) are elided here
        — use the `/v1/harness/agents/{name}` HTTP route for the full
        prompt text."""
        from opentracy.harness.registry import AgentRegistry

        registry = AgentRegistry()
        out = []
        for agent in registry.list_agents():
            out.append({
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "output_schema": {
                    "type": agent.output_schema.type,
                    "fields": agent.output_schema.fields,
                },
            })
        return out


# ---------------------------------------------------------------------------
# Write tools (Phase 1.5: critic-gated mutations)
# ---------------------------------------------------------------------------


def _register_write_tools(server: FastMCP) -> None:
    @server.tool()
    async def run_agent(
        name: str,
        input: str,
        objective_id: Optional[str] = None,
        use_tools: bool = False,
    ) -> dict:
        """Run a harness agent with `input`. Inspector agents execute
        directly; proposers and critics are gated through the
        `budget_justifier` critic, which writes a decision row to the
        ledger before allowing the run. A rejected verdict skips the
        run and returns the rationale. Pass `objective_id` so the
        critic can read the relevant trend."""
        from opentracy.harness import writes as harness_writes

        return await harness_writes.run_agent_gated(
            name,
            input,
            objective_id=objective_id,
            use_tools=use_tools,
        )

    @server.tool()
    async def propose_action(
        kind: str,
        payload: dict,
        summary: str = "",
        objective_id: Optional[str] = None,
    ) -> dict:
        """Create a proposal for an action that should run later. The
        critic runs immediately and the verdict is stored on the
        proposal so an operator (or Phase 2 auto-research) can approve
        or reject without re-asking the critic. Returns the new
        proposal_id and the verdict."""
        from opentracy.harness import writes as harness_writes

        return await harness_writes.propose_action(
            kind=kind,
            payload=payload or {},
            objective_id=objective_id,
            summary=summary,
        )

    @server.tool()
    async def approve_proposal(proposal_id: str) -> dict:
        """Approve a pending proposal. Re-runs the critic — guards
        against the objective recovering between propose and approve.
        On reject, the proposal is closed without execution."""
        from opentracy.harness import writes as harness_writes

        return await harness_writes.approve_proposal(proposal_id)

    @server.tool()
    def reject_proposal(proposal_id: str, reason: str = "") -> dict:
        """Reject a proposal. No critic re-check — manual rejection is
        always permitted."""
        from opentracy.harness import writes as harness_writes

        return harness_writes.reject_proposal(proposal_id, reason=reason)

    @server.tool()
    def record_outcome(
        proposal_id: str,
        result: dict,
        outcome: str,
    ) -> dict:
        """Record the outcome of an approved proposal that was actually
        executed. `outcome` must be one of: ok, failed, rolled_back,
        skipped. No critic check — this records reality."""
        from opentracy.harness import writes as harness_writes

        if outcome not in {"ok", "failed", "rolled_back", "skipped"}:
            raise ValueError(
                f"outcome must be ok|failed|rolled_back|skipped, got {outcome!r}"
            )
        return harness_writes.record_outcome(
            proposal_id,
            result=result or {},
            outcome=outcome,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main() -> None:
    """Stdio transport — used when Claude Code spawns the module as a
    subprocess via `python -m opentracy.harness.mcp_server`."""
    server = build_server()
    server.run("stdio")


if __name__ == "__main__":
    main()
