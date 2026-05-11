"""MCP server exposing introspection tools.

Run via:
  uv run python -m harness.introspection.mcp_server     # stdio for Claude Code

Registered in .mcp.json at the repo root so Claude Code (or any MCP client)
discovers it automatically when you open this project. The same server is
intended to be reused in production by the runtime's /v1/introspect endpoint
(via Anthropic SDK with MCP client).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from harness.introspection import lib

server: Server = Server("opentracy-introspection")


# ---------- tool registrations ----------


@server.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_recent_promotions",
            description=(
                "List recent agent promotions (a candidate becoming live). "
                "Each item carries mutations, Δoverall vs baseline, and — when the "
                "proposer made a falsifiable claim — the prediction + verification "
                "verdict. Use this for 'what changed?' or 'what got promoted?' questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "since_iso": {
                        "type": "string",
                        "description": "ISO 8601 lower bound (e.g. '2026-05-06T00:00:00Z'). Empty = no bound.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max items to return (default 20).",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="list_recent_rollbacks",
            description=(
                "List recent rollbacks (live agent reverted to a prior version). "
                "Useful to answer 'why did we revert?' or 'what regressed?'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "since_iso": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="get_lesson",
            description=(
                "Fetch one approved Lesson by id. Lessons are the user-visible cards "
                "tied to each promotion: title, summary, voice, kind, mutations, delta."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lesson_id": {"type": "string"},
                },
                "required": ["lesson_id"],
            },
        ),
        Tool(
            name="get_day_epoch",
            description=(
                "Read the distilled day-epoch — counts (proposals/promotions/rollbacks), "
                "top events, references. Use for 'what happened on YYYY-MM-DD?'. "
                "If not yet distilled, will distill on-demand."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "YYYY-MM-DD",
                    },
                },
                "required": ["date"],
            },
        ),
        Tool(
            name="list_predictions",
            description=(
                "Find promotions whose proposer made a falsifiable claim, paired with "
                "the actual outcome (verification verdict: verified | partial | wrong | "
                "no_change). Use for 'how often are predictions correct?' or "
                "'show me wrong predictions'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "description": "Filter: verified | partial | wrong | no_change. Empty = all.",
                    },
                    "limit": {"type": "integer", "default": 50},
                },
            },
        ),
        Tool(
            name="list_available_epochs",
            description=(
                "Discover which days and versions have distilled epochs available. "
                "Use this before calling get_day_epoch / get_version_epoch."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="router_health_check",
            description=(
                "Read-only snapshot of the router's current state (P15.3). "
                "Returns cold_start flag, K, model_count, cost_weight, "
                "trace_count_since_last_fit, drift_score, last_fit_age_hours, "
                "current_avg_error, current_win_rate, needs_reclustering, "
                "cluster_distribution, fitted_from. Use this BEFORE deciding "
                "whether to call propose_router_retrain."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="propose_router_retrain",
            description=(
                "Trigger a router_config retrain (P15.3). Runs proposer → "
                "critic → approver → executor → ledger. Returns the action "
                "taken (promoted/queued/rejected/blocked) + the Lesson ID. "
                "Pass a short rationale explaining why a retrain is "
                "warranted. Gated by Policy.overrides['router_config']."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "Why a retrain is needed (1-2 sentences).",
                    },
                },
            },
        ),
        Tool(
            name="dataset_health_check",
            description=(
                "Read-only snapshot of one or all datasets (P15.4). Returns "
                "per-dataset: name, size, source, sourceType, use, owner, "
                "growing, version, last_curation_at, gap_score, "
                "cluster_distribution, adapter_available. Pass `name` to "
                "scope to a single dataset; omit for all. Use BEFORE deciding "
                "whether to call propose_dataset_curation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Specific dataset; omit for all.",
                    },
                },
            },
        ),
        Tool(
            name="propose_dataset_curation",
            description=(
                "Trigger a dataset curation cycle (P15.4). Runs "
                "DatasetProposer → DatasetCritic → policy branch → "
                "promote_dataset. Returns the action (promoted/queued/"
                "rejected/blocked) + Lesson ID. Optionally override the "
                "mining adapter with `source`. Pass a rationale. Gated by "
                "Policy.overrides['dataset']."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Registered dataset name.",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Optional adapter override: 'flagged traces', "
                            "'language router', 'failed lookups'."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why curation is needed (1-2 sentences).",
                    },
                },
                "required": ["name"],
            },
        ),
    ]


# ---------- dispatcher ----------


_HANDLERS = {
    "list_recent_promotions": lib.list_recent_promotions,
    "list_recent_rollbacks": lib.list_recent_rollbacks,
    "get_lesson": lib.get_lesson,
    "get_day_epoch": lib.get_day_epoch,
    "list_predictions": lib.list_predictions,
    "list_available_epochs": lib.list_available_epochs,
    "router_health_check": lib.router_health_check,
    "propose_router_retrain": lib.propose_router_retrain,
    "dataset_health_check": lib.dataset_health_check,
    "propose_dataset_curation": lib.propose_dataset_curation,
}


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    handler = _HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"unknown tool: {name!r}")]
    try:
        # Strip empty-string args so callable defaults kick in
        cleaned = {k: v for k, v in (arguments or {}).items() if v != "" or k == "lesson_id"}
        result = handler(**cleaned)
    except Exception as e:
        return [TextContent(type="text", text=f"error: {type(e).__name__}: {e}")]
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ---------- entry point ----------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
