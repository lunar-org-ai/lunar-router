"""Shared tool registry for the MCP introspection server (P16.2).

The same 10 tools are exposed via two transports:

  - **stdio** (OSS local) — ``harness/introspection/mcp_server.py`` runs
    in a subprocess that Claude Code spawns from ``.mcp.json``.
  - **HTTP/SSE** (hosted) — ``runtime/mcp/http.py`` mounts the
    Streamable HTTP + SSE endpoints on the runtime so customer Claude
    Code CLIs can connect remotely with a per-tenant Bearer.

Tool definitions + handlers live here so both transports stay in
sync. The handlers themselves come from :mod:`harness.introspection.lib`;
this module is purely the MCP-schema layer.

:func:`register` takes any ``mcp.server.Server`` and attaches the
``list_tools`` + ``call_tool`` decorators. Callers own the Server
instance and the transport.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from harness.introspection import lib


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


TOOLS: list[Tool] = [
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


HANDLERS: dict[str, Any] = {
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


def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    """Single-source dispatcher used by both transports.

    The MCP SDK's ``@server.call_tool()`` decorator wraps this with an
    async signature; the body is sync because every handler is sync
    today. Errors are surfaced as a text reply rather than raised so
    Claude Code can render them inline."""
    handler = HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"unknown tool: {name!r}")]
    # Strip empty-string args so callable defaults kick in — except
    # for ``lesson_id`` which is a real required input that happens to
    # take an empty string as "no match yet" in some test flows.
    cleaned = {
        k: v
        for k, v in (arguments or {}).items()
        if v != "" or k == "lesson_id"
    }
    try:
        result = handler(**cleaned)
    except Exception as e:
        return [TextContent(type="text", text=f"error: {type(e).__name__}: {e}")]
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------


def register(server: Server) -> None:
    """Attach list_tools + call_tool handlers to a fresh Server.

    The stdio entry point and the HTTP mount each create their own
    Server and call this. Keeping it a function (not module-level
    side effect) lets tests instantiate isolated servers per case.
    """

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return call_tool(name, arguments)
