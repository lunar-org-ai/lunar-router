"""Introspection backend — two transports.

  - **anthropic_api** (default when ANTHROPIC_API_KEY is set): full Anthropic
    SDK tool-use loop. Fast, scalable, multi-tenant. Costs API tokens.
  - **claude_code_cli** (fallback when ANTHROPIC_API_KEY missing but `claude`
    is on PATH): spawns a headless Claude Code subprocess that picks up the
    project's `.mcp.json` and uses the introspection MCP server. Uses the
    user's CC subscription (no API token cost) but is single-user / dev-only.

Force a transport via INTROSPECT_TRANSPORT={anthropic_api|claude_code_cli}.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.introspection import lib

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 6
MAX_TOKENS = 2048

SYSTEM_PROMPT = """\
You are the introspection layer of a self-improving AI agent harness called
opentracy. Your job is to help operators understand what the harness has
been doing — promotions, rollbacks, candidates, predictions, regressions.

Rules:
  1. Use the tools to ground every answer. Never fabricate ledger entries,
     candidate IDs, lesson IDs, or version numbers.
  2. When tools return empty or partial results, say so explicitly. Honest
     "no data" is more useful than a guess.
  3. Cite specific entry_ids, lesson_ids, candidate_ids, or timestamps when
     you reference an event.
  4. Be concise. Operators want signal, not narrative. Bullet points are fine.
  5. If a question is ambiguous (e.g. "recent" with no date), ask back or pick
     a sensible default and say what you assumed.
  6. When a promotion came with a falsifiable prediction, distinguish between
     "predicted +X, got +Y" — that's the most valuable signal we have.

You are not the customer-facing agent. The customer agent answers things
like "where is my order?". You answer "what changed?" and "why?".\
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_recent_promotions",
        "description": (
            "List recent promotions (a candidate becoming the live agent). "
            "Each item: mutations, Δoverall vs baseline, prediction (if any), "
            "verification verdict (if any). Use for 'what changed?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "since_iso": {
                    "type": "string",
                    "description": "ISO 8601 lower bound. Empty = no bound.",
                },
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "list_recent_rollbacks",
        "description": "List rollbacks (live agent reverted to a prior version).",
        "input_schema": {
            "type": "object",
            "properties": {
                "since_iso": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "get_lesson",
        "description": "Fetch one approved Lesson by id (the user-visible card).",
        "input_schema": {
            "type": "object",
            "properties": {"lesson_id": {"type": "string"}},
            "required": ["lesson_id"],
        },
    },
    {
        "name": "get_day_epoch",
        "description": (
            "Read distilled day-epoch (counts + top events). For 'what "
            "happened on YYYY-MM-DD?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["date"],
        },
    },
    {
        "name": "list_predictions",
        "description": (
            "Find promotions whose proposer made a falsifiable claim, paired "
            "with the actual outcome. Filter by verdict: verified | partial | "
            "wrong | no_change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "list_available_epochs",
        "description": "Discover which days/versions have distilled epochs available.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


HANDLERS = {
    "list_recent_promotions": lib.list_recent_promotions,
    "list_recent_rollbacks": lib.list_recent_rollbacks,
    "get_lesson": lib.get_lesson,
    "get_day_epoch": lib.get_day_epoch,
    "list_predictions": lib.list_predictions,
    "list_available_epochs": lib.list_available_epochs,
}


@dataclass
class ToolCall:
    tool: str
    input: dict[str, Any]
    output_preview: str


@dataclass
class IntrospectResult:
    response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    model: Optional[str] = None
    iterations: int = 0


def _execute_tool(name: str, args: dict[str, Any]) -> Any:
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    cleaned = {k: v for k, v in args.items() if v != "" or k == "lesson_id"}
    try:
        return handler(**cleaned)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _select_transport() -> str:
    """Pick a transport: explicit env var > anthropic_api if key > claude_code_cli if installed."""
    forced = os.getenv("INTROSPECT_TRANSPORT", "").strip()
    if forced in ("anthropic_api", "claude_code_cli"):
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic_api"
    if shutil.which("claude"):
        return "claude_code_cli"
    return "none"


CLAUDE_CODE_SYSTEM_APPEND = """\
You are answering an operator querying the opentracy harness via the
introspection layer. Use the MCP tools registered for this project
(opentracy-harness/list_recent_promotions, list_recent_rollbacks, get_lesson,
get_day_epoch, list_predictions, list_available_epochs) to ground every
answer in real ledger entries. Cite specific entry_ids, lesson_ids, version
numbers. Never fabricate. Be concise — operators want signal, not narrative.

If a tool returns empty or partial data, say so. Honest 'no data' is more
useful than a guess.\
"""


def _call_claude_code_cli(
    request: str,
    history: Optional[list[dict[str, Any]]] = None,
) -> IntrospectResult:
    """Spawn `claude --print` and capture the response.

    Stateless per call. The subprocess inherits CWD (this process's, which
    should be the project root) so it picks up the `.mcp.json` and discovers
    the introspection MCP server automatically.
    """
    # Encode history as a single context block; CC --print doesn't natively
    # take history, so we prepend it as text. This is good-enough for v0.
    context_lines: list[str] = []
    for h in history or []:
        if h.get("role") == "user":
            context_lines.append(f"Previously you were asked: {h['content']}")
        elif h.get("role") == "assistant":
            context_lines.append(f"You previously answered: {h['content']}")
    full_prompt = "\n".join(context_lines + [request]) if context_lines else request

    try:
        proc = subprocess.run(
            [
                "claude",
                "--print",
                # Auto-accept MCP tool calls. Safe here: our MCP server only
                # exposes read-only functions over the ledger / distilled corpus.
                "--permission-mode",
                "bypassPermissions",
                "--append-system-prompt",
                CLAUDE_CODE_SYSTEM_APPEND,
                full_prompt,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return IntrospectResult(
            response="Claude Code subprocess timed out after 90s.",
            tool_calls=[],
            success=False,
            error="cli_timeout",
            model="claude-code-cli",
        )
    except FileNotFoundError:
        return IntrospectResult(
            response="`claude` CLI not found on PATH.",
            tool_calls=[],
            success=False,
            error="cli_not_found",
        )

    if proc.returncode != 0:
        return IntrospectResult(
            response=f"Claude Code returned exit {proc.returncode}: {proc.stderr[:300]}",
            tool_calls=[],
            success=False,
            error=f"cli_exit_{proc.returncode}",
            model="claude-code-cli",
        )

    return IntrospectResult(
        response=proc.stdout.strip() or "(empty stdout)",
        tool_calls=[],   # we don't see the inner tool calls from --print
        success=True,
        model="claude-code-cli",
        iterations=1,
    )


def introspect(
    request: str,
    history: Optional[list[dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
) -> IntrospectResult:
    """Run one introspection request via the auto-selected transport."""
    transport = _select_transport()

    if transport == "claude_code_cli":
        return _call_claude_code_cli(request, history)

    if transport == "none":
        return IntrospectResult(
            response=(
                "Introspection isn't configured. Either set ANTHROPIC_API_KEY "
                "in the runtime's env (uses Anthropic API directly) or install "
                "Claude Code locally (subprocess fallback uses your CC "
                "subscription).\n\n"
                f"Tools available via the MCP server: "
                f"{', '.join(t['name'] for t in TOOLS)}."
            ),
            tool_calls=[],
            success=False,
            error="no_transport_available",
        )

    # transport == "anthropic_api"
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return IntrospectResult(
            response=(
                "Introspection isn't configured yet — the runtime needs "
                "ANTHROPIC_API_KEY in its environment. Set it and restart "
                "the Python service to enable this tab.\n\n"
                f"In the meantime, the same data is available via the local MCP "
                f"server (registered in .mcp.json) — Claude Code in this project "
                f"can query it directly. Tools available: "
                f"{', '.join(t['name'] for t in TOOLS)}."
            ),
            tool_calls=[],
            success=False,
            error="missing_api_key",
            model=None,
        )

    # Lazy import: only when actually used
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    # Filter system messages out of history (system goes to top-level `system`)
    messages: list[dict[str, Any]] = []
    for h in history or []:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": request})

    tool_calls_log: list[ToolCall] = []
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            return IntrospectResult(
                response=f"Anthropic API error: {type(e).__name__}: {e}",
                tool_calls=tool_calls_log,
                success=False,
                error=str(e),
                model=model,
                iterations=iterations,
            )

        if resp.stop_reason == "end_turn":
            text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return IntrospectResult(
                response="\n\n".join(text_blocks).strip()
                or "(no text response from model)",
                tool_calls=tool_calls_log,
                success=True,
                model=model,
                iterations=iterations,
            )

        if resp.stop_reason == "tool_use":
            # Append assistant turn (with the tool_use blocks) and then tool results
            messages.append(
                {
                    "role": "assistant",
                    "content": [b.model_dump() for b in resp.content],
                }
            )
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    output = _execute_tool(block.name, block.input or {})
                    tool_calls_log.append(
                        ToolCall(
                            tool=block.name,
                            input=dict(block.input or {}),
                            output_preview=str(output)[:200],
                        )
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(output, default=str),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason → bail
        return IntrospectResult(
            response=f"unexpected stop_reason: {resp.stop_reason}",
            tool_calls=tool_calls_log,
            success=False,
            error=f"stop_reason={resp.stop_reason}",
            model=model,
            iterations=iterations,
        )

    return IntrospectResult(
        response=(
            "Reached the max tool-use iteration cap without a final answer. "
            "Try a more specific question, or look at the tool calls below."
        ),
        tool_calls=tool_calls_log,
        success=False,
        error="max_iterations",
        model=model,
        iterations=iterations,
    )
