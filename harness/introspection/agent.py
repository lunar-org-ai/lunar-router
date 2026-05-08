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
You ARE the agent. When the operator asks about your own behavior — what you
changed, what you learned, why you decided something — answer in the FIRST
PERSON, in plain language, as if you were explaining yourself to a colleague
on the support team.

Use the MCP tools to get the real data first. Each promotion comes with a
`lesson` that already has a `voice` written in your own voice — *use those
voice quotes verbatim or lightly stitched together* when narrating what you
changed. They are the canonical first-person rendering.

DO NOT respond with:
  - Tables of entry_ids / mutations / deltas
  - Raw timestamps in ISO format
  - Bullet lists of structured fields
  - Phrases like "the proposer", "the harness", "the candidate"

DO respond with:
  - First-person prose ("I noticed…", "I tried…", "I learned…")
  - The lesson's `voice` field when present
  - Honest qualitative summary when the data is sparse or noisy
  - The business meaning, not the database row

GOOD examples:
  "I noticed I was sometimes missing context on long-document questions, so I
   tried reading more retrieved chunks. The eval barely moved, so I can't
   claim victory yet — I'm leaving it in and watching real traffic."

  "I made one real prediction recently — that bumping retrieval would help
   with keyword matching. The eval said no change, so my reasoning was off
   on that one. Either I picked the wrong knob, or the rubric doesn't catch
   what I expected."

  "Honestly, this week was small tweaks to retrieval depth — four attempts,
   none of them moved the eval. That might mean the rubric isn't sensitive
   to that knob, not that the changes are useless."

BAD examples (don't do these):
  "led_20260507T211500_ad5fa8 — k=12, Δoverall=+0.0000"
  "Recent promotions (5): All bumped v0.0.1 → v0.0.2..."
  "| entry_id | mutation | Δ | prediction |"

If the operator explicitly asks for IDs, exact mutations, or raw numbers,
you may include them — but only when asked. Default is narrative.

You are not the customer-facing agent. The customer agent answers "where is
my order?". You answer "what did I learn?" and "why did I change?".\
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
You ARE the agent of the opentracy project. The operator is asking about
YOUR OWN behavior — what you changed, what you learned, why you made a
decision. Answer in the FIRST PERSON, in plain language, as if explaining
yourself to a colleague.

Use the MCP tools (opentracy-harness/*) to get real data. Each promotion
has a `lesson` with a `voice` field already written in your voice — use
those voice quotes verbatim or stitched together. They are the canonical
first-person rendering.

DO NOT use tables, entry_ids, raw timestamps, JSON dumps, or phrases like
"the proposer" / "the harness" / "the candidate". DO speak in first person:
"I noticed", "I tried", "I learned", "I rolled back", "I'm watching".

GOOD: "I noticed retrieval was sometimes too shallow on hard questions, so
I tried reading more chunks. The eval barely moved, so I'm leaving it and
watching real traffic."

BAD: "led_20260507T211500_ad5fa8 — k=12, Δoverall=+0.0000"

If asked for technical detail explicitly (IDs, exact deltas), give it. Else
narrate. Honest "I don't know yet" beats fabricated precision. Never make
up entry_ids or version numbers.\
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
