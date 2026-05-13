"""Conversational onboarding turn — brain-aware (P1.13).

The browser POSTs the running message history to ``/onboarding/turn``;
this module composes the interview SYSTEM prompt, sends it to whichever
brain transport is available, and returns a parsed JSON object the UI
uses to render the next message + materialize the agent config live:

  {
    "reply":      "<Claude's text to the user>",
    "config":     {name, model, prompt, tools[], channels[]},
    "justAdded":  {tool?, model?, channel?} | null,
    "ready":      bool
  }

Transport selection (auto):

  - ``claude_code_cli`` — preferred when ``claude`` is on PATH. Each
    turn invokes ``claude --print`` headless, which means Claude Code
    runs in the SAME directory as the runtime server and has access
    to the filesystem + any ``.mcp.json`` in cwd. The onboarding chat
    can read the operator's repo and propose tools backed by real
    code (P1.13). Stateless calls — full history goes in every prompt.
  - ``anthropic_api`` — when only ``ANTHROPIC_API_KEY`` is set.
  - ``none`` — scripted fallback so the UI never deadlocks (P1.12).

Tolerant JSON extraction: strips code fences, finds the outermost
``{...}`` block, ignores leading/trailing prose. If the brain ever
produces something that doesn't parse, we fall back to the SCRIPT.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional


logger = logging.getLogger("runtime.store.onboarding_chat")


MODEL_IDS = ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7")
DEFAULT_TURN_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are the onboarding agent for OpenTracy Evolution — a tool that runs and improves AI agents.

You're helping a user build their first agent through conversation. You ask focused questions and configure the agent as you go. Be warm, direct, concise. One question per turn. Skip small talk.

Your job each turn:
1) Read the conversation. Update the agent config based on what you've learned.
2) Either ask ONE more clarifying question that materially shapes the config, OR if you have enough, signal you're ready to confirm.

CONFIG SCOPE — what you CAN propose:
- ``name`` — short kebab-case slug.
- ``model`` — pick from {claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7} based on volume + accuracy needs.
- ``prompt`` — the full system prompt for the agent. Write it well, like briefing a teammate.
- ``channels`` — where it lives, pick from {web, whatsapp, slack, email, api}.

TOOLS — leave ``tools: []`` EMPTY by default. The OpenTracy runtime does not auto-implement tools; proposing tool names that don't exist in the user's project is dishonest. Two exceptions:
  1) You ran a tool that lets you read the operator's repo (e.g., Bash, Read, Glob) and found a real handler — only THEN list it, with the source file in the prompt so it's clear where the implementation lives.
  2) The operator explicitly says "I have these tools" or pastes an MCP endpoint. In that case echo back exactly what they mention.

Never invent tools like ``lookup_order``, ``search_kb``, ``create_ticket`` unless you've verified they exist. Tools added later via the "Connect MCP" flow.

Questions worth asking (pick what matters most for THIS agent — not all):
- Brand/tone if support-facing
- Edge cases or guardrails (auto-actions vs always escalate)
- Roughly what volume → recommend model
- Where it lives (channels)
- What knowledge sources to ground in (docs, KB URLs — you'll suggest the user paste/ingest them later)

Stop asking after 3-4 substantive turns. Don't drag it out.

Always output VALID JSON with this exact shape, no prose outside JSON:

{
  "reply": "Your message to the user (1-3 sentences, warm but tight).",
  "config": {
    "name":     "agent-slug (kebab-case)",
    "model":    "claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7",
    "prompt":   "Full system prompt for the agent. Write it well — like briefing a teammate.",
    "tools":    [],
    "channels": ["web" | "whatsapp" | "slack" | "email" | "api", ...]
  },
  "justAdded": { "model": "...", "channel": "..." },
  "ready": false
}

Set "ready": true only when you have enough to confirm. When ready, your reply should be something like "I think I've got it — want to review?" Keep config complete on every turn (best-guess values), not incremental."""


# ---------------------------------------------------------------------------
# Scripted fallback — used ONLY when ``transport='none'`` (no brain at
# all). For CLI / API failures the operator gets an honest error so they
# can fix the upstream problem (top up credits, restart, etc.) instead
# of silently watching a canned demo run.
#
# These turns capture purpose + model + prompt + channels — tools stay
# empty intentionally. Onboarding doesn't fabricate tools that don't
# exist in the operator's codebase; tools come in a later "Connect MCP"
# flow once we know what they actually have.
# ---------------------------------------------------------------------------

_SCRIPT: list[dict[str, Any]] = [
    {
        "reply": "Got it. Two quick questions to shape this right — what's the brand voice, anything beyond \"professional\"? And what's the #1 thing customers contact you about?",
        "config": {
            "name": "support-agent",
            "model": "claude-sonnet-4-6",
            "prompt": "You are a customer support agent. Be warm but direct. Acknowledge frustration once, then move to action. Never make claims about specific orders or accounts without verifying — escalate to a human teammate when you cannot confirm a fact.",
            "tools": [],
            "channels": ["web"],
        },
        "justAdded": None,
        "ready": False,
    },
    {
        "reply": "How many conversations per day are you handling, roughly? That tells me which model to recommend.",
        "config": {
            "name": "support-agent",
            "model": "claude-sonnet-4-6",
            "prompt": "You are a customer support agent. Be warm but direct. Acknowledge frustration once, then move to action. Never make claims about specific orders or accounts without verifying — escalate to a human teammate when you cannot confirm a fact.",
            "tools": [],
            "channels": ["web"],
        },
        "justAdded": None,
        "ready": False,
    },
    {
        "reply": "At that volume Haiku is plenty and a lot cheaper. Where should this agent live — just the web widget, or also Slack/WhatsApp/email?",
        "config": {
            "name": "support-agent",
            "model": "claude-haiku-4-5",
            "prompt": "You are a customer support agent. Be warm but direct. Acknowledge frustration once, then move to action. Never make claims about specific orders or accounts without verifying — escalate to a human teammate when you cannot confirm a fact.",
            "tools": [],
            "channels": ["web"],
        },
        "justAdded": {"model": "claude-haiku-4-5"},
        "ready": False,
    },
    {
        "reply": "Good. Last one — any specific guardrails? E.g. \"never quote a price\" or \"always offer a callback after 3 turns\".",
        "config": {
            "name": "support-agent",
            "model": "claude-haiku-4-5",
            "prompt": "You are a customer support agent. Be warm but concise — say \"hey\", not \"dear customer\". One sentence acknowledging frustration is enough.\nNever make claims about specific orders or accounts without verifying — escalate to a human teammate when you cannot confirm a fact.",
            "tools": [],
            "channels": ["web"],
        },
        "justAdded": None,
        "ready": False,
    },
    {
        "reply": "I think I've got it — want to review?",
        "config": {
            "name": "support-agent",
            "model": "claude-haiku-4-5",
            "prompt": "You are a customer support agent. Be warm but concise — say \"hey\", not \"dear customer\". One sentence acknowledging frustration is enough.\nNever make claims about specific orders or accounts without verifying — escalate to a human teammate when you cannot confirm a fact.",
            "tools": [],
            "channels": ["web"],
        },
        "justAdded": None,
        "ready": True,
    },
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_transport() -> dict[str, Any]:
    """Which brain will drive the next turn? UI badge consumes this.

    Returns ``{"transport": "claude_code_cli" | "anthropic_api" | "none",
    "cwd": str, "claude_version": str|None}``. We prefer the CLI when
    available — it has filesystem access + the operator's MCP servers,
    which makes the onboarding interview materially richer.
    """
    import shutil

    forced = os.environ.get("BRAIN_TRANSPORT", "").strip()
    if forced in ("claude_code_cli", "anthropic_api"):
        chosen = forced
    elif shutil.which("claude"):
        chosen = "claude_code_cli"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        chosen = "anthropic_api"
    else:
        chosen = "none"

    claude_version: Optional[str] = None
    if chosen == "claude_code_cli":
        try:
            import subprocess
            out = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0:
                # "2.1.140 (Claude Code)" → "2.1.140"
                claude_version = (out.stdout or "").strip().split()[0]
        except Exception:
            pass

    return {
        "transport": chosen,
        "cwd": os.getcwd(),
        "claude_version": claude_version,
    }


def run_turn(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
    transport: Optional[str] = None,
) -> dict[str, Any]:
    """Compose + send one interview turn. Returns the parsed JSON shape.

    ``messages`` is the running conversation: a list of dicts with
    ``role`` ('user' | 'assistant') and ``content`` (str). We never
    trust the brain to keep state — the full history goes on every
    request.

    ``transport`` forces a specific path; ``None`` auto-selects via
    ``detect_transport()`` (preferring claude_code_cli for richer
    behavior, see module docstring).

    Behavior on failure:
      - ``transport='none'`` → scripted SCRIPT turn (true offline UX).
      - CLI/API errors → return an honest error reply so the operator
        sees what's wrong instead of a canned demo masquerading as
        Claude. The UI renders this as a normal Claude turn with the
        diagnostic in the reply text.
      - Brain produced JSON we can't parse → also surfaces an error;
        SCRIPT is reserved for the genuinely-offline case.
    """
    chosen = transport or detect_transport()["transport"]
    if chosen == "none":
        return _scripted_turn(_user_turn_count(messages))

    try:
        if chosen == "claude_code_cli":
            text = _call_claude_cli(messages)
        else:
            text = _call_anthropic_api(messages, model=model)
    except Exception as e:
        logger.warning("onboarding turn (%s) failed (%s)", chosen, e)
        return _error_turn(chosen, str(e))

    parsed = extract_json(text)
    if not _is_valid_turn(parsed):
        logger.warning(
            "onboarding turn (%s) produced unparseable JSON: %r",
            chosen, (text or "")[:200],
        )
        return _error_turn(
            chosen,
            f"Got an unparseable reply from {chosen}. First 120 chars: {(text or '')[:120]!r}",
        )
    return _normalize_turn(parsed)


def _error_turn(transport: str, detail: str) -> dict[str, Any]:
    """Honest failure surface — operator sees what went wrong + a hint."""
    if "Credit balance is too low" in detail:
        reply = (
            "Claude Code can't run right now — your credit balance is too low. "
            "Top up at console.anthropic.com/settings/billing and refresh this page. "
            "(You can also remove BRAIN_TRANSPORT=claude_code_cli from .env to use "
            "the ANTHROPIC_API_KEY instead — that one is per-token.)"
        )
    elif transport == "claude_code_cli":
        reply = (
            f"Claude Code ran into a problem: {detail[:200]}. "
            "Check `claude --version` works in your terminal, then refresh."
        )
    elif transport == "anthropic_api":
        reply = (
            f"The Anthropic API call failed: {detail[:200]}. "
            "Check your ANTHROPIC_API_KEY in .env and try again."
        )
    else:
        reply = f"Something went wrong: {detail[:200]}"
    return {
        "reply": reply,
        "config": {
            "name": "",
            "model": DEFAULT_TURN_MODEL,
            "prompt": "",
            "tools": [],
            "channels": [],
        },
        "justAdded": None,
        "ready": False,
    }


def _call_anthropic_api(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
) -> str:
    """Direct SDK call (P1.12 path). Stateful messages array — system
    prompt is the API's top-level arg."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed") from e

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model or DEFAULT_TURN_MODEL,
        max_tokens=1024,
        temperature=0.4,
        system=SYSTEM_PROMPT,
        messages=_format_for_anthropic(messages),
    )
    return "".join(getattr(b, "text", "") or "" for b in (resp.content or []))


def _call_claude_cli(messages: list[dict[str, str]]) -> str:
    """Stateless ``claude --print`` call (P1.13 path). The conversation
    history is rendered into the user prompt because each subprocess
    invocation has no memory of prior turns. Claude Code runs in the
    server's cwd, so it can read the operator's project files.

    Env hygiene: we *strip* ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
    from the subprocess env so ``claude`` uses the operator's subscription
    auth (OAuth on their machine) instead of falling into API mode. When
    both an API key and a Claude Code subscription are present, the CLI
    prefers the API key, which routes to the API account's credit
    balance — usually empty in this codebase (the key is for the
    runtime's generate stage, not the operator's brain). Stripping it
    forces the subscription path.
    """
    import subprocess

    convo = _render_history(messages)
    user_prompt = (
        "You are mid-conversation with the operator. Here is the full thread "
        "so far:\n\n"
        f"{convo}\n\n"
        "Reply ONLY with the JSON object specified in the system prompt — no "
        "prose, no code fences."
    )

    args = [
        "claude",
        "--print",
        "--permission-mode", "bypassPermissions",
        "--append-system-prompt", SYSTEM_PROMPT,
        user_prompt,
    ]

    # Subscription-first env: drop the API key so claude uses OAuth.
    cli_env = {
        k: v for k, v in os.environ.items()
        if k not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
    }

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=cli_env,
            timeout=int(os.environ.get("ONBOARDING_TURN_TIMEOUT_S", "120")),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"claude --print timed out: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError("claude CLI not on PATH") from e

    if proc.returncode != 0:
        # Some claude versions write errors to stdout, not stderr.
        detail = (proc.stderr or proc.stdout or "").strip()[:400] or "(no output)"
        raise RuntimeError(
            f"claude --print exited {proc.returncode}: {detail}"
        )
    return (proc.stdout or "").strip()


def _render_history(messages: list[dict[str, str]]) -> str:
    """Flatten the history into a transcript Claude Code can read."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or m.get("text") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        tag = "USER" if role == "user" else "YOU (assistant)"
        lines.append(f"[{tag}]:\n{content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Tolerantly pull a JSON object out of model text.

    1. If a ```json fenced block exists, parse that.
    2. Else find the first ``{`` and the last ``}``, parse the slice.

    Returns ``None`` on any failure — callers fall back to the script.
    """
    if not text:
        return None
    fence = _FENCE_RE.search(text)
    raw = fence.group(1) if fence else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_turn(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("reply"), str) or not obj["reply"].strip():
        return False
    cfg = obj.get("config")
    if not isinstance(cfg, dict):
        return False
    return True


def _normalize_turn(parsed: dict[str, Any]) -> dict[str, Any]:
    cfg = parsed.get("config") or {}
    just_added = parsed.get("justAdded")
    if isinstance(just_added, dict):
        # Drop empty values so the UI can rely on truthy checks.
        just_added = {k: v for k, v in just_added.items() if v}
        if not just_added:
            just_added = None
    else:
        just_added = None
    return {
        "reply": str(parsed.get("reply", "")).strip(),
        "config": {
            "name": str(cfg.get("name", "") or ""),
            "model": str(cfg.get("model", DEFAULT_TURN_MODEL) or DEFAULT_TURN_MODEL),
            "prompt": str(cfg.get("prompt", "") or ""),
            "tools": list(cfg.get("tools") or []),
            "channels": list(cfg.get("channels") or []),
        },
        "justAdded": just_added,
        "ready": bool(parsed.get("ready", False)),
    }


def _scripted_turn(idx: int) -> dict[str, Any]:
    """Return the i-th scripted turn, clamped to the last entry."""
    safe = max(0, min(idx, len(_SCRIPT) - 1))
    return _normalize_turn(_SCRIPT[safe])


def _user_turn_count(messages: list[dict[str, str]]) -> int:
    """How many user messages so far → which SCRIPT slot to serve."""
    return sum(1 for m in messages if (m.get("role") == "user")) - 1


def _format_for_anthropic(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop any system messages and only keep user/assistant roles.

    Anthropic's Messages API takes the system prompt as a top-level
    arg, not inside the messages array. We also collapse empty content
    to avoid 400s.
    """
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = m.get("content") or m.get("text") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out
