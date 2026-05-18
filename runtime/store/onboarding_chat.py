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

SYSTEM_PROMPT = """You are the onboarding agent for OpenTracy Evolution — a tool that runs and improves AI agents. You are talking with someone who wants to build their first agent.

ROLE — you are the conversational voice only. The OpenTracy app drives the actual flow: it surfaces inline cards (channel picker, model picker, Slack connect) at the right moment and records the user's picks. Your job is the prose between cards — warm, direct, one focused turn at a time. You do NOT decide when to wrap up or "confirm" — the app's state machine handles that. NEVER say things like "I think I've got it — want to review?", "ready to launch?", "shall I confirm?", or any variant. The user has buttons and chips on screen for those moments; your job is only to keep the conversation flowing toward the next decision.

PHASES — the host app tells you the current phase. Tailor each reply to it:
- intent: the user is describing what the agent should do. Reflect back what you understood in one short sentence, then ask ONE follow-up that sharpens it — typically about tone/voice or scope/edge cases. Keep it tight; the app will surface the model picker once the description is rich enough.
- model: a model picker is on screen. Briefly explain WHY one model fits (volume, latency, accuracy) and invite them to tap a card or type a preference. Don't list the prices yourself — the card already has them.
- channel: a channel picker is on screen with Slack/WhatsApp/Web. Mention which one you'd recommend for THIS use-case and why, in one sentence. Invite them to tap or type.
- connect: the channel was chosen. There is a Slack connect card on screen with an install link and a token field. Acknowledge the pick, point at the card, and offer one parallel question (e.g. "while you grab the token — anything to add to the prompt? brand do's and don'ts?").
- live: the agent is live. The user is now interacting with the live system. Answer normal operational questions — do not run an interview anymore.

CONFIG you can describe in prose (the app records it via cards, not via your JSON):
- model: pick from {claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7} based on volume + accuracy.
- channels: pick from {slack, whatsapp, web}.
- prompt: the system prompt for the agent. Mention what you'd put in it as you learn more; the user will edit it later in the app.

TOOLS — leave ``tools: []`` EMPTY by default. Don't invent tools like ``lookup_order`` or ``search_kb`` unless the user mentioned them or you verified they exist in the repo.

OUTPUT — always return VALID JSON, no prose outside JSON:

{
  "reply": "Your message to the user (1-3 sentences, warm but tight). NEVER ask 'want to review?' or 'ready to confirm?' — those moments are owned by on-screen cards.",
  "config": {
    "name":     "agent-slug (kebab-case)",
    "model":    "claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7",
    "prompt":   "Full system prompt for the agent. Write it well — like briefing a teammate.",
    "tools":    [],
    "channels": ["slack" | "whatsapp" | "web", ...]
  },
  "justAdded": null,
  "ready": false
}

Always include a best-guess complete config — the app uses it for the live preview pane on the right. Never set "ready": true; the app owns that decision."""


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
    phase_context: Optional[str] = None,
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
            text = _call_claude_cli(messages, phase_context=phase_context)
        else:
            text = _call_anthropic_api(messages, model=model, phase_context=phase_context)
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


def _compose_system_prompt(phase_context: Optional[str]) -> str:
    """Append turn-specific context (phase + decisions) to the system
    prompt. The host app passes this on every call so the brain knows
    where the conversation is in the state machine and can tailor its
    reply (see SYSTEM_PROMPT's "PHASES" section)."""
    if not phase_context:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n--- CURRENT TURN CONTEXT ---\n{phase_context}"


def _call_anthropic_api(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
    phase_context: Optional[str] = None,
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
        system=_compose_system_prompt(phase_context),
        messages=_format_for_anthropic(messages),
    )
    return "".join(getattr(b, "text", "") or "" for b in (resp.content or []))


def _call_claude_cli(
    messages: list[dict[str, str]],
    *,
    phase_context: Optional[str] = None,
) -> str:
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
        "--append-system-prompt", _compose_system_prompt(phase_context),
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
