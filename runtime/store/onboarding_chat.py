"""Conversational onboarding turn — Claude-led interview (P1.12).

The browser POSTs the running message history to ``/onboarding/turn``;
this module composes the interview SYSTEM prompt, calls Anthropic, and
returns a parsed JSON object the UI uses to render the next message +
materialize the agent config live:

  {
    "reply":      "<Claude's text to the user>",
    "config":     {name, model, prompt, tools[], channels[]},
    "justAdded":  {tool?, model?, channel?} | null,
    "ready":      bool
  }

Tolerant JSON extraction: strips code fences, finds the outermost
``{...}`` block, ignores leading/trailing prose. If the LLM ever
produces something that doesn't parse, we fall back to the
hard-coded SCRIPT path so the wizard always advances.

Offline-safe: if ``ANTHROPIC_API_KEY`` is missing, every turn comes
from the scripted fallback. The UI still functions; the conversation
just feels canned. This mirrors the runtime stage's offline behavior
(P1.9).
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

Questions worth asking (pick what matters most for THIS agent — not all):
- Brand/tone if support-facing
- Edge cases or guardrails (auto-actions vs always escalate)
- Roughly what volume → recommend model
- Where it lives (channels)
- What tools/data it needs access to

Stop asking after 3-4 substantive turns. Don't drag it out.

Always output VALID JSON with this exact shape, no prose outside JSON:

{
  "reply": "Your message to the user (1-3 sentences, warm but tight).",
  "config": {
    "name":     "agent-slug (kebab-case)",
    "model":    "claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7",
    "prompt":   "Full system prompt for the agent. Write it well — like briefing a teammate.",
    "tools":    ["snake_case_tool", ...],
    "channels": ["web" | "whatsapp" | "slack" | "email" | "api", ...]
  },
  "justAdded": { "tool": "...", "model": "...", "channel": "..." },
  "ready": false
}

Set "ready": true only when you have enough to confirm. When ready, your reply should be something like "I think I've got it — want to review?" Keep config complete on every turn (best-guess values), not incremental."""


# ---------------------------------------------------------------------------
# Scripted fallback — used when no API key or the model misbehaves.
# Five turns, support-agent flavor, mirrors the design's SCRIPT.
# ---------------------------------------------------------------------------

_SCRIPT: list[dict[str, Any]] = [
    {
        "reply": "Got it. A couple things to lock in: when a customer asks for a refund, should the agent ever auto-issue, or always create a ticket for a human?",
        "config": {
            "name": "checkout-support",
            "model": "claude-sonnet-4-6",
            "prompt": "You are a customer support agent for {{company}}.\nBe warm but direct. Acknowledge frustration once, then move to action.\nUse lookup_order before guessing about order state.",
            "tools": ["lookup_order", "search_kb"],
            "channels": ["web"],
        },
        "justAdded": {"tool": "lookup_order"},
        "ready": False,
    },
    {
        "reply": "Perfect. And how many conversations per day are you handling now? Affects which model I'd recommend.",
        "config": {
            "name": "checkout-support",
            "model": "claude-sonnet-4-6",
            "prompt": "You are a customer support agent for {{company}}.\nBe warm but direct. Acknowledge frustration once, then move to action.\nNever auto-issue refunds — always open a ticket via create_refund_ticket.\nUse lookup_order before guessing about order state.",
            "tools": ["lookup_order", "search_kb", "create_refund_ticket"],
            "channels": ["web"],
        },
        "justAdded": {"tool": "create_refund_ticket"},
        "ready": False,
    },
    {
        "reply": "At ~400/day, Haiku will be plenty and a lot cheaper. What's the brand voice? Anything past \"professional\"?",
        "config": {
            "name": "checkout-support",
            "model": "claude-haiku-4-5",
            "prompt": "You are a customer support agent for {{company}}.\nBe warm but direct. Acknowledge frustration once, then move to action.\nNever auto-issue refunds — always open a ticket via create_refund_ticket.\nUse lookup_order before guessing about order state.",
            "tools": ["lookup_order", "search_kb", "create_refund_ticket"],
            "channels": ["web"],
        },
        "justAdded": {"model": "claude-haiku-4-5"},
        "ready": False,
    },
    {
        "reply": "Where should this agent live — web widget, WhatsApp, both?",
        "config": {
            "name": "checkout-support",
            "model": "claude-haiku-4-5",
            "prompt": "You are a customer support agent for {{company}}.\nBe warm but concise — say \"hey\", not \"dear customer\". One sentence acknowledging frustration is enough.\nNever auto-issue refunds — always open a ticket via create_refund_ticket.\nUse lookup_order before guessing about order state.",
            "tools": ["lookup_order", "search_kb", "create_refund_ticket"],
            "channels": ["web"],
        },
        "justAdded": None,
        "ready": False,
    },
    {
        "reply": "I think I've got it — want to review?",
        "config": {
            "name": "checkout-support",
            "model": "claude-haiku-4-5",
            "prompt": "You are a customer support agent for {{company}}.\nBe warm but concise — say \"hey\", not \"dear customer\". One sentence acknowledging frustration is enough.\nNever auto-issue refunds — always open a ticket via create_refund_ticket.\nUse lookup_order before guessing about order state.",
            "tools": ["lookup_order", "search_kb", "create_refund_ticket"],
            "channels": ["web", "whatsapp"],
        },
        "justAdded": {"channel": "whatsapp"},
        "ready": True,
    },
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_turn(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Compose + send one interview turn. Returns the parsed JSON shape.

    ``messages`` is the running conversation: a list of dicts with
    ``role`` ('user' | 'assistant') and ``content`` (str). We never
    trust the model to keep state — the full history goes on every
    request.

    On any failure (no key, network blip, malformed JSON) we fall
    through to the SCRIPT. The UI shouldn't crash because of a model
    hiccup during day-0 setup.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _scripted_turn(_user_turn_count(messages))

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — onboarding falls back to script")
        return _scripted_turn(_user_turn_count(messages))

    chosen_model = model or DEFAULT_TURN_MODEL
    client = Anthropic(api_key=api_key)
    api_messages = _format_for_anthropic(messages)
    try:
        resp = client.messages.create(
            model=chosen_model,
            max_tokens=1024,
            temperature=0.4,
            system=SYSTEM_PROMPT,
            messages=api_messages,
        )
    except Exception as e:
        logger.warning("onboarding turn Anthropic call failed (%s) — falling back", e)
        return _scripted_turn(_user_turn_count(messages))

    text = "".join(getattr(b, "text", "") or "" for b in (resp.content or []))
    parsed = extract_json(text)
    if not _is_valid_turn(parsed):
        logger.warning("onboarding turn produced unparseable JSON — falling back. Raw: %r", text[:200])
        return _scripted_turn(_user_turn_count(messages))
    return _normalize_turn(parsed)


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
