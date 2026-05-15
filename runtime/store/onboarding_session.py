"""Onboarding session — stateful, card-driven conversation (Phase A).

Replaces the stateless ``/onboarding/turn`` flow with a persistent
session per tenant. The UI no longer ships the running history on
every call; the server is the source of truth and the UI just
fetches the current shape.

State machine
-------------

Phases progress in one direction. ``rewind`` is the only way back.

    intent → model → channel → connect → live → done

Each phase has a ``decision_key``. Picking a card materializes the
decision, advances the phase, and (when the phase warrants it) emits
the next deterministic card on the assistant's next turn.

  intent  → user describes purpose (no card). Brain decides via
            ``ready_for_next_phase: true`` when it has enough.
  model   → server emits ModelPicker; user picks → advance.
  channel → server emits ChannelPicker; user picks → advance.
  connect → server emits ConnectSlackCard with install URL +
            agent-key preview + status. UI polls
            /connect/slack/status; when ``installed`` flips to
            true, the card transitions to ``connected``.
  live    → server emits TracePreview from the first real message.
  done    → user has acked; route to the main app.

Brain integration
-----------------

The brain keeps its existing JSON contract (``reply``, ``config``,
``ready``) but reads + writes the *server-driven* phase. The phase
transition is server logic; the brain's only signal is
``ready_for_next_phase``. The server only acts on it once per
phase (idempotent).

Storage
-------

One JSON file per tenant: ``tenants/<id>/agent/onboarding_session.json``.
Schema versioned via ``version: 1`` so we can migrate later. Writes
go through ``runtime.tenants.tokens``-style atomic rename, with the
same concat-recovery loader if two requests race.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from runtime.store.onboarding_chat import (
    DEFAULT_TURN_MODEL,
    run_turn as _run_brain_turn,
)


logger = logging.getLogger("runtime.store.onboarding_session")


_SESSION_FILENAME = "onboarding_session.json"
_AGENT_DIRNAME = "agent"

# Per-tenant session locks so two concurrent /say or /decide calls
# don't lose state. The session file is small and the critical
# section is short (read → mutate → save), so the lock is cheap.
_SESSION_LOCKS: dict[str, threading.Lock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


def _lock_for(session_path: Path) -> threading.Lock:
    key = str(session_path)
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[key] = lock
        return lock


# ─── Card types ──────────────────────────────────────────────────────


_MODEL_OPTIONS = [
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "cost_per_million_in": "$0.80",
        "p50_latency_s": 1.2,
        "tag": "fast",
    },
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "cost_per_million_in": "$3.00",
        "p50_latency_s": 2.4,
        "tag": "balanced",
    },
    {
        "id": "claude-opus-4-7",
        "name": "Claude Opus 4.7",
        "cost_per_million_in": "$15.00",
        "p50_latency_s": 3.6,
        "tag": "smartest",
    },
]

_CHANNEL_OPTIONS = [
    {"id": "slack", "name": "Slack", "sub": "For internal teams"},
    {"id": "whatsapp", "name": "WhatsApp", "sub": "For customers"},
    {"id": "web", "name": "Web widget", "sub": "For your site"},
    {"id": "api", "name": "API", "sub": "Direct HTTP calls"},
]


# ─── State ───────────────────────────────────────────────────────────


@dataclass
class Turn:
    """One assistant or user message in the session."""

    id: str
    ts: str
    role: str            # 'assistant' | 'user'
    text: str = ""
    cards: list[dict[str, Any]] = field(default_factory=list)
    # When the user PICKS something on a card, the resulting settled
    # chip rides on the *next* user turn. The decision_key tells the
    # UI which decision this turn represents so the chip can be
    # rendered + the "edit" affordance can wire to /rewind.
    decision_key: Optional[str] = None
    decision_value: Optional[str] = None
    decision_label: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "role": self.role,
            "text": self.text,
            "cards": list(self.cards),
            "decision_key": self.decision_key,
            "decision_value": self.decision_value,
            "decision_label": self.decision_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            id=str(data.get("id") or ""),
            ts=str(data.get("ts") or ""),
            role=str(data.get("role") or ""),
            text=str(data.get("text") or ""),
            cards=list(data.get("cards") or []),
            decision_key=data.get("decision_key"),
            decision_value=data.get("decision_value"),
            decision_label=data.get("decision_label"),
        )


@dataclass
class Session:
    """Top-level state for one tenant's onboarding."""

    version: int = 1
    session_id: str = ""
    started_at: str = ""
    phase: str = "intent"   # intent|model|channel|connect|live|done
    turns: list[Turn] = field(default_factory=list)
    decisions: dict[str, Any] = field(default_factory=dict)
    # The agent gets created when the user picks a channel. Holding
    # the id here means later phases (connect, live) can target the
    # right agent without leaking it through every endpoint.
    agent_id: Optional[str] = None
    # Slack-specific live state. ``installed`` flips to true after
    # the OAuth callback runs (Phase C); ``first_message_at`` is set
    # when the first real Slack event arrives via /onboarding/connect/slack/event.
    slack: dict[str, Any] = field(default_factory=lambda: {
        "installed": False,
        "first_message_at": None,
    })
    # Agent-key plaintext shown ONCE inside the Connect card. We
    # never persist it again — only the masked preview the UI
    # renders. (The hash + tenant binding live in the regular
    # tokens table.)
    agent_key_preview: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "phase": self.phase,
            "turns": [t.to_dict() for t in self.turns],
            "decisions": dict(self.decisions),
            "agent_id": self.agent_id,
            "slack": dict(self.slack),
            "agent_key_preview": self.agent_key_preview,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            version=int(data.get("version") or 1),
            session_id=str(data.get("session_id") or ""),
            started_at=str(data.get("started_at") or ""),
            phase=str(data.get("phase") or "intent"),
            turns=[Turn.from_dict(t) for t in (data.get("turns") or [])],
            decisions=dict(data.get("decisions") or {}),
            agent_id=data.get("agent_id"),
            slack=dict(data.get("slack") or {"installed": False, "first_message_at": None}),
            agent_key_preview=data.get("agent_key_preview"),
        )


# ─── Path resolution (mirrors runtime.store.onboarding) ──────────────


def _resolve_session_path() -> Path:
    """Tenant-aware path to the session JSON.

    Multi-tenant: ``tenants/<active>/agent/onboarding_session.json``.
    OSS-local: ``agent/onboarding_session.json`` at the runtime cwd.
    """
    from runtime.tenants.feature import is_multi_tenant_enabled

    if not is_multi_tenant_enabled():
        return Path(_AGENT_DIRNAME) / _SESSION_FILENAME

    from runtime.tenant_context import get_active

    return Path("tenants") / get_active() / _AGENT_DIRNAME / _SESSION_FILENAME


# ─── Persistence ─────────────────────────────────────────────────────


def _load(path: Path) -> Optional[Session]:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return Session.from_dict(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("onboarding session at %s unreadable (%s) — starting fresh", path, e)
        return None


def _save(session: Session, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


# ─── Helpers ─────────────────────────────────────────────────────────


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def _model_label(model_id: str) -> str:
    for opt in _MODEL_OPTIONS:
        if opt["id"] == model_id:
            return str(opt["name"])
    return model_id


def _channel_label(channel_id: str) -> str:
    for opt in _CHANNEL_OPTIONS:
        if opt["id"] == channel_id:
            return str(opt["name"])
    return channel_id


# ─── Card factories ──────────────────────────────────────────────────


def _model_card(rationale: str) -> dict[str, Any]:
    return {
        "type": "model_picker",
        "recommended_id": "claude-haiku-4-5",
        "rationale": rationale,
        "options": list(_MODEL_OPTIONS),
    }


def _channel_card() -> dict[str, Any]:
    return {
        "type": "channel_picker",
        "recommended_id": "slack",
        "options": list(_CHANNEL_OPTIONS),
    }


def _connect_card(session: Session) -> dict[str, Any]:
    """Channel-specific connect card. Each channel has its own UX:

    - Slack: OAuth install button + paste agent key into Slack app config
    - WhatsApp: webhook URL + verify token (Meta Business setup)
    - Web: <script> snippet for site embed
    - API: curl example + agent key

    The ``session.slack`` field is reused as a generic install/status
    bag so we don't have to refactor the session schema for every
    channel — the meaning of "installed" is channel-dependent.
    """
    channel = (session.decisions or {}).get("channel", "slack")
    masked = session.agent_key_preview or "ot_live_••••••••••••••••"
    connected = bool(session.slack.get("installed"))
    status = "connected" if connected else "waiting"

    if channel == "whatsapp":
        return {
            "type": "connect_whatsapp",
            "webhook_url": "https://api.opentracy.cloud/v1/channels/whatsapp/webhook",
            "verify_token_preview": masked,
            "status": status,
        }
    if channel == "web":
        # The P3.5 widget infra owns the runtime side: each widget gets a
        # widget_id and a self-hostable embed JS at /widget/<id>/v1.js. We
        # mint the widget config during _materialize_agent so session.slack
        # carries widget_id + signing_secret_plaintext_once for this card.
        host = os.environ.get(
            "PUBLIC_BASE_URL", "https://api.dev.opentracy.cloud",
        ).rstrip("/")
        widget_id = session.slack.get("widget_id") or "<your-widget-id>"
        embed_url = f"{host}/widget/{widget_id}/v1.js"
        return {
            "type": "connect_web",
            "embed_snippet": (
                f"<script async src=\"{embed_url}\"></script>"
            ),
            "agent_key_preview": masked,
            "status": status,
        }
    if channel == "api":
        # Hostname is environment-specific. Cloud Run injects
        # PUBLIC_BASE_URL via the runtime-{env}.yaml spec; defaults to
        # the dev gateway for local runs. Note: until a Cloud Run
        # domain mapping is wired for api.dev.opentracy.cloud (manual
        # step, requires Google domain verification), this URL will
        # 502 from the operator's browser — fall back to the direct
        # *.run.app URL by overriding PUBLIC_BASE_URL in the spec.
        host = os.environ.get(
            "PUBLIC_BASE_URL", "https://api.dev.opentracy.cloud",
        ).rstrip("/")
        agent_id = session.agent_id or "<your-agent-id>"
        endpoint = f"{host}/v1/api/{agent_id}/chat"
        return {
            "type": "connect_api",
            "endpoint": endpoint,
            "agent_id": agent_id,
            "agent_key_preview": masked,
            "curl_example": (
                f"curl {endpoint} \\\n"
                f"  -H 'Authorization: Bearer {masked}' \\\n"
                "  -H 'Content-Type: application/json' \\\n"
                "  -d '{\"request\": \"hello\"}'"
            ),
            "status": status,
        }
    # default: slack
    return {
        "type": "connect_slack",
        "install_url": "#stub-slack-install",
        "agent_key_preview": masked,
        "status": status,
    }


def _trace_preview_card(session: Session) -> dict[str, Any]:
    # Placeholder until we have a real trace to surface. Phase C will
    # replace this with the actual first Slack event.
    return {
        "type": "trace_preview",
        "trace_id": "trc_pending",
        "summary": {
            "channel": "support-test",
            "duration_s": 0,
            "turn_count": 0,
            "status": "waiting",
        },
    }


# ─── Brain bridge ────────────────────────────────────────────────────


def _render_brain_history(session: Session) -> list[dict[str, str]]:
    """Flatten turns into the role/content list the brain expects.

    Cards + chips don't go to the brain — only the prose. The brain's
    job is conversational copy; cards are server-emitted.
    """
    out: list[dict[str, str]] = []
    for t in session.turns:
        if t.role not in {"user", "assistant"}:
            continue
        # Chip-only user turns (no text) still need to be represented
        # to the brain so it knows the user picked something.
        if t.role == "user" and not t.text and t.decision_label:
            out.append({"role": "user", "content": f"[picked: {t.decision_label}]"})
            continue
        if not t.text.strip():
            continue
        out.append({"role": t.role, "content": t.text})
    return out


_PHASE_GUIDANCE = {
    "intent": (
        "User is describing what the agent should do. Reflect what you "
        "understood in ONE short sentence, then ask ONE follow-up that "
        "sharpens scope, tone, or audience. Do not propose models or "
        "channels yet — the host app will surface a model picker as "
        "soon as the description is rich enough."
    ),
    "model": (
        "A model-picker card is on screen. Briefly say which model fits "
        "and WHY (volume, latency, accuracy). Invite the user to tap a "
        "card or type a preference. Do not list prices — the card has them."
    ),
    "channel": (
        "A channel-picker card (Slack / WhatsApp / Web / API) is on screen. "
        "Mention which channel you'd recommend for THIS use-case in one "
        "sentence and invite the user to tap or type."
    ),
    "connect": (
        "A channel-specific connect card is on screen — Slack install link, "
        "WhatsApp webhook URL, Web embed snippet, or API curl example. "
        "Acknowledge the choice, point at the card, and offer ONE parallel "
        "question (e.g. brand do's and don'ts, escalation rules)."
    ),
    "live": (
        "The agent is live. Stop interviewing — answer normal operational "
        "questions like a colleague."
    ),
    "done": "Onboarding is complete. Reply briefly to anything the user says.",
}


def _phase_context(session: Session) -> str:
    """Compact turn-context string the brain reads to know where we are
    in the state machine. The host app owns the state — this is just a
    snapshot for the brain to tailor its prose."""
    decisions = session.decisions or {}
    settled: list[str] = []
    for key in ("purpose", "tone", "channel", "model"):
        v = decisions.get(key)
        if v:
            settled.append(f"  - {key}: {v}")
    settled_block = "\n".join(settled) if settled else "  (none yet)"
    guidance = _PHASE_GUIDANCE.get(session.phase, "")
    return (
        f"phase: {session.phase}\n"
        f"decisions so far:\n{settled_block}\n"
        f"this-turn guidance: {guidance}"
    )


def _brain_reply(session: Session) -> str:
    """Ask the brain for the next assistant message.

    Errors fall back to a phase-specific default so the conversation
    never deadlocks on a brain outage. The brain's structured
    ``config`` payload is ignored here — the session module owns the
    config materialization via deterministic phase transitions, and
    ``ready`` is also ignored (the on-screen cards close the loop).
    """
    history = _render_brain_history(session)
    try:
        out = _run_brain_turn(
            history,
            model=DEFAULT_TURN_MODEL,
            phase_context=_phase_context(session),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("brain turn failed (%s) — using default reply", e)
        return _default_reply(session.phase)
    reply = (out.get("reply") or "").strip()
    return reply or _default_reply(session.phase)


_DEFAULT_REPLIES = {
    "intent": (
        "Hi — I'm here to set up your first agent.\n\n"
        "Tell me what it should do and who it's for. The more specific, the better I can pick a model and write the prompt."
    ),
    "model": "Based on what you've told me, here's the model I'd start with — happy to swap if you want something else.",
    "channel": "Got it. Where should this agent live? Pick one to start — you can add more later.",
    "connect": "Great. I've created your agent. Two things and you'll see your first real message land here.",
    "live": "🎉 Your first message just landed. From here on, every conversation lands in your inbox.",
    "done": "All set — you can switch over to the main app anytime.",
}


def _default_reply(phase: str) -> str:
    return _DEFAULT_REPLIES.get(phase, "Let's keep going.")


# ─── Public API ──────────────────────────────────────────────────────


def get_or_create_session() -> Session:
    """Return the current session, seeding it on first call."""
    path = _resolve_session_path()
    with _lock_for(path):
        session = _load(path)
        if session is not None:
            return session
        return _seed(path)


def _seed(path: Path) -> Session:
    """First touch — create the session and emit the opening turn."""
    session = Session(
        session_id=_new_id("ses"),
        started_at=_now_iso(),
        phase="intent",
    )
    opening = Turn(
        id=_new_id("t"),
        ts=_now_iso(),
        role="assistant",
        text=_default_reply("intent"),
    )
    session.turns.append(opening)
    _save(session, path)
    return session


def say(text: str) -> Session:
    """Append a user message and produce the assistant's reply.

    Phase transitions happen here when the brain signals readiness or
    when the user's message implies advancement (e.g. they typed
    "slack" instead of clicking the picker). The state machine stays
    in the same phase if the brain isn't sure yet.
    """
    text = (text or "").strip()
    if not text:
        return get_or_create_session()
    path = _resolve_session_path()
    with _lock_for(path):
        session = _load(path) or _seed(path)
        user_turn = Turn(
            id=_new_id("t"),
            ts=_now_iso(),
            role="user",
            text=text,
        )
        session.turns.append(user_turn)
        # Update intent decision if we're in that phase. The
        # *latest* user message defines the purpose; the model and
        # channel cards then nudge the conversation forward.
        if session.phase == "intent":
            session.decisions["purpose"] = text

        reply_text = _brain_reply(session)
        reply = Turn(
            id=_new_id("t"),
            ts=_now_iso(),
            role="assistant",
            text=reply_text,
        )

        # Phase-driven card emission. The brain's job is the prose;
        # we attach a card when this phase calls for one. The card
        # only renders once per phase — subsequent assistant turns in
        # the same phase carry prose only.
        if session.phase == "intent" and _has_enough_intent(session):
            session.phase = "model"
            reply.cards.append(_model_card(_model_rationale(session)))
        elif session.phase == "model" and "model" not in session.decisions:
            # Brain hasn't moved us yet but the user asked for the
            # picker again — re-emit. Rare path but cheap to support.
            reply.cards.append(_model_card(_model_rationale(session)))

        session.turns.append(reply)
        _save(session, path)
        return session


def decide(decision_key: str, value: str) -> Session:
    """Record a card pick and advance the phase.

    ``decision_key`` is one of ``model | channel``. We materialize
    the chip on the user side, advance the phase, and emit the next
    card on a fresh assistant turn. The brain isn't asked here — the
    card pick is a deterministic transition.
    """
    if decision_key not in {"model", "channel"}:
        raise ValueError(f"unknown decision_key: {decision_key!r}")
    path = _resolve_session_path()
    with _lock_for(path):
        session = _load(path) or _seed(path)

        label = _model_label(value) if decision_key == "model" else _channel_label(value)
        session.decisions[decision_key] = value

        # Chip-bearing user turn — empty text, the chip carries the
        # decision. The UI renders this as a settled pill on the
        # user's side of the thread.
        chip_turn = Turn(
            id=_new_id("t"),
            ts=_now_iso(),
            role="user",
            text="",
            decision_key=decision_key,
            decision_value=value,
            decision_label=label,
        )
        session.turns.append(chip_turn)

        # Advance + emit next deterministic card.
        if decision_key == "model":
            session.phase = "channel"
            reply = Turn(
                id=_new_id("t"),
                ts=_now_iso(),
                role="assistant",
                text="Got it — using " + label + ". Where should this agent live?",
                cards=[_channel_card()],
            )
            session.turns.append(reply)
        elif decision_key == "channel":
            session.phase = "connect"
            # Materialize the agent now so the connect card has a
            # real key to surface. We defer to a callback so this
            # module doesn't pull in the agents registry directly
            # (avoids an import cycle in tests).
            _materialize_agent(session)
            reply = Turn(
                id=_new_id("t"),
                ts=_now_iso(),
                role="assistant",
                text=(
                    "Created your agent. Two things and you'll see your first real "
                    "message land here."
                ),
                cards=[_connect_card(session)],
            )
            session.turns.append(reply)

        _save(session, path)
        return session


def rewind(decision_key: str) -> Session:
    """Drop a decision and re-emit the picker.

    Truncates all turns after the chip-bearing turn for this
    decision, removes the decision from the dict, sets the phase
    back to the picker's phase, and emits a fresh assistant turn
    with the picker re-attached.
    """
    if decision_key not in {"model", "channel"}:
        raise ValueError(f"unknown decision_key: {decision_key!r}")
    path = _resolve_session_path()
    with _lock_for(path):
        session = _load(path)
        if session is None:
            return _seed(path)

        # Find the chip turn that recorded this decision.
        anchor: Optional[int] = None
        for i, t in enumerate(session.turns):
            if t.decision_key == decision_key:
                anchor = i
                break
        if anchor is None:
            # Decision wasn't actually settled — nothing to do.
            return session

        # Drop the chip turn AND everything after it.
        session.turns = session.turns[:anchor]
        session.decisions.pop(decision_key, None)

        # Cascading rewind: if the user rewinds model, they also
        # lose the channel decision (which depended on the model
        # picker having closed). Keeps the state machine coherent.
        if decision_key == "model":
            session.decisions.pop("channel", None)
            session.agent_id = None
            session.agent_key_preview = None
            session.slack["installed"] = False
            session.slack["first_message_at"] = None

        # Reset phase + re-emit the picker on a new assistant turn.
        session.phase = "model" if decision_key == "model" else "channel"
        card = _model_card(_model_rationale(session)) if decision_key == "model" else _channel_card()
        reply = Turn(
            id=_new_id("t"),
            ts=_now_iso(),
            role="assistant",
            text="Sure — pick a different one:",
            cards=[card],
        )
        session.turns.append(reply)
        _save(session, path)
        return session


def mark_first_message(trace_id: str, summary: Optional[dict[str, Any]] = None) -> Session:
    """Phase C entry point — call this when Slack delivers the first
    real message. Advances to ``live`` and emits the trace preview.
    """
    path = _resolve_session_path()
    with _lock_for(path):
        session = _load(path) or _seed(path)
        if session.phase not in {"connect", "live"}:
            return session
        session.slack["installed"] = True
        session.slack["first_message_at"] = _now_iso()
        session.phase = "live"
        reply = Turn(
            id=_new_id("t"),
            ts=_now_iso(),
            role="assistant",
            text="🎉 Your first message just landed — the agent is live.",
            cards=[{
                "type": "trace_preview",
                "trace_id": trace_id,
                "summary": summary or {},
            }],
        )
        session.turns.append(reply)
        _save(session, path)
        return session


def reset() -> Session:
    """Wipe the session and start fresh. Used after onboarding
    completion or by an explicit "start over" affordance."""
    path = _resolve_session_path()
    with _lock_for(path):
        if path.is_file():
            try:
                path.unlink()
            except OSError as e:
                logger.warning("could not delete %s: %s", path, e)
        return _seed(path)


# ─── State-machine helpers ───────────────────────────────────────────


def _has_enough_intent(session: Session) -> bool:
    """Heuristic: at least one substantive user message describing
    purpose. Brain integration will eventually drive this, but a
    simple rule keeps the flow moving even when the brain is
    unavailable.
    """
    user_msgs = [t for t in session.turns if t.role == "user" and t.text.strip()]
    if not user_msgs:
        return False
    longest = max((len(t.text) for t in user_msgs), default=0)
    return longest >= 30  # one descriptive sentence


def _model_rationale(session: Session) -> str:
    """A one-liner the model card surfaces under the recommended row.
    Keep it grounded in what the user actually said — no fake stats.
    """
    purpose = (session.decisions.get("purpose") or "").lower()
    if "support" in purpose or "customer" in purpose or "ticket" in purpose:
        return "Plenty for customer-support volume; cheap and fast."
    return "Best balance of speed + cost for what you described."


def _materialize_agent(session: Session) -> None:
    """Create the agent + mint the right kind of bearer for the chosen channel.

    The Connect card needs to surface a real (masked) key the user can
    paste into the channel's setup screen. There are two token types
    in play:

    - ``otrcy_live_…`` — tenant-level bearer minted by ``runtime.tenants.tokens``.
      Used for Slack, WhatsApp, and Web widget connects (those channels
      authenticate via tenant context on the gateway).
    - ``ot_…``        — API-channel-specific token minted by
      ``runtime.agents.channels`` and persisted to
      ``agents/<id>/integrations/api.json``. The public ``/v1/api/{agent_id}/chat``
      endpoint validates against THIS token exclusively. Tenant
      bearers won't be accepted by the API channel.

    We mint based on ``session.decisions["channel"]`` so the card
    surfaces the right thing for the user to paste.
    """
    if session.agent_id is not None:
        return

    channel = (session.decisions or {}).get("channel", "slack")

    # Register the agent in the registry so /v1/api/<id>/chat and the
    # widget endpoints can resolve it. Without this step the connect
    # cards hand out tokens for a ghost id and every smoke-test returns
    # `agent_not_found`. The registry seeds the agent dir from _default
    # (or whatever the active seed is) so `agents/<id>/integrations/...`
    # writes a few lines below land in a real directory.
    purpose = (session.decisions or {}).get("purpose") or ""
    payload = {
        "name": purpose.strip() or "agent",
        "prompt": purpose.strip(),
        "model": (session.decisions or {}).get("model") or "claude-sonnet-4-6",
        "tools": [],
        "channels": [channel],
    }
    try:
        from runtime.agents.registry import create_agent
        meta = create_agent(payload)
        session.agent_id = meta.id
    except Exception as e:  # noqa: BLE001
        # Fall through with a random id — connect cards still surface,
        # but the channel won't actually route until the operator fixes
        # the registry issue (most likely a tenant-context problem).
        logger.warning("create_agent failed during onboarding: %s", e)
        session.agent_id = _new_id("agt")

    if channel == "api":
        _mint_api_channel_token(session)
    elif channel == "web":
        _provision_web_channel(session)
    else:
        _mint_tenant_token(session)


def _mint_tenant_token(session: Session) -> None:
    """Mint an ``otrcy_live_*`` tenant bearer for Slack / WhatsApp / Web."""
    try:
        from runtime.tenants.tokens import mint_token
        from runtime.tenant_context import get_active

        tenant_id = get_active()
        token, _record = mint_token(
            tenant_id,
            label=f"onboarding-{session.agent_id}",
        )
        if len(token) >= 16:
            head = token[:11]   # "otrcy_live_"
            tail = token[-4:]
            session.agent_key_preview = f"{head}••••••••••••{tail}"
        else:
            session.agent_key_preview = token
        # Carry the plaintext ONLY on the current connect card. The
        # session file persists the masked preview only — the plaintext
        # doesn't survive a page refresh.
        session.slack["agent_key_plaintext_once"] = token
    except Exception as e:  # noqa: BLE001
        logger.warning("could not mint tenant onboarding token: %s", e)
        session.agent_key_preview = "ot_live_••••••••••••••••"


def _provision_web_channel(session: Session) -> None:
    """Mint a widget_id + signing secret and persist to integrations/web.json.

    Mirrors POST /agents/<id>/channels/web/connect (P3.5) but in-process —
    onboarding is already inside the runtime, no need for the HTTP hop. The
    public /widget/<widget_id>/v1.js endpoint will then resolve back to
    this agent and serve the embed JS. allowed_domains stays empty so the
    operator can test on localhost first; they pin domains from the agent
    dashboard once the widget is on a real site.
    """
    try:
        import secrets as _secrets

        from runtime.agents.channels import _mask_token, save

        widget_id = "w_" + _secrets.token_hex(8)
        signing_secret = "whsec_" + _secrets.token_urlsafe(24)
        installed_at = _now_iso()
        save(session.agent_id, "web", {
            "widget_id": widget_id,
            "signing_secret": signing_secret,
            "allowed_domains": [],
            "settings": {
                "position": "br",
                "shape": "circle",
                "accent": "green",
                "greeting": "",
                "welcome": "",
                "fallback": "",
                "show_greeting": True,
                "require_email": False,
                "pill_label": "Chat",
            },
            "installed_at": installed_at,
        })
        # widget_id is public (it's in the embed snippet); the signing
        # secret is only needed for outbound webhooks — surface it once
        # so power users can grab it, but the masked preview is the
        # signing secret mask, not the widget_id.
        session.slack["widget_id"] = widget_id
        session.slack["signing_secret_plaintext_once"] = signing_secret
        session.agent_key_preview = _mask_token(signing_secret)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not provision web widget: %s", e)
        session.agent_key_preview = "whsec_••••••••••••••••"


def _mint_api_channel_token(session: Session) -> None:
    """Mint an ``ot_*`` API-channel token + persist to integrations/api.json.

    Bypasses the HTTP layer (no roundtrip to ``/agents/{id}/channels/api/connect``)
    because we're already in-process. We replicate the same on-disk
    shape the connect endpoint produces so subsequent calls into
    ``/v1/api/{agent_id}/chat`` validate cleanly.
    """
    try:
        import secrets as _secrets

        from runtime.agents.channels import _mask_token, save

        token = f"ot_{_secrets.token_urlsafe(32)}"
        created_at = _now_iso()
        save(session.agent_id, "api", {
            "token": token,
            "created_at": created_at,
            "last_used_at": None,
        })
        session.agent_key_preview = _mask_token(token)
        session.slack["agent_key_plaintext_once"] = token
    except Exception as e:  # noqa: BLE001
        logger.warning("could not mint API channel token: %s", e)
        session.agent_key_preview = "ot_••••••••••••••••"
