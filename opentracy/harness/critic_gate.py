"""Single chokepoint for critic-gated write paths in the harness.

Every harness mutation that costs money or moves real state should flow
through `critic_check`. The critic (`budget_justifier` agent) decides
approve/reject; this module materializes that decision as a `decision`
ledger row so the trail is auditable, then returns the verdict to the
caller. Callers do not write the decision row themselves — keeping the
write here is the only way to guarantee the audit log can never diverge
from the gating logic.

The critic itself costs money on every call (it's a real LLM call), so
we cache verdicts: the same (action_kind, payload, objective_id) inside
a one-hour bucket reuses the prior verdict and writes a `cached_verdict`
ledger row so the audit chain is still complete. Without the cache, an
over-eager Claude Code session calling `propose_action` 100 times in a
minute would trigger 100 critic LLM calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .ledger import LedgerEntry, LedgerStore, get_ledger_store

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS = 600  # 10 minutes
CACHE_MAX_ENTRIES = 256


class _RunnerProtocol(Protocol):
    async def run(self, agent_name: str, user_input: str) -> Any: ...


@dataclass
class CriticVerdict:
    """The output of one critic_check call.

    `decision_entry_id` points at the ledger row this verdict was
    written as, so callers can chain follow-on entries (`run`,
    `proposal`) off it via parent_id.
    """

    decision: str  # "approve" | "reject"
    rationale: str
    estimated_cost_usd: float
    estimated_benefit: str
    decision_entry_id: str

    @property
    def approved(self) -> bool:
        return self.decision == "approve"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "rationale": self.rationale,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_benefit": self.estimated_benefit,
            "decision_entry_id": self.decision_entry_id,
        }


@dataclass
class _CachedVerdict:
    decision: str
    rationale: str
    estimated_cost_usd: float
    estimated_benefit: str
    expires_at: float


# Process-local LRU keyed by (action_kind, payload_hash, objective_id, hour_bucket).
# Tests pass their own dict via the `cache` arg so they never see this.
_VERDICT_CACHE: "dict[str, _CachedVerdict]" = {}
_CACHE_LOCK = asyncio.Lock()


def _payload_hash(payload: Any) -> str:
    """Deterministic hash of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_key(action_kind: str, payload: Any, objective_id: Optional[str]) -> str:
    bucket = int(time.time() // 3600)
    return f"{action_kind}:{_payload_hash(payload)}:{objective_id or '-'}:{bucket}"


def _gc_cache(cache: dict[str, _CachedVerdict]) -> None:
    """Trim expired and oldest entries. Called inline; no background task."""
    now = time.time()
    expired = [k for k, v in cache.items() if v.expires_at <= now]
    for k in expired:
        cache.pop(k, None)
    if len(cache) > CACHE_MAX_ENTRIES:
        # Drop the oldest (smallest expires_at) until we're back at cap.
        ordered = sorted(cache.items(), key=lambda kv: kv[1].expires_at)
        for k, _ in ordered[: len(cache) - CACHE_MAX_ENTRIES]:
            cache.pop(k, None)


def _build_context(
    *,
    action_kind: str,
    payload: dict[str, Any],
    objective_id: Optional[str],
    ledger: LedgerStore,
) -> dict[str, Any]:
    """Build the JSON context the critic agent reads.

    Keeps shape small: critic prompt is tuned for ~400 tokens output and
    we don't want to blow the input window on a kitchen-sink dump.
    """
    objective_summary: dict[str, Any] = {"id": objective_id}
    recent_actions: list[dict[str, Any]] = []

    if objective_id:
        # Pull the last 24h of measurements so the critic sees the trend.
        from datetime import datetime, timedelta, timezone

        start_iso = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        ts_entries = ledger.time_series(objective_id, start=start_iso, limit=200)
        measurements = [
            {"ts": e.ts, "value": e.data.get("value")}
            for e in ts_entries
            if e.type == "observation" and "measurement" in e.tags and e.data.get("value") is not None
        ]
        objective_summary["recent_measurements"] = measurements[-10:]
        # Recent actions for this objective in the same window.
        recent_actions = [
            {"ts": e.ts, "kind": e.data.get("kind"), "outcome": e.outcome}
            for e in ts_entries
            if e.type == "action"
        ]

    return {
        "objective": objective_summary,
        "proposal": {"kind": action_kind, "payload": payload},
        "recent_actions": {
            "window_hours": 24,
            "count": len(recent_actions),
            "items": recent_actions[-10:],
        },
    }


def _parse_verdict(raw: dict[str, Any]) -> tuple[str, str, float, str]:
    decision = str(raw.get("decision", "")).strip().lower()
    if decision not in {"approve", "reject"}:
        # Degraded LLM output: default to reject. The ledger will record
        # the original raw text so an operator can see what happened.
        decision = "reject"
    rationale = str(raw.get("rationale", ""))
    try:
        cost = float(raw.get("estimated_cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    benefit = str(raw.get("estimated_benefit", ""))
    return decision, rationale, cost, benefit


async def critic_check(
    *,
    action_kind: str,
    payload: dict[str, Any],
    objective_id: Optional[str],
    ledger: Optional[LedgerStore] = None,
    runner: Optional[_RunnerProtocol] = None,
    cache: Optional[dict[str, _CachedVerdict]] = None,
) -> CriticVerdict:
    """Gate one harness write through the budget critic.

    Returns the verdict; never raises on `reject`. Always writes a
    `decision` ledger row before returning so the audit chain is intact
    regardless of cache hit/miss.
    """
    store = ledger if ledger is not None else get_ledger_store()
    cache_dict = _VERDICT_CACHE if cache is None else cache
    key = _cache_key(action_kind, payload, objective_id)

    cached: Optional[_CachedVerdict] = None
    async with _CACHE_LOCK:
        _gc_cache(cache_dict)
        hit = cache_dict.get(key)
        if hit and hit.expires_at > time.time():
            cached = hit

    if cached is not None:
        entry = LedgerEntry(
            type="decision",
            objective_id=objective_id,
            agent="budget_justifier",
            data={
                "decision": cached.decision,
                "rationale": cached.rationale,
                "estimated_cost_usd": cached.estimated_cost_usd,
                "estimated_benefit": cached.estimated_benefit,
                "action_kind": action_kind,
            },
            tags=["critic_check", action_kind, cached.decision, "cached_verdict"],
            cost_usd=0.0,
            outcome="ok",
        )
        store.append(entry)
        return CriticVerdict(
            decision=cached.decision,
            rationale=cached.rationale,
            estimated_cost_usd=cached.estimated_cost_usd,
            estimated_benefit=cached.estimated_benefit,
            decision_entry_id=entry.id,
        )

    # Cache miss → real critic call.
    if runner is None:
        from .runner import AgentRunner

        runner = AgentRunner()

    context = _build_context(
        action_kind=action_kind,
        payload=payload,
        objective_id=objective_id,
        ledger=store,
    )

    try:
        result = await runner.run("budget_justifier", json.dumps(context, default=str))
        raw = getattr(result, "data", result if isinstance(result, dict) else {})
    except Exception as exc:
        logger.warning("budget_justifier call failed (%s); defaulting to reject", exc)
        raw = {
            "decision": "reject",
            "rationale": f"critic unavailable: {exc}",
            "estimated_cost_usd": 0.0,
            "estimated_benefit": "",
        }

    decision, rationale, cost, benefit = _parse_verdict(raw if isinstance(raw, dict) else {})

    entry = LedgerEntry(
        type="decision",
        objective_id=objective_id,
        agent="budget_justifier",
        data={
            "decision": decision,
            "rationale": rationale,
            "estimated_cost_usd": cost,
            "estimated_benefit": benefit,
            "action_kind": action_kind,
        },
        tags=["critic_check", action_kind, decision],
        outcome="ok",
    )
    store.append(entry)

    async with _CACHE_LOCK:
        cache_dict[key] = _CachedVerdict(
            decision=decision,
            rationale=rationale,
            estimated_cost_usd=cost,
            estimated_benefit=benefit,
            expires_at=time.time() + CACHE_TTL_SECONDS,
        )

    return CriticVerdict(
        decision=decision,
        rationale=rationale,
        estimated_cost_usd=cost,
        estimated_benefit=benefit,
        decision_entry_id=entry.id,
    )


def reset_cache_for_tests() -> None:
    """Drop every cached verdict — used by tests to isolate runs."""
    _VERDICT_CACHE.clear()
