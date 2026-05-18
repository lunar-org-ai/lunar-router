"""Per-tenant billing state + tier limits (P17.1).

Each tenant has a ``tenants/<tid>/billing.json`` document with:

  * ``tier`` — one of ``free`` / ``starter`` / ``team`` / ``scale``
  * ``period`` — ``YYYY-MM`` rollover marker; counters reset when the
    current month overtakes this.
  * ``counters`` — monthly usage (``traces``, ``evolutions``).
  * ``updated_at`` — ISO timestamp of the last mutation.

Design:

  * **BYOK-aware**: LLM cost is on the customer; we charge for the
    platform (AHE loop, evals, dataset curation, MCP gateway, infra).
    Limits cap *our* GCP spend (Cloud Run + GCS + KMS), not LLM tokens.
  * **Free tier is a funnel**: 1 agent, 1k traces/mo, 7-day retention,
    NO evolution, NO hosted MCP. Generous enough to feel real value,
    gated enough that the killer features (AHE + scale) need Starter.
  * **Lazy reset**: We don't run a cron. Every mutation calls
    ``_ensure_current_period(state)`` which zero-fills counters when the
    month rolls over. Idempotent.
  * **OSS local mode**: When multi-tenant is disabled, all gates are
    bypassed at the call-site — this module still works, but tenants
    don't exist, so it never runs. Single-tenant deployments stay
    unbilled by construction.

Stripe webhooks (future) only ever flip ``tier`` — the rest of this
module is pure local state, so the integration surface is tiny.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("runtime.tenants.billing")


_FILENAME = "billing.json"


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


# Ordered: each later tier strictly supersedes earlier ones. Used by
# ``tier_at_least()`` for feature gates ("evolution requires >= starter").
TIERS = ("free", "starter", "team", "scale")


@dataclass(frozen=True)
class TierLimits:
    """Caps that apply to a single tier.

    ``-1`` means unlimited. ``allow_evolution`` and ``allow_hosted_mcp``
    are explicit booleans rather than counts because they're binary
    feature gates, not per-unit billing.
    """

    name: str
    monthly_traces: int          # -1 = unlimited
    max_agents: int              # -1 = unlimited
    max_integrations_per_agent: int
    allow_evolution: bool        # AHE loop access
    allow_hosted_mcp: bool       # sse / http MCP transports
    retention_days: int          # trace retention; -1 = unlimited
    rate_limit_per_minute: int   # /run rate cap


# Numbers chosen to match the pricing discussion: free is a funnel,
# starter unlocks AHE, team scales up agents, scale removes caps.
_TIER_LIMITS: dict[str, TierLimits] = {
    "free": TierLimits(
        name="free",
        monthly_traces=1_000,
        max_agents=1,
        max_integrations_per_agent=2,
        allow_evolution=False,
        allow_hosted_mcp=False,
        retention_days=7,
        rate_limit_per_minute=60,
    ),
    "starter": TierLimits(
        name="starter",
        monthly_traces=10_000,
        max_agents=1,
        max_integrations_per_agent=10,
        allow_evolution=True,
        allow_hosted_mcp=True,
        retention_days=30,
        rate_limit_per_minute=300,
    ),
    "team": TierLimits(
        name="team",
        monthly_traces=100_000,
        max_agents=5,
        max_integrations_per_agent=-1,
        allow_evolution=True,
        allow_hosted_mcp=True,
        retention_days=90,
        rate_limit_per_minute=1_200,
    ),
    "scale": TierLimits(
        name="scale",
        monthly_traces=-1,
        max_agents=-1,
        max_integrations_per_agent=-1,
        allow_evolution=True,
        allow_hosted_mcp=True,
        retention_days=-1,
        rate_limit_per_minute=-1,
    ),
}


def limits_for(tier: str) -> TierLimits:
    """Return the limits for ``tier`` — falls back to free if unknown."""
    return _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])


def tier_at_least(tier: str, required: str) -> bool:
    """``True`` when ``tier`` is at or above ``required`` in the
    free→starter→team→scale order."""
    try:
        return TIERS.index(tier) >= TIERS.index(required)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class BillingState:
    """On-disk shape of ``tenants/<tid>/billing.json``."""

    tier: str = "free"
    period: str = ""             # "YYYY-MM"
    traces: int = 0
    evolutions: int = 0
    updated_at: str = ""
    # Stripe wiring lives here when a tenant upgrades. None on free.
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "period": self.period,
            "counters": {"traces": self.traces, "evolutions": self.evolutions},
            "updated_at": self.updated_at,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BillingState":
        counters = data.get("counters") or {}
        return cls(
            tier=str(data.get("tier") or "free"),
            period=str(data.get("period") or ""),
            traces=int(counters.get("traces") or 0),
            evolutions=int(counters.get("evolutions") or 0),
            updated_at=str(data.get("updated_at") or ""),
            stripe_customer_id=data.get("stripe_customer_id"),
            stripe_subscription_id=data.get("stripe_subscription_id"),
        )


# ---------------------------------------------------------------------------
# Path resolution + locks
# ---------------------------------------------------------------------------


def _path(tenant_id: str, *, root: Optional[Path] = None) -> Path:
    from runtime.tenants.registry import get_tenant_dir
    return get_tenant_dir(tenant_id, root=root) / _FILENAME


# Per-tenant lock so concurrent increments inside one worker don't lose
# counts. Cross-instance races on Cloud Run are accepted — undercounting
# by a few percent is fine; overshooting limits by <1% won't sink margins.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(tenant_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(tenant_id)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[tenant_id] = lock
        return lock


# ---------------------------------------------------------------------------
# Period bookkeeping
# ---------------------------------------------------------------------------


def _current_period(now: Optional[datetime] = None) -> str:
    n = now or datetime.now(timezone.utc)
    return f"{n.year:04d}-{n.month:02d}"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _ensure_current_period(state: BillingState) -> bool:
    """Reset monthly counters if the period rolled over. Returns True
    when a reset happened (caller may want to persist)."""
    current = _current_period()
    if state.period == current:
        return False
    state.period = current
    state.traces = 0
    state.evolutions = 0
    return True


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def load(tenant_id: str, *, root: Optional[Path] = None) -> BillingState:
    """Return the tenant's billing state. Defaults to a free tier with
    zeroed counters when the file is missing — never raises on a
    non-existent tenant dir, since the first /run for a new tenant must
    not 500 just because nobody wrote billing.json yet.
    """
    path = _path(tenant_id, root=root)
    if not path.is_file():
        state = BillingState(tier="free", period=_current_period(), updated_at=_now_iso())
        return state
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return BillingState.from_dict(data)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("billing.json unreadable at %s (%s) — falling back to free", path, e)
        return BillingState(tier="free", period=_current_period(), updated_at=_now_iso())


def save(tenant_id: str, state: BillingState, *, root: Optional[Path] = None) -> Path:
    """Atomic write. Caller already holds the per-tenant lock when
    increment helpers call this."""
    path = _path(tenant_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = _now_iso()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def set_tier(
    tenant_id: str,
    tier: str,
    *,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    root: Optional[Path] = None,
) -> BillingState:
    """Flip a tenant to a new tier. Used by the Stripe webhook handler
    and by operator tooling. Resets the period marker so the new tier's
    quotas apply from now (otherwise an upgrade mid-month could leave a
    rolled-over `period` stuck at the prior month)."""
    if tier not in TIERS:
        raise ValueError(f"unknown_tier: {tier}")
    with _lock_for(tenant_id):
        state = load(tenant_id, root=root)
        state.tier = tier
        _ensure_current_period(state)
        if stripe_customer_id is not None:
            state.stripe_customer_id = stripe_customer_id
        if stripe_subscription_id is not None:
            state.stripe_subscription_id = stripe_subscription_id
        save(tenant_id, state, root=root)
        return state


def increment_traces(tenant_id: str, n: int = 1, *, root: Optional[Path] = None) -> BillingState:
    """Bump the monthly trace counter. Quota check is the caller's
    responsibility — this is the post-success bookkeeping step."""
    with _lock_for(tenant_id):
        state = load(tenant_id, root=root)
        _ensure_current_period(state)
        state.traces += n
        save(tenant_id, state, root=root)
        return state


def increment_evolutions(tenant_id: str, n: int = 1, *, root: Optional[Path] = None) -> BillingState:
    """Bump the monthly AHE evolution counter."""
    with _lock_for(tenant_id):
        state = load(tenant_id, root=root)
        _ensure_current_period(state)
        state.evolutions += n
        save(tenant_id, state, root=root)
        return state


# ---------------------------------------------------------------------------
# Read-only views (UI / billing endpoint)
# ---------------------------------------------------------------------------


def snapshot(tenant_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    """Combined view for the /tenant/billing endpoint. Includes the
    tier limits inline so the UI can render usage bars without a
    second round-trip.
    """
    state = load(tenant_id, root=root)
    # Don't mutate the disk on a pure read — but DO reflect the current
    # period in the view, so the UI shows zeroed counters the second the
    # month flips even if no trace has hit yet.
    if _ensure_current_period(state):
        pass
    lim = limits_for(state.tier)
    return {
        "tier": state.tier,
        "period": state.period,
        "updated_at": state.updated_at,
        "usage": {
            "traces": state.traces,
            "evolutions": state.evolutions,
        },
        "limits": {
            "monthly_traces": lim.monthly_traces,
            "max_agents": lim.max_agents,
            "max_integrations_per_agent": lim.max_integrations_per_agent,
            "allow_evolution": lim.allow_evolution,
            "allow_hosted_mcp": lim.allow_hosted_mcp,
            "retention_days": lim.retention_days,
            "rate_limit_per_minute": lim.rate_limit_per_minute,
        },
        "stripe_customer_id": state.stripe_customer_id,
        "stripe_subscription_id": state.stripe_subscription_id,
    }
