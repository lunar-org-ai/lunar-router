"""Tenant quota enforcement (P17.1).

This module wraps :mod:`runtime.tenants.billing` with the *decision*
layer the API endpoints care about: "is this tenant allowed to do X
right now?". All checks are no-ops when multi-tenant mode is off — the
OSS single-tenant deployment never sees these gates.

Conventions
-----------

  * Each ``check_*`` helper returns ``None`` when allowed, or raises
    :class:`QuotaExceeded` with a structured ``detail`` payload. The
    server's exception handler maps this to HTTP 402 (Payment Required)
    so the UI can render an upgrade CTA from one well-typed error
    instead of parsing strings.

  * Counter increments are NOT part of the check — they happen in the
    success path of the endpoint, after the work succeeded. The
    rationale: a /run that 500'd on the LLM didn't consume a billable
    trace, so it shouldn't count.

  * ``check_*`` reads tenant state but does NOT write. The only writes
    happen in ``increment_*`` calls back in ``billing``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from runtime.tenants import billing


# ---------------------------------------------------------------------------
# Exception type
# ---------------------------------------------------------------------------


@dataclass
class QuotaExceeded(Exception):
    """Raised when a tenant action is denied by their tier limits.

    ``detail`` is the structured payload that should land in the HTTP
    response body so the UI can react without string-matching.
    """

    code: str            # short stable identifier ("trace_quota_exceeded")
    message: str         # human-readable, shown in the upgrade banner
    detail: dict[str, Any]

    def __post_init__(self) -> None:
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Active-tenant resolution
# ---------------------------------------------------------------------------


def _active_tenant() -> Optional[str]:
    """Return the currently-active tenant id, or ``None`` if we're
    running OSS local mode (no multi-tenancy). All gates short-circuit
    to "allowed" when this is ``None``.
    """
    try:
        from runtime.tenants.feature import is_multi_tenant_enabled
        if not is_multi_tenant_enabled():
            return None
        from runtime.tenant_context import get_active
        tid = get_active()
        return tid or None
    except Exception:
        return None


def _exceeded(used: int, cap: int) -> bool:
    """Cap convention: ``-1`` means unlimited; otherwise ``used >= cap``
    is exceeded. Use ``>=`` not ``>`` so the cap is the limit, not the
    first denial value (1000-trace cap means trace #1001 is denied)."""
    return cap >= 0 and used >= cap


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def check_trace_quota() -> None:
    """Called BEFORE accepting a /run that will produce a trace.

    Free tier: 1k traces/mo. Anything above raises. Starter / team have
    larger caps; scale is uncapped.
    """
    tid = _active_tenant()
    if tid is None:
        return
    state = billing.load(tid)
    limits = billing.limits_for(state.tier)
    if _exceeded(state.traces, limits.monthly_traces):
        raise QuotaExceeded(
            code="trace_quota_exceeded",
            message=(
                f"Monthly trace limit reached ({limits.monthly_traces:,}) on "
                f"the {state.tier} tier. Upgrade to keep recording."
            ),
            detail={
                "tier": state.tier,
                "used": state.traces,
                "limit": limits.monthly_traces,
                "period": state.period,
            },
        )


def check_agent_count(*, current_count: int) -> None:
    """Called BEFORE creating a new agent. ``current_count`` is the
    number of agents the tenant already has (caller has it cheap because
    they just listed them)."""
    tid = _active_tenant()
    if tid is None:
        return
    state = billing.load(tid)
    limits = billing.limits_for(state.tier)
    if _exceeded(current_count, limits.max_agents):
        raise QuotaExceeded(
            code="agent_quota_exceeded",
            message=(
                f"You're at the {limits.max_agents}-agent limit on the "
                f"{state.tier} tier. Upgrade to add more."
            ),
            detail={
                "tier": state.tier,
                "used": current_count,
                "limit": limits.max_agents,
            },
        )


def check_evolution_allowed() -> None:
    """Called BEFORE kicking off an AHE evolution rollout. Free tier
    does NOT include evolution; Starter+ does.

    This is the killer paywall: AHE is the differentiator and the most
    expensive feature to run, so we want every free user to see "your
    agent could improve here — upgrade to run" rather than silently
    serve them.
    """
    tid = _active_tenant()
    if tid is None:
        return
    state = billing.load(tid)
    limits = billing.limits_for(state.tier)
    if not limits.allow_evolution:
        raise QuotaExceeded(
            code="evolution_not_in_tier",
            message=(
                "Evolution loop is a Starter+ feature. Upgrade to let "
                "your agent self-improve from production traces."
            ),
            detail={
                "tier": state.tier,
                "required_tier": "starter",
            },
        )


def check_integration_allowed(
    *,
    transport: str,
    current_count: int,
) -> None:
    """Called BEFORE adding an MCP integration. Two gates:

      * **Hosted MCP (sse/http)** is a Starter+ feature. Free tenants
        can still configure ``stdio`` (subprocess) MCPs because those
        run in their own environment, not ours.
      * **Per-agent count** is capped by tier.
    """
    tid = _active_tenant()
    if tid is None:
        return
    state = billing.load(tid)
    limits = billing.limits_for(state.tier)

    t = (transport or "stdio").lower()
    if t in ("sse", "http") and not limits.allow_hosted_mcp:
        raise QuotaExceeded(
            code="hosted_mcp_not_in_tier",
            message=(
                f"{t.upper()} MCP integrations are a Starter+ feature. "
                "Free tier supports local stdio servers only."
            ),
            detail={
                "tier": state.tier,
                "required_tier": "starter",
                "transport": t,
            },
        )

    if _exceeded(current_count, limits.max_integrations_per_agent):
        raise QuotaExceeded(
            code="integration_quota_exceeded",
            message=(
                f"You're at the {limits.max_integrations_per_agent}-"
                f"integration limit per agent on the {state.tier} tier."
            ),
            detail={
                "tier": state.tier,
                "used": current_count,
                "limit": limits.max_integrations_per_agent,
            },
        )


# ---------------------------------------------------------------------------
# Convenience: trip the gate AND increment in one call
# ---------------------------------------------------------------------------


def consume_trace() -> None:
    """Post-success bookkeeping — increment after a /run succeeded.
    No-op in OSS mode. Safe to call from a thread."""
    tid = _active_tenant()
    if tid is None:
        return
    billing.increment_traces(tid)


def consume_evolution() -> None:
    """Post-success bookkeeping after an AHE rollout."""
    tid = _active_tenant()
    if tid is None:
        return
    billing.increment_evolutions(tid)


# ---------------------------------------------------------------------------
# Retention window helper (used by /traces queries to hide stale data
# on free tier without deleting from GCS — simpler + cheaper)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rate limit (sliding 60s window, in-memory per tenant)
#
# Single-instance only — Cloud Run with N replicas effectively gives each
# tenant N × rate_limit_per_minute headroom. Acceptable for v1: the cap
# exists to stop runaway scripts, not to enforce penny-precise SLAs. A
# Redis-backed limiter is the upgrade path when we hit that ceiling.
# ---------------------------------------------------------------------------


_RATE_WINDOW_S = 60.0
_rate_buckets: dict[str, deque] = {}
_rate_lock = threading.Lock()


def check_rate_limit() -> None:
    """Sliding-window rate check + record. Raises QuotaExceeded when the
    tenant exceeded their tier's per-minute cap on /run.

    Records the call in the same step — there's no separate ``consume``
    helper because rate limiting is about *attempts*, not successes
    (otherwise a buggy client looping on errors could escape the cap).
    """
    tid = _active_tenant()
    if tid is None:
        return
    state = billing.load(tid)
    limits = billing.limits_for(state.tier)
    cap = limits.rate_limit_per_minute
    if cap < 0:
        return

    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_S
    with _rate_lock:
        bucket = _rate_buckets.get(tid)
        if bucket is None:
            bucket = deque()
            _rate_buckets[tid] = bucket
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= cap:
            # Retry-after = seconds until the oldest call falls out.
            retry_after = max(1, int(bucket[0] + _RATE_WINDOW_S - now) + 1)
            raise QuotaExceeded(
                code="rate_limit_exceeded",
                message=(
                    f"Rate limit of {cap} requests/minute on the "
                    f"{state.tier} tier was hit. Try again shortly."
                ),
                detail={
                    "tier": state.tier,
                    "limit_per_minute": cap,
                    "retry_after_s": retry_after,
                },
            )
        bucket.append(now)


def retention_cutoff_iso() -> Optional[str]:
    """Return an ISO timestamp; traces older than this should not be
    returned to the tenant. ``None`` means no cutoff (OSS mode, or a
    tier with unlimited retention)."""
    tid = _active_tenant()
    if tid is None:
        return None
    state = billing.load(tid)
    limits = billing.limits_for(state.tier)
    if limits.retention_days < 0:
        return None
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=limits.retention_days)
    return cutoff.isoformat(timespec="seconds").replace("+00:00", "Z")
