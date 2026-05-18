"""Stripe checkout + webhook glue (P17.1).

Two surfaces:

  * :func:`create_checkout_session` — server-to-server call that mints a
    Stripe Checkout session for a tenant upgrading to a paid tier.
    Returns the redirect URL the UI sends the browser to.

  * :func:`handle_webhook_event` — verifies the Stripe signature and
    flips the tenant's tier when a subscription becomes active / gets
    canceled.

Environment
-----------

  * ``STRIPE_SECRET_KEY``       — sk_test_… or sk_live_… (required)
  * ``STRIPE_WEBHOOK_SECRET``   — whsec_… (required for webhook)
  * ``STRIPE_PRICE_STARTER``    — price_… for the Starter monthly plan
  * ``STRIPE_PRICE_TEAM``       — price_… for the Team monthly plan
  * ``STRIPE_CHECKOUT_SUCCESS_URL`` — defaults to ``/billing?status=success``
  * ``STRIPE_CHECKOUT_CANCEL_URL``  — defaults to ``/billing?status=cancel``

The ``stripe`` Python package is imported lazily so OSS / single-tenant
deploys don't have to install it.

Tenant ↔ Stripe linking
-----------------------

We pass ``client_reference_id=<tenant_id>`` to the Checkout session and
mirror ``tenant_id`` into the subscription metadata. The webhook reads
that field to know which tenant to upgrade — Stripe customer IDs would
work too, but ``client_reference_id`` is set unconditionally on every
session, so the upgrade path doesn't depend on a prior customer record.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from runtime.tenants import billing


logger = logging.getLogger("runtime.tenants.stripe")


# Map tier slug → env var holding the matching Stripe price id. Scale is
# absent because it's contact-sales (no self-serve checkout).
_PRICE_ENV = {
    "starter": "STRIPE_PRICE_STARTER",
    "team": "STRIPE_PRICE_TEAM",
}


class StripeNotConfigured(RuntimeError):
    """Raised when a Stripe endpoint is hit without the env vars set.
    Endpoint handlers map this to 503 so the operator gets a precise
    error rather than a generic 500 from a missing key.
    """


def _require_secret_key() -> str:
    key = (os.environ.get("STRIPE_SECRET_KEY") or "").strip()
    if not key:
        raise StripeNotConfigured("STRIPE_SECRET_KEY is not set")
    return key


def _stripe():
    """Lazy import. Raises StripeNotConfigured (with a useful hint) when
    the package isn't installed — typical OSS deploys won't have it."""
    try:
        import stripe  # type: ignore[import-not-found]
    except ImportError as e:
        raise StripeNotConfigured(
            "the `stripe` package is not installed — `pip install stripe` "
            "to enable checkout"
        ) from e
    stripe.api_key = _require_secret_key()
    return stripe


def _price_id_for(tier: str) -> str:
    env_var = _PRICE_ENV.get(tier)
    if env_var is None:
        raise StripeNotConfigured(f"tier {tier!r} has no self-serve checkout")
    price = (os.environ.get(env_var) or "").strip()
    if not price:
        raise StripeNotConfigured(f"{env_var} is not set")
    return price


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------


def create_checkout_session(
    tenant_id: str,
    tier: str,
    *,
    success_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
) -> str:
    """Mint a Stripe Checkout session for ``tenant_id`` to subscribe to
    ``tier``. Returns the URL the UI should redirect to.

    The session carries ``client_reference_id=tenant_id`` and a
    ``metadata.tier`` field; the webhook resolves both back when the
    subscription transitions to ``active``.
    """
    if tier not in _PRICE_ENV:
        raise ValueError(f"tier {tier!r} not eligible for self-serve checkout")
    stripe = _stripe()
    price = _price_id_for(tier)

    state = billing.load(tenant_id)
    success = success_url or os.environ.get(
        "STRIPE_CHECKOUT_SUCCESS_URL",
        "/billing?status=success",
    )
    cancel = cancel_url or os.environ.get(
        "STRIPE_CHECKOUT_CANCEL_URL",
        "/billing?status=cancel",
    )

    kwargs: dict[str, Any] = {
        "mode": "subscription",
        "line_items": [{"price": price, "quantity": 1}],
        "success_url": success,
        "cancel_url": cancel,
        "client_reference_id": tenant_id,
        "metadata": {"tenant_id": tenant_id, "tier": tier},
        "subscription_data": {
            "metadata": {"tenant_id": tenant_id, "tier": tier},
        },
        # Allow promo codes — tiny addition that often nudges trial users
        # over the line without our involvement.
        "allow_promotion_codes": True,
    }
    # Re-use the customer when we already have one on file so Stripe
    # doesn't double-charge a customer who upgrades twice.
    if state.stripe_customer_id:
        kwargs["customer"] = state.stripe_customer_id

    session = stripe.checkout.Session.create(**kwargs)
    return str(session.url)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


_HANDLED_EVENTS = {
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
}


def handle_webhook_event(payload: bytes, signature: str) -> dict[str, Any]:
    """Verify the signature, dispatch by event type, return a small
    diagnostic dict the endpoint logs. Idempotent: replaying the same
    event twice lands the tier in the same state."""
    stripe = _stripe()
    secret = (os.environ.get("STRIPE_WEBHOOK_SECRET") or "").strip()
    if not secret:
        raise StripeNotConfigured("STRIPE_WEBHOOK_SECRET is not set")

    try:
        event = stripe.Webhook.construct_event(payload, signature, secret)
    except Exception as e:  # noqa: BLE001 — Stripe raises SignatureVerificationError
        logger.warning("stripe webhook signature rejected: %s", e)
        raise

    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", "")
    if event_type not in _HANDLED_EVENTS:
        return {"ignored": True, "type": event_type}

    data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event.data.object
    obj = dict(data) if not isinstance(data, dict) else data
    tenant_id = _extract_tenant_id(obj)
    if not tenant_id:
        logger.warning(
            "stripe webhook %s arrived without a tenant_id reference — dropping",
            event_type,
        )
        return {"ignored": True, "reason": "no_tenant_id"}

    if event_type == "customer.subscription.deleted":
        billing.set_tier(tenant_id, "free")
        return {"handled": True, "type": event_type, "tier": "free", "tenant_id": tenant_id}

    # checkout.session.completed and subscription.created/updated all
    # end with the tenant on the tier their subscription points to.
    tier = _extract_tier(obj)
    customer_id = obj.get("customer") if isinstance(obj.get("customer"), str) else None
    subscription_id = (
        obj.get("subscription") if isinstance(obj.get("subscription"), str) else obj.get("id")
    )

    if tier:
        billing.set_tier(
            tenant_id,
            tier,
            stripe_customer_id=customer_id,
            stripe_subscription_id=str(subscription_id) if subscription_id else None,
        )
        return {"handled": True, "type": event_type, "tier": tier, "tenant_id": tenant_id}

    # Subscription updated to a price we don't recognize — log and skip.
    logger.warning(
        "stripe webhook %s for tenant %s had no resolvable tier; ignoring",
        event_type, tenant_id,
    )
    return {"ignored": True, "reason": "no_tier_in_metadata"}


def _extract_tenant_id(obj: dict[str, Any]) -> Optional[str]:
    """Resolve the tenant id from any of the places we stash it on
    Checkout sessions / subscriptions."""
    metadata = obj.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("tenant_id"):
        return str(metadata["tenant_id"])
    ref = obj.get("client_reference_id")
    if ref:
        return str(ref)
    # Subscription update events embed the subscription in
    # ``data.object`` directly; the tenant_id lives in its own metadata,
    # already covered above. Customer-side lookup is intentionally NOT
    # done here — the webhook would have to round-trip Stripe to fetch
    # the customer; we just require the metadata round-trip happened
    # at subscription creation (which our checkout flow guarantees).
    return None


def _extract_tier(obj: dict[str, Any]) -> Optional[str]:
    """Resolve the tier from metadata.tier or, failing that, by matching
    the price id back to one of our known tier price envs."""
    metadata = obj.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("tier"):
        return str(metadata["tier"])
    # Subscription objects expose price under items.data[0].price.id.
    items = ((obj.get("items") or {}).get("data") or []) if isinstance(obj.get("items"), dict) else []
    for item in items:
        price = (item.get("price") or {}) if isinstance(item, dict) else {}
        price_id = price.get("id") if isinstance(price, dict) else None
        if not price_id:
            continue
        for tier, env_var in _PRICE_ENV.items():
            if (os.environ.get(env_var) or "").strip() == price_id:
                return tier
    return None
