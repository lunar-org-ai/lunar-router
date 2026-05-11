"""routing technique — small_first stub + uniroute (P15.3.8) variants.

``small_first`` (default): always picks the small model. Stub left from
the original P1 scaffolding.

``uniroute`` (P15.3): reads ``versions/router_config_current``, calls
``UniRouteRouter.route(prompt)``, sets ``ctx.routing.model`` to the
selected model and ``ctx.routing.decision`` to the full RoutingDecision
dict (cluster_id, all_scores, cluster_probabilities, reasoning).

Cold-start: if no router_config exists yet, the variant falls back to
``knobs.small`` (or ``knobs.default`` when set) and stamps a
``fallback_reason`` so the UI can render "no fitted config" honestly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from runtime.protocols import BaseTechnique, Context, RoutingDecision, Stage


logger = logging.getLogger("techniques.routing")


class _SmallFirstStub:
    def __init__(self, knobs: dict[str, Any]) -> None:
        self.small: str = str(knobs.get("small", "claude-haiku-4-5"))
        self.big: str = str(knobs.get("big", "claude-sonnet-4-6"))
        self.threshold: float = float(knobs.get("confidence_threshold", 0.7))
        self.escalate_on_failure: bool = bool(knobs.get("escalate_on_failure", True))

    def execute(self, context: Context) -> Context:
        # Stub: always picks small. Real classifier comes later.
        context.routing = RoutingDecision(
            model=self.small,
            reason="stub: small_first defaults to small",
            confidence=0.9,
        )
        return context


class _UniRouteStage:
    """Routes via the trained UniRoute config (P15.3).

    Reads ``versions/router_config_current`` lazily on each call (cheap;
    config_io's symlink/marker resolution is filesystem-only). Cold-start
    falls back to the small / default knob.
    """

    def __init__(self, knobs: dict[str, Any]) -> None:
        # ``default`` wins over ``small`` when set; otherwise we use the
        # same small_first knob so existing route.yaml setups keep working.
        self.default_model: str = str(
            knobs.get("default") or knobs.get("small", "claude-haiku-4-5")
        )
        self.cost_weight_override: Optional[float] = (
            float(knobs["cost_weight"]) if "cost_weight" in knobs else None
        )

    def execute(self, context: Context) -> Context:
        from router.config_io import load_current_config
        from router.errors import (
            RouterColdStartError,
            RouterConfigInvalidError,
            RouterConfigNotFoundError,
        )
        from router.uniroute import UniRouteRouter
        from runtime.embedder_pool import get_pool

        try:
            assigner, registry, lam = load_current_config()
        except RouterConfigNotFoundError:
            context.routing = RoutingDecision(
                model=self.default_model,
                reason="cold_start: no fitted router_config yet",
                confidence=0.5,
                decision={
                    "selected_model": self.default_model,
                    "fallback_reason": "router_not_initialized",
                    "cold_start": True,
                },
            )
            return context
        except RouterConfigInvalidError as e:
            logger.warning("router_config invalid, falling back: %s", e)
            context.routing = RoutingDecision(
                model=self.default_model,
                reason=f"fallback: invalid router_config ({e})",
                confidence=0.3,
                decision={
                    "selected_model": self.default_model,
                    "fallback_reason": f"router_config_invalid: {e}",
                    "cold_start": False,
                },
            )
            return context

        try:
            embedder = get_pool().get()
            router = UniRouteRouter(
                embedder=embedder,
                cluster_assigner=assigner,
                registry=registry,
                cost_weight=lam,
            )
        except RouterColdStartError as e:
            context.routing = RoutingDecision(
                model=self.default_model,
                reason=f"cold_start: {e}",
                confidence=0.5,
                decision={
                    "selected_model": self.default_model,
                    "fallback_reason": f"cold_start: {e}",
                    "cold_start": True,
                },
            )
            return context

        try:
            decision = router.route(
                context.request,
                cost_weight_override=self.cost_weight_override,
            )
        except Exception as e:
            logger.warning("uniroute decision failed, falling back: %s", e)
            context.routing = RoutingDecision(
                model=self.default_model,
                reason=f"fallback: route() raised {type(e).__name__}",
                confidence=0.3,
                decision={
                    "selected_model": self.default_model,
                    "fallback_reason": f"route_error: {type(e).__name__}: {e}",
                    "cold_start": False,
                },
            )
            return context

        context.routing = RoutingDecision(
            model=decision.selected_model,
            reason=decision.reasoning or "uniroute",
            confidence=float(decision.cluster_probabilities[decision.cluster_id]),
            decision={
                **decision.to_dict(),
                "fallback_reason": None,
                "cold_start": False,
            },
        )
        return context


class RoutingTechnique(BaseTechnique):
    name = "routing"
    variants = ("small_first", "uniroute")

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(
                f"routing: unknown variant {variant!r}; expected one of {self.variants}"
            )
        if variant == "uniroute":
            return _UniRouteStage(knobs)
        return _SmallFirstStub(knobs)
