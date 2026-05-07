"""routing technique — STUB implementations.

Phase 1.4 stub: small_first always picks the small model with high confidence.
Real routing (eg. a classifier deciding small vs big) lands in a later phase.
"""

from __future__ import annotations

from typing import Any

from runtime.protocols import BaseTechnique, Context, RoutingDecision, Stage


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


class RoutingTechnique(BaseTechnique):
    name = "routing"
    variants = ("small_first",)

    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        if variant not in self.variants:
            raise ValueError(
                f"routing: unknown variant {variant!r}; expected one of {self.variants}"
            )
        return _SmallFirstStub(knobs)
