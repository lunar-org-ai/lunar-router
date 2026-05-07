"""Built-in rubrics for v0 — no LLM required.

Imported by evals.rubrics.__init__ so the @register_rubric decorators run.
Add new rubrics here (or in sibling modules) and re-export from __init__.
"""

from __future__ import annotations

from evals.rubrics.base import EvalContext, Rubric, register_rubric
from evals.types import RubricResult


@register_rubric
class PipelineSuccess(Rubric):
    """Did the pipeline run end-to-end without error?"""

    type = "pipeline_success"

    def score(self, ctx: EvalContext) -> RubricResult:
        return RubricResult(
            rubric=self.name,
            type=self.type,
            score=1.0 if ctx.success else 0.0,
            passed=ctx.success,
            detail=ctx.error,
        )


@register_rubric
class ResponseNonEmpty(Rubric):
    """Did the agent produce a non-empty response?"""

    type = "response_nonempty"

    def score(self, ctx: EvalContext) -> RubricResult:
        text = (ctx.response or "").strip()
        passed = len(text) > 0
        return RubricResult(
            rubric=self.name,
            type=self.type,
            score=1.0 if passed else 0.0,
            passed=passed,
            detail=f"len={len(text)}",
        )


@register_rubric
class ContainsKeywords(Rubric):
    """Fraction of expected keywords found in the response.

    Params:
      keys: list[str]              # overrides golden.expected.contains if set
      case_sensitive: bool = false
    """

    type = "contains_keywords"

    def score(self, ctx: EvalContext) -> RubricResult:
        keys: list[str] = self.params.get("keys") or list(ctx.golden.expected.contains)
        case_sensitive: bool = bool(self.params.get("case_sensitive", False))

        if not keys:
            return RubricResult(
                rubric=self.name, type=self.type,
                score=1.0, passed=True, detail="no keys to check",
            )

        text = ctx.response or ""
        if not case_sensitive:
            text = text.lower()
            keys = [k.lower() for k in keys]

        hits = [k for k in keys if k in text]
        score = len(hits) / len(keys)
        return RubricResult(
            rubric=self.name,
            type=self.type,
            score=score,
            passed=score == 1.0,
            detail=f"{len(hits)}/{len(keys)} hit: {hits}",
        )


@register_rubric
class LatencyUnderBudget(Rubric):
    """Pipeline ran within the latency budget.

    Params:
      max_ms: float = 2000
    """

    type = "latency_under_budget"

    def score(self, ctx: EvalContext) -> RubricResult:
        max_ms = float(self.params.get("max_ms", 2000.0))
        passed = ctx.duration_ms <= max_ms
        return RubricResult(
            rubric=self.name,
            type=self.type,
            score=1.0 if passed else 0.0,
            passed=passed,
            detail=f"{ctx.duration_ms:.2f}ms vs budget {max_ms}ms",
        )
