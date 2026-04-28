"""Recipe schema.

Kept deliberately small: agent call OR action call, optional input
from a prior step, optional skip-condition. No loops, no branching
beyond skip, no expressions. Complexity belongs in agents and actions,
not in the recipe DSL — that's what makes recipes reviewable as YAML
diffs in a PR.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class RecipeCondition(BaseModel):
    """Skip-condition on a step.

    If the step named `from_step` produced data where `field == equals`,
    this step runs. Otherwise it is recorded as outcome=skipped with
    parent_id chaining preserved.
    """

    from_step: str
    field: str
    equals: Union[str, int, float, bool]


class RecipeStep(BaseModel):
    """One step in a recipe.

    Exactly one of `agent` or `action` must be set, matching `type`.
    Validation fails loudly at load time so typos in YAML surface
    immediately, not at runtime when a signal fires.
    """

    id: str
    type: Literal["agent", "action"]
    agent: Optional[str] = None
    action: Optional[str] = None
    # Either a literal string, or a pointer to another step's output.
    input: Optional[str] = None
    input_from: Optional[str] = None
    condition: Optional[RecipeCondition] = None

    @model_validator(mode="after")
    def _ensure_shape(self) -> "RecipeStep":
        if self.type == "agent" and not self.agent:
            raise ValueError(f"step {self.id!r} has type='agent' but no 'agent' name")
        if self.type == "action" and not self.action:
            raise ValueError(f"step {self.id!r} has type='action' but no 'action' name")
        if self.input and self.input_from:
            raise ValueError(
                f"step {self.id!r} sets both 'input' and 'input_from'; pick one"
            )
        return self


class RecipeBudget(BaseModel):
    """Hard cap summed across all steps in this single invocation.

    A per-day cap across invocations lives on the policy, not the recipe
    — policies govern frequency, recipes govern per-run cost.
    """

    max_cost_usd: float = 10.0


class Recipe(BaseModel):
    id: str
    description: str = ""
    steps: list[RecipeStep]
    budget: RecipeBudget = Field(default_factory=RecipeBudget)

    @model_validator(mode="after")
    def _check_step_refs(self) -> "Recipe":
        """Surface bad `input_from` / `condition.from_step` references at
        load time, not when the policy fires in production."""
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate step id {step.id!r}")
            seen.add(step.id)
            for ref in (step.input_from, step.condition.from_step if step.condition else None):
                if ref is not None and ref not in seen:
                    raise ValueError(
                        f"step {step.id!r} references {ref!r} which isn't a prior step"
                    )
        return self
