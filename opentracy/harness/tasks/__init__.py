"""Task recipes — YAML-defined sequences of agent calls and actions.

A recipe is the "what should happen when a policy fires" side of the
system, mirroring policies ("when should something happen"). Recipes
are data, not code: a contributor adds a new YAML file under
`recipes/` and the executor runs it without a Python change.
"""

from .executor import ExecutionResult, RecipeExecutor
from .loader import load_recipe, load_recipes
from .schema import Recipe, RecipeBudget, RecipeCondition, RecipeStep

__all__ = [
    "ExecutionResult",
    "Recipe",
    "RecipeBudget",
    "RecipeCondition",
    "RecipeExecutor",
    "RecipeStep",
    "load_recipe",
    "load_recipes",
]
