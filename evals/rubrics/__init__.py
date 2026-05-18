"""evals.rubrics — scoring functions.

Built-in rubrics register themselves on import via @register_rubric. Adding a
new rubric: drop a file with class(es) decorated with @register_rubric and
import it from this __init__.
"""

from evals.rubrics.base import EvalContext, Rubric, make_rubric, register_rubric
from evals.rubrics import builtins  # noqa: F401  (registers built-in types)

__all__ = ["EvalContext", "Rubric", "make_rubric", "register_rubric"]
