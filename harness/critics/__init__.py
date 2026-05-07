"""harness.critics — gates between Proposal and live agent.

Critics are split into two stages:

- **Pre-flight** (`stage="pre"`): see only the Proposal. Cheap. Catch obvious
  problems (scope violations, budget overruns) before branching/scoring.
- **Post-eval** (`stage="post"`): see Proposal + CandidateResult. Decide
  whether scores justify promotion.

A loop runs all `pre` critics → if any block, reject. Else branch+score →
run all `post` critics → if any block, reject. Else approve.
"""

from harness.critics.base import Critic, CriticStage, make_critic, register_critic
from harness.critics import builtins  # noqa: F401  (registers built-ins)

__all__ = ["Critic", "CriticStage", "make_critic", "register_critic"]
