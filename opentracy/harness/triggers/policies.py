"""Policy schema + YAML loader.

A policy is a declarative `signal → dispatch` mapping. The trigger
engine reads them as data and matches every incoming signal against
every policy; behavior changes by editing YAML, not code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


DEFINITIONS_DIR = Path(__file__).parent / "definitions"


class PolicyMatch(BaseModel):
    """Conjunction of filters against a ledger signal entry.

    Empty fields are wildcards. `signal_tags` matches when every tag in
    the list is present on the signal (AND semantics, not OR — lets a
    policy narrow in on e.g. "regression signals for a specific
    objective" without needing a richer expression language).
    """

    signal_tags: list[str] = Field(default_factory=list)
    objective_id: Optional[str] = None


class PolicyBudget(BaseModel):
    """Per-policy rate limit. `max_per_day` counts dispatches written to
    the ledger in the trailing 24h window. Hard cap; no soft overrides."""

    max_per_day: int = 10


class PolicyDispatch(BaseModel):
    """What the engine should invoke when the policy matches.

    Exactly one of `agent` or `recipe` must be set. Recipes are the
    preferred form — they encode multi-step flows as YAML so
    contributors extend behavior without touching Python. Agent-only
    dispatch is kept for simple policies (one signal → one agent run)
    where writing a recipe is overkill.

    Extra fields inside `parameters` are preserved so handlers can read
    policy-specific config (e.g. `days_lookback`, `limit`) without a
    schema change here.
    """

    agent: Optional[str] = None
    recipe: Optional[str] = None
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_exactly_one(self) -> "PolicyDispatch":
        if not self.agent and not self.recipe:
            raise ValueError("PolicyDispatch requires either `agent` or `recipe`")
        if self.agent and self.recipe:
            raise ValueError(
                "PolicyDispatch must set exactly one of `agent` or `recipe`, not both"
            )
        return self


class Policy(BaseModel):
    id: str
    description: str = ""
    match: PolicyMatch
    dispatch: PolicyDispatch
    budget: PolicyBudget = Field(default_factory=PolicyBudget)


def load_policies(definitions_dir: Optional[Path] = None) -> list[Policy]:
    """Read every *.yaml under the definitions directory into Policy
    models. Files that fail to parse raise; a broken policy shouldn't
    silently disable the engine."""
    root = definitions_dir or DEFINITIONS_DIR
    if not root.exists():
        return []
    policies: list[Policy] = []
    for path in sorted(root.glob("*.yaml")):
        with path.open() as f:
            raw = yaml.safe_load(f)
        policies.append(Policy.model_validate(raw))
    return policies


def load_policy(
    policy_id: str, definitions_dir: Optional[Path] = None,
) -> Optional[Policy]:
    for policy in load_policies(definitions_dir):
        if policy.id == policy_id:
            return policy
    return None
