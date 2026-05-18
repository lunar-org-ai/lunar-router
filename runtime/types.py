"""Shared types for the runtime.

These are the data classes that pass between stages and represent the agent's
configuration. They're the contract between agent.yaml, the compiler, the
executor, and the techniques.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class StageConfig(BaseModel):
    """One stage of the pipeline (or one cross-cutting concern).

    `stage` is the role (retrieve, rerank, route, generate). `technique` picks
    a technique from techniques/. `variant` picks an implementation. `knobs`
    are the technique-specific parameters validated against schema.yaml.
    """

    stage: Optional[str] = None
    technique: str
    variant: str
    knobs: dict[str, Any] = Field(default_factory=dict)


class BudgetConfig(BaseModel):
    max_latency_ms: int = 30_000
    max_cost_usd: float = 1.0


class CrossCuttingConfig(BaseModel):
    """Concerns that span the pipeline (memory, caching, etc.).

    Extensible: add fields here as new cross-cutting techniques are introduced.
    """

    memory: Optional[StageConfig] = None


class AgentConfig(BaseModel):
    """Top-level agent definition. Loaded from agent/agent.yaml."""

    version: str
    description: Optional[str] = None
    pipeline: list[StageConfig]
    cross_cutting: CrossCuttingConfig = Field(default_factory=CrossCuttingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
