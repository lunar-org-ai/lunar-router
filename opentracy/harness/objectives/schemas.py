"""Pydantic models for harness objectives."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Direction = Literal["higher_is_better", "lower_is_better"]


class GuardrailSpec(BaseModel):
    """A single guardrail check attached to an objective."""

    type: str
    threshold: Optional[float] = None
    min_n: Optional[int] = None


class Objective(BaseModel):
    """A user-declared measurable objective.

    Loaded from YAML in `opentracy/harness/objectives/definitions/`.
    """

    id: str
    description: str
    compute_fn: str = Field(description="'module:function' pointer to the compute function")
    unit: str
    direction: Direction
    baseline: Optional[float] = None
    target: Optional[float] = None
    window_hours: int = 168
    update_cadence: str = "hourly"
    dimensions: list[str] = Field(default_factory=list)
    owner_agents: list[str] = Field(default_factory=list)
    guardrails: list[GuardrailSpec] = Field(default_factory=list)


class ObjectiveMeasurement(BaseModel):
    """One measurement of an objective for a specific dimension-slice."""

    objective_id: str
    value: Optional[float] = None
    unit: str
    sample_size: int
    window_start: str
    window_end: str
    dimension_values: dict[str, str] = Field(default_factory=dict)
    computed_at: str
