"""Rubric ABC, EvalContext, and the type registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from evals.types import Golden, RubricResult, RubricSpec


@dataclass
class EvalContext:
    """What a rubric sees: the golden + how the agent answered it."""

    golden: Golden
    response: Optional[str]
    duration_ms: float
    success: bool          # pipeline succeeded?
    error: Optional[str]


class Rubric(ABC):
    """Base class for rubrics.

    Subclasses set the class-level `type` (matched against suite YAML) and
    implement `score(ctx) -> RubricResult`.
    """

    type: str = ""  # subclass overrides

    def __init__(self, name: str, params: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.params: dict[str, Any] = params or {}

    @abstractmethod
    def score(self, ctx: EvalContext) -> RubricResult: ...


_REGISTRY: dict[str, type[Rubric]] = {}


def register_rubric(cls: type[Rubric]) -> type[Rubric]:
    """Decorator: register a Rubric subclass under its `type`."""
    if not cls.type:
        raise ValueError(f"{cls.__name__} must set class attribute `type`")
    if cls.type in _REGISTRY:
        raise ValueError(f"rubric type {cls.type!r} already registered")
    _REGISTRY[cls.type] = cls
    return cls


def make_rubric(spec: RubricSpec) -> Rubric:
    """Construct a Rubric instance from a suite YAML spec."""
    cls = _REGISTRY.get(spec.type)
    if cls is None:
        raise ValueError(
            f"unknown rubric type {spec.type!r}; registered: {sorted(_REGISTRY)}"
        )
    return cls(spec.name, spec.params)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)
