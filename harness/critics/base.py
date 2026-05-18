"""Critic ABC + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from harness.types import CriticContext, CriticVerdict


class CriticStage(str, Enum):
    PRE = "pre"      # see Proposal only — runs before branching
    POST = "post"    # see Proposal + candidate_result — runs after scoring


class Critic(ABC):
    """Base class for critics."""

    name: str = ""
    stage: CriticStage = CriticStage.PRE

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params: dict[str, Any] = params or {}

    @abstractmethod
    def verdict(self, ctx: CriticContext) -> CriticVerdict: ...


_REGISTRY: dict[str, type[Critic]] = {}


def register_critic(cls: type[Critic]) -> type[Critic]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set class attribute `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"critic {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def make_critic(name: str, params: Optional[dict[str, Any]] = None) -> Critic:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown critic {name!r}; registered: {sorted(_REGISTRY)}")
    return cls(params)


def critics_for_stage(stage: CriticStage, names: list[str]) -> list[Critic]:
    """Build the subset of named critics that run at the given stage."""
    out: list[Critic] = []
    for n in names:
        c = make_critic(n)
        if c.stage == stage:
            out.append(c)
    return out


def registered_critics() -> list[str]:
    return sorted(_REGISTRY)
