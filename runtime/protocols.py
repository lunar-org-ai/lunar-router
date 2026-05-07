"""Core protocols and data classes for runtime stages.

A technique is a class that knows how to compile a (variant, knobs) pair into
a Stage. A Stage is anything that takes a Context and returns a (possibly new)
Context. The Context carries everything a stage might need: the request,
history, accumulated documents, the routing decision, the final response, and
a free-form state dict for anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Document:
    """A retrieved or generated piece of context."""

    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in a conversation history."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class RoutingDecision:
    """Set by the route stage; consumed by generate."""

    model: str
    reason: str = ""
    confidence: float = 1.0


@dataclass
class Context:
    """Everything the pipeline carries from stage to stage.

    Well-known slots (request, history, documents, routing, response) are the
    stable contract — techniques that touch them respect their semantics.
    `state` is a free-form dict for anything else a technique wants to pass.
    """

    request: str
    history: list[Message] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    routing: Optional[RoutingDecision] = None
    response: Optional[str] = None
    state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """A compiled, executable pipeline stage.

    Stages are produced by BaseTechnique.compile(). They take a Context, do
    their work, and return a Context (the same instance mutated, by
    convention — but a new instance is allowed).
    """

    def execute(self, context: Context) -> Context: ...


class BaseTechnique(ABC):
    """Base class for techniques.

    A technique is the named entry under techniques/<name>/. The framework
    discovers techniques by importing `techniques.<name>` and reading the
    module-level `TECHNIQUE` symbol (set in that package's __init__.py).

    Conventions:
      - `name` matches the directory name in techniques/.
      - `variants` lists the variant names the technique supports.
      - `compile(variant, knobs)` returns a Stage.

    The compiler validates `variant ∈ self.variants` and `knobs` against the
    technique's schema.yaml before calling compile().
    """

    name: str
    variants: tuple[str, ...]

    @abstractmethod
    def compile(self, variant: str, knobs: dict[str, Any]) -> Stage:
        """Build an executable Stage from a variant name and knob values."""
