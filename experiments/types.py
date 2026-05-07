"""Experiment data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Mutation:
    """One change applied to a candidate.

    `file` is a path relative to agent/ (e.g. "pipeline/retrieve.yaml" or
    "agent.yaml"). `path` is a dotted key path within the YAML root. `value`
    replaces whatever was at that path.

    Example: Mutation(file="pipeline/retrieve.yaml", path="knobs.k", value=12)
    bumps rag.k from its baseline to 12.
    """

    file: str
    path: str
    value: Any

    def describe(self) -> str:
        return f"{self.file}:{self.path}={self.value}"

    @classmethod
    def parse(cls, spec: str) -> "Mutation":
        """Parse 'file:path=value' shorthand from CLI args.

        Value is parsed as JSON when possible, else kept as string. So
        'knobs.k=12' yields int 12, 'variant="hybrid"' yields string "hybrid",
        'enabled=true' yields bool True.
        """
        if ":" not in spec or "=" not in spec:
            raise ValueError(
                f"mutation spec must look like 'file:path=value', got {spec!r}"
            )
        file_part, rest = spec.split(":", 1)
        path_part, value_str = rest.split("=", 1)
        try:
            import json
            value: Any = json.loads(value_str)
        except (ValueError, TypeError):
            value = value_str
        return cls(file=file_part.strip(), path=path_part.strip(), value=value)


@dataclass
class CandidateManifest:
    """The metadata file written into each candidate directory."""

    id: str
    parent_version: str
    parent_path: str           # path to baseline agent.yaml
    created_at: str            # ISO 8601
    description: Optional[str]
    mutations: list[Mutation] = field(default_factory=list)
