"""YAML loader with $ref resolution.

agent.yaml composes the pipeline from per-stage files via {$ref: path/to.yaml}.
This loader walks the tree, replaces ref nodes with their loaded content
(recursively, paths relative to the file containing the ref), and validates
the result as an AgentConfig.
"""

from pathlib import Path
from typing import Any

import yaml

from runtime.types import AgentConfig

REF_KEY = "$ref"


def _resolve_refs(data: Any, base_path: Path) -> Any:
    """Recursively substitute {$ref: path} nodes with their loaded content.

    base_path is the directory of the file currently being processed; refs are
    resolved relative to it.
    """
    if isinstance(data, dict):
        if REF_KEY in data and len(data) == 1:
            ref_path = (base_path / data[REF_KEY]).resolve()
            with open(ref_path) as f:
                loaded = yaml.safe_load(f)
            return _resolve_refs(loaded, ref_path.parent)
        return {k: _resolve_refs(v, base_path) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_refs(item, base_path) for item in data]
    return data


def load_agent(path: str | Path) -> AgentConfig:
    """Load and validate an agent definition from a YAML file."""
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_refs(raw, path.parent)
    if "agent" not in resolved:
        raise ValueError(f"{path}: missing top-level 'agent' key")
    return AgentConfig.model_validate(resolved["agent"])
