"""Load objective definitions from YAML files and resolve compute functions."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Optional

import yaml

from .schemas import Objective, ObjectiveMeasurement

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def load_all() -> list[Objective]:
    """Load every *.yaml in the definitions directory."""
    result: list[Objective] = []
    for path in sorted(DEFINITIONS_DIR.glob("*.yaml")):
        with path.open() as f:
            raw = yaml.safe_load(f)
        result.append(Objective.model_validate(raw))
    return result


def load(objective_id: str) -> Optional[Objective]:
    """Load a single objective by id, or None if not found."""
    for obj in load_all():
        if obj.id == objective_id:
            return obj
    return None


def resolve_compute_fn(
    objective: Objective,
) -> Callable[..., list[ObjectiveMeasurement]]:
    """Turn the 'module:function' pointer into a real callable."""
    module_path, func_name = objective.compute_fn.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
