"""YAML loader for recipe definitions.

Mirrors `objectives/loader.py` and `triggers/policies.py` — a new
recipe is a drop-in YAML under `recipes/`, no code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .schema import Recipe


DEFINITIONS_DIR = Path(__file__).parent / "recipes"


def load_recipes(recipes_dir: Optional[Path] = None) -> list[Recipe]:
    root = recipes_dir or DEFINITIONS_DIR
    if not root.exists():
        return []
    out: list[Recipe] = []
    for path in sorted(root.glob("*.yaml")):
        with path.open() as f:
            raw = yaml.safe_load(f)
        out.append(Recipe.model_validate(raw))
    return out


def load_recipe(recipe_id: str, recipes_dir: Optional[Path] = None) -> Optional[Recipe]:
    for r in load_recipes(recipes_dir):
        if r.id == recipe_id:
            return r
    return None
