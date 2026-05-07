"""Load Goldens and Suites from YAML.

Goldens live in evals/golden/<id>.yaml. Suites live anywhere — the canonical
location is evals/suites/<name>.yaml but a runner can be pointed at any path.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from evals.types import Golden, Suite

GOLDEN_DIR = Path("evals/golden")


def load_golden(golden_id: str, golden_dir: Path | str = GOLDEN_DIR) -> Golden:
    """Load a single Golden by id."""
    path = Path(golden_dir) / f"{golden_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"golden not found: {path}")
    with path.open() as f:
        return Golden.model_validate(yaml.safe_load(f))


def load_goldens(ids: list[str], golden_dir: Path | str = GOLDEN_DIR) -> list[Golden]:
    return [load_golden(i, golden_dir) for i in ids]


def load_suite(path: Path | str) -> Suite:
    """Load and validate a Suite from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"suite not found: {p}")
    with p.open() as f:
        return Suite.model_validate(yaml.safe_load(f))
