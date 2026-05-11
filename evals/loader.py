"""Load Goldens and Suites from YAML.

Goldens live in evals/golden/<id>.yaml. Suites live anywhere — the canonical
location is evals/suites/<name>.yaml but a runner can be pointed at any path.

P15.4: suites may now reference a dataset by name (datasets/<name>/) instead
of a hand-curated list of goldens. The legacy ``goldens: [id, ...]`` form
still works but emits a DeprecationWarning.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import yaml

from evals.types import Golden, GoldenExpected, GoldenInput, Suite

GOLDEN_DIR = Path("evals/golden")
SUITES_DIR = Path("evals/suites")


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
    """Load and validate a Suite from a YAML file.

    Emits a ``DeprecationWarning`` if the suite uses the legacy
    ``goldens: [id, ...]`` form. Prefer ``dataset: <name>``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"suite not found: {p}")
    with p.open() as f:
        suite = Suite.model_validate(yaml.safe_load(f))
    if suite.goldens:
        warnings.warn(
            f"suite {suite.suite!r} uses legacy 'goldens: [...]' form. "
            "Migrate to 'dataset: <name>' (see P15.4 datasets).",
            DeprecationWarning,
            stacklevel=2,
        )
    return suite


def resolve_goldens(
    suite: Suite,
    *,
    golden_dir: Path | str = GOLDEN_DIR,
    datasets_dir: Optional[Path] = None,
) -> list[Golden]:
    """Return the list of Goldens a suite references, regardless of form.

    Legacy ``goldens: [id, ...]`` → reads ``evals/golden/<id>.yaml``.
    New ``dataset: <name>`` → loads the current dataset version and
    projects each DatasetSample into a synthetic Golden.
    """
    if suite.dataset:
        # Late import — keeps evals.loader free of the router dependency
        # for environments that only need legacy goldens.
        from router.data.dataset_io import load_current
        ds = load_current(suite.dataset, datasets_dir=datasets_dir)
        return [_sample_to_golden(s) for s in ds.samples]
    if suite.goldens:
        return load_goldens(list(suite.goldens), golden_dir=golden_dir)
    raise ValueError(
        f"suite {suite.suite!r} has neither 'dataset' nor 'goldens' set"
    )


def _sample_to_golden(sample) -> Golden:
    """Project a P15.4 DatasetSample into a (synthetic) Golden record."""
    expected_kwargs: dict = {}
    if sample.ground_truth:
        expected_kwargs["exact"] = sample.ground_truth
        expected_kwargs["contains"] = [sample.ground_truth]
    if sample.tag:
        expected_kwargs["category"] = sample.tag
    return Golden(
        id=sample.id,
        input=GoldenInput(request=sample.prompt),
        expected=GoldenExpected(**expected_kwargs),
        metadata={
            "trace_id": sample.trace_id,
            "source": sample.source,
            "added_at": sample.added_at,
        },
    )


def find_suites_for_dataset(
    name: str,
    *,
    suites_dir: Path | str = SUITES_DIR,
) -> list[str]:
    """Reverse-lookup: return the names of suites that reference dataset `name`.

    Used by P15.4.4's critic to find which suites are affected by a
    dataset change. Reads YAML directly to avoid triggering the
    DeprecationWarning that ``load_suite`` would emit.
    """
    suites_dir = Path(suites_dir)
    if not suites_dir.exists():
        return []
    matches: list[str] = []
    for path in sorted(suites_dir.glob("*.yaml")):
        try:
            with path.open() as f:
                raw = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            continue
        if raw.get("dataset") == name:
            suite_name = raw.get("suite") or path.stem
            matches.append(suite_name)
    return matches
