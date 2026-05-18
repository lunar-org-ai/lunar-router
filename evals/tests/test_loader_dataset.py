"""Tests for the P15.4 suite-loader extension.

Covers:
- Suite.dataset / Suite.goldens mutual exclusion via Pydantic validator.
- resolve_goldens() for the new dataset form (projects DatasetSamples into Goldens).
- find_suites_for_dataset reverse lookup.
- DeprecationWarning fires on the legacy goldens form.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import yaml

from evals.loader import (
    find_suites_for_dataset,
    load_suite,
    resolve_goldens,
)
from evals.types import RubricSpec, Suite


def _write_suite_yaml(path: Path, body: dict) -> None:
    path.write_text(yaml.safe_dump(body))


def _dataset_payload(name: str = "demo", n: int = 2) -> dict:
    return {
        "version": 1,
        "name": name,
        "desc": "test",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "embedder_model": "test",
        "embedding_dim": 4,
        "samples": [
            {
                "id": f"smp_{i:04d}",
                "prompt": f"prompt {i}",
                "ground_truth": f"gt {i}",
                "tag": "policy",
                "trace_id": None,
                "added_at": "2026-05-09T18:30:00Z",
                "source": "manual",
                "embedding": [0.1, 0.2, 0.3, 0.4],
            }
            for i in range(n)
        ],
        "history": [],
        "metadata": {},
    }


@pytest.fixture
def tmp_datasets(tmp_path: Path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_suite_dataset_only_validates():
    s = Suite(suite="x", dataset="demo", rubrics=[RubricSpec(name="r", type="response_nonempty")])
    assert s.dataset == "demo"
    assert s.goldens is None


def test_suite_goldens_only_validates():
    s = Suite(suite="x", goldens=["g1"], rubrics=[RubricSpec(name="r", type="response_nonempty")])
    assert s.dataset is None
    assert s.goldens == ["g1"]


def test_suite_neither_field_raises():
    with pytest.raises(Exception):
        Suite(suite="x", rubrics=[])


def test_suite_both_fields_raises():
    with pytest.raises(Exception):
        Suite(suite="x", dataset="demo", goldens=["g1"], rubrics=[])


# ---------------------------------------------------------------------------
# load_suite + DeprecationWarning
# ---------------------------------------------------------------------------


def test_legacy_goldens_form_emits_deprecation_warning(tmp_path: Path):
    suite_yaml = tmp_path / "smoke.yaml"
    _write_suite_yaml(
        suite_yaml,
        {
            "suite": "smoke",
            "goldens": ["g1"],
            "rubrics": [{"name": "r", "type": "response_nonempty"}],
        },
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_suite(suite_yaml)
        assert any(issubclass(x.category, DeprecationWarning) for x in w), \
            "legacy goldens form should emit DeprecationWarning"


def test_dataset_form_does_not_emit_warning(tmp_path: Path):
    suite_yaml = tmp_path / "smoke.yaml"
    _write_suite_yaml(
        suite_yaml,
        {
            "suite": "smoke",
            "dataset": "demo",
            "rubrics": [{"name": "r", "type": "response_nonempty"}],
        },
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_suite(suite_yaml)
        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deprecations == []


# ---------------------------------------------------------------------------
# resolve_goldens for dataset form
# ---------------------------------------------------------------------------


def test_resolve_goldens_from_dataset(tmp_datasets: Path):
    from router.data.dataset_io import save_dataset
    save_dataset(_dataset_payload("demo", n=3), datasets_dir=tmp_datasets)
    suite = Suite(
        suite="smoke",
        dataset="demo",
        rubrics=[RubricSpec(name="r", type="response_nonempty")],
    )
    goldens = resolve_goldens(suite, datasets_dir=tmp_datasets)
    assert len(goldens) == 3
    assert goldens[0].input.request == "prompt 0"
    assert goldens[0].expected.exact == "gt 0"
    assert goldens[0].expected.category == "policy"
    # id is the sample's stable hash id
    assert goldens[0].id.startswith("smp_")


def test_resolve_goldens_from_legacy_goldens_list(tmp_path: Path):
    """Legacy form still routes through evals/golden/<id>.yaml."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "g1.yaml").write_text(
        "id: g1\ninput:\n  request: \"hello\"\nexpected:\n  contains: [\"hi\"]\n"
    )
    suite = Suite(
        suite="smoke",
        goldens=["g1"],
        rubrics=[RubricSpec(name="r", type="response_nonempty")],
    )
    goldens = resolve_goldens(suite, golden_dir=golden_dir)
    assert len(goldens) == 1
    assert goldens[0].id == "g1"


# ---------------------------------------------------------------------------
# find_suites_for_dataset
# ---------------------------------------------------------------------------


def test_find_suites_for_dataset_returns_matches(tmp_path: Path):
    suites = tmp_path / "suites"
    suites.mkdir()
    _write_suite_yaml(
        suites / "uses_demo.yaml",
        {
            "suite": "uses_demo",
            "dataset": "demo",
            "rubrics": [{"name": "r", "type": "response_nonempty"}],
        },
    )
    _write_suite_yaml(
        suites / "uses_other.yaml",
        {
            "suite": "uses_other",
            "dataset": "other",
            "rubrics": [{"name": "r", "type": "response_nonempty"}],
        },
    )
    _write_suite_yaml(
        suites / "legacy.yaml",
        {
            "suite": "legacy",
            "goldens": ["g1"],
            "rubrics": [{"name": "r", "type": "response_nonempty"}],
        },
    )
    assert find_suites_for_dataset("demo", suites_dir=suites) == ["uses_demo"]
    assert find_suites_for_dataset("other", suites_dir=suites) == ["uses_other"]
    assert find_suites_for_dataset("missing", suites_dir=suites) == []


def test_find_suites_for_dataset_missing_dir_returns_empty(tmp_path: Path):
    assert find_suites_for_dataset("demo", suites_dir=tmp_path / "nope") == []
