"""Tests for router/data/dataset_io."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router.data.dataset_io import (
    get_current_version,
    load_current,
    load_dataset_payload,
    save_dataset,
)
from router.errors import DatasetInvalidError, DatasetNotFoundError


def _payload(name: str = "goldens", version: int = 1, n_samples: int = 2) -> dict:
    return {
        "version": version,
        "name": name,
        "desc": "test",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "created_at": "2026-05-09T18:30:00Z",
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
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
            for i in range(n_samples)
        ],
        "history": [{"when": "2026-05-09T18:30:00Z", "what": "seed"}],
        "metadata": {"phase": "P15.4.1"},
    }


@pytest.fixture
def tmp_datasets(tmp_path: Path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


def test_save_load_round_trip(tmp_datasets: Path):
    payload = _payload(version=1, n_samples=3)
    save_dataset(payload, datasets_dir=tmp_datasets)
    ds = load_current("goldens", datasets_dir=tmp_datasets)
    assert ds.version == 1
    assert ds.metadata.name == "goldens"
    assert ds.size() == 3
    assert ds.samples[0].prompt == "prompt 0"
    assert ds.samples[0].embedding == [0.1, 0.2, 0.3, 0.4]


def test_save_creates_versioned_files(tmp_datasets: Path):
    save_dataset(_payload(version=1), datasets_dir=tmp_datasets)
    save_dataset(_payload(version=2), datasets_dir=tmp_datasets)
    save_dataset(_payload(version=3), datasets_dir=tmp_datasets)
    base = tmp_datasets / "goldens"
    assert (base / "v1.json").exists()
    assert (base / "v2.json").exists()
    assert (base / "v3.json").exists()
    assert get_current_version("goldens", datasets_dir=tmp_datasets) == 3


def test_load_current_cold_start_raises(tmp_datasets: Path):
    with pytest.raises(DatasetNotFoundError):
        load_current("ghost", datasets_dir=tmp_datasets)


def test_load_dataset_payload_missing_version_raises(tmp_datasets: Path):
    save_dataset(_payload(version=1), datasets_dir=tmp_datasets)
    with pytest.raises(DatasetNotFoundError):
        load_dataset_payload("goldens", 99, datasets_dir=tmp_datasets)


def test_invalid_schema_raises_on_save(tmp_datasets: Path):
    bad = {"version": 1, "name": "x"}  # no 'samples'
    with pytest.raises(DatasetInvalidError):
        save_dataset(bad, datasets_dir=tmp_datasets)


def test_invalid_schema_raises_on_load(tmp_datasets: Path):
    base = tmp_datasets / "goldens"
    base.mkdir(parents=True)
    (base / "v1.json").write_text(json.dumps({"foo": "bar"}))
    (base / "current.txt").write_text("1")
    with pytest.raises(DatasetInvalidError):
        load_current("goldens", datasets_dir=tmp_datasets)


def test_pointer_txt_fallback_works(tmp_datasets: Path):
    """Manually drop a v1.json + current.txt; loader should find it."""
    base = tmp_datasets / "goldens"
    base.mkdir(parents=True)
    (base / "v1.json").write_text(json.dumps(_payload(version=1)))
    (base / "current.txt").write_text("1")
    # No symlink, only .txt — get_current_version still resolves.
    assert get_current_version("goldens", datasets_dir=tmp_datasets) == 1
    ds = load_current("goldens", datasets_dir=tmp_datasets)
    assert ds.version == 1


def test_save_dataset_returns_json_path(tmp_datasets: Path):
    p = save_dataset(_payload(version=1), datasets_dir=tmp_datasets)
    assert p.name == "v1.json"
    assert p.exists()
