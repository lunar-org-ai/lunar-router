"""Tests for router/data/dataset_registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from router.data.dataset import DatasetMetadata
from router.data.dataset_io import save_dataset
from router.data.dataset_registry import (
    delete_dataset,
    get_dataset_meta,
    list_datasets,
    register_dataset,
    update_dataset_meta,
)


@pytest.fixture
def tmp_datasets(tmp_path: Path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


def _meta(name: str = "goldens", owner: str = "human") -> DatasetMetadata:
    return DatasetMetadata(
        name=name,
        desc="test",
        source="manual",
        sourceType="manual",
        use=["Eval"],
        owner=owner,
        growing=False,
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    )


def _payload(name: str = "goldens", version: int = 1) -> dict:
    return {
        "version": version,
        "name": name,
        "desc": "test",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 4,
        "samples": [
            {
                "id": "smp_0001",
                "prompt": "p1",
                "ground_truth": "g1",
                "tag": None,
                "trace_id": None,
                "added_at": "2026-05-09T18:30:00Z",
                "source": "manual",
                "embedding": [0.0, 0.0, 0.0, 0.0],
            }
        ],
        "history": [],
        "metadata": {},
    }


def test_register_creates_entry(tmp_datasets: Path):
    register_dataset(_meta(), current_version=1, size=8, datasets_dir=tmp_datasets)
    meta = get_dataset_meta("goldens", datasets_dir=tmp_datasets)
    assert meta is not None
    assert meta.name == "goldens"
    assert meta.embedding_dim == 384


def test_list_datasets_lists_registered(tmp_datasets: Path):
    register_dataset(_meta("goldens"), current_version=1, size=8, datasets_dir=tmp_datasets)
    register_dataset(_meta("preferences", owner="agent"), current_version=1, size=4, datasets_dir=tmp_datasets)
    names = {m.name for m in list_datasets(datasets_dir=tmp_datasets)}
    assert names == {"goldens", "preferences"}


def test_list_datasets_excludes_soft_deleted(tmp_datasets: Path):
    register_dataset(_meta("a"), current_version=1, size=2, datasets_dir=tmp_datasets)
    register_dataset(_meta("b"), current_version=1, size=2, datasets_dir=tmp_datasets)
    delete_dataset("a", datasets_dir=tmp_datasets)
    names = {m.name for m in list_datasets(datasets_dir=tmp_datasets)}
    assert names == {"b"}
    assert get_dataset_meta("a", datasets_dir=tmp_datasets) is None


def test_update_meta_persists(tmp_datasets: Path):
    register_dataset(_meta(), current_version=1, size=8, datasets_dir=tmp_datasets)
    update_dataset_meta("goldens", growing=True, size=12, datasets_dir=tmp_datasets)
    # growing/size are stored in the registry entry; re-read raw to verify
    import json
    raw = json.loads((tmp_datasets / "_registry.json").read_text())
    assert raw["datasets"]["goldens"]["growing"] is True
    assert raw["datasets"]["goldens"]["size"] == 12


def test_save_dataset_syncs_registry(tmp_datasets: Path):
    """save_dataset() should auto-register the dataset in the registry."""
    save_dataset(_payload(), datasets_dir=tmp_datasets)
    meta = get_dataset_meta("goldens", datasets_dir=tmp_datasets)
    assert meta is not None
    assert meta.name == "goldens"


def test_update_meta_missing_is_noop(tmp_datasets: Path):
    """Updating an unregistered dataset shouldn't raise — it just no-ops."""
    update_dataset_meta("nonexistent", growing=True, datasets_dir=tmp_datasets)
    assert get_dataset_meta("nonexistent", datasets_dir=tmp_datasets) is None
