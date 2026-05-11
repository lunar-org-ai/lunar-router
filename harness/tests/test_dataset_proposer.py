"""Tests for harness.proposer.dataset_proposer.DatasetProposer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.proposer.dataset_proposer import (
    DatasetProposer,
    DatasetProposerConfig,
    NoAdapterError,
    NothingToAddError,
)
from router.core.clustering import KMeansClusterAssigner
from router.data.dataset_io import save_dataset


class _MockEmbedder:
    """Deterministic 4-dim embedder — stable per prompt."""

    def embed(self, prompt: str) -> list[float]:
        s = sum(ord(c) for c in prompt)
        return [(s % 7) / 10.0, (s % 11) / 10.0, (s % 13) / 10.0, (s % 17) / 10.0]


def _seed_failed_lookup_traces(raw_dir: Path, n: int) -> None:
    import json
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "2026-05-09.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "trace_id": f"t{i}",
                "request": f"unanswered prompt {i}",
                "stages": [
                    {"stage": "retrieve", "technique": "rag", "docs_in": 0, "docs_out": 0}
                ],
            }) + "\n")


def _seed_failed_lookups_dataset(name: str, datasets_dir: Path) -> None:
    payload = {
        "version": 1,
        "name": name,
        "desc": "RAG gaps",
        "source": "failed lookups",
        "sourceType": "auto",
        "use": ["Eval"],
        "owner": "agent",
        "growing": True,
        "embedder_model": "test",
        "embedding_dim": 4,
        "samples": [],
        "history": [],
        "metadata": {},
    }
    save_dataset(payload, datasets_dir=datasets_dir)


@pytest.fixture
def tmp_datasets(tmp_path: Path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


def test_propose_yields_dataset_kind_proposal(tmp_datasets, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _seed_failed_lookup_traces(raw_dir, n=3)
    monkeypatch.setattr(
        "harness.proposer.dataset.mining.failed_lookups._DEFAULT_TRACES_RAW",
        raw_dir,
    )
    _seed_failed_lookups_dataset("rag-gaps", tmp_datasets)

    proposer = DatasetProposer(embedder=_MockEmbedder())
    proposal = proposer.propose("rag-gaps")

    assert proposal.source == "claude_code"
    assert len(proposal.mutations) == 1
    mut = proposal.mutations[0]
    assert mut.file == "datasets/rag-gaps/v2.json"
    payload = mut.value
    assert payload["name"] == "rag-gaps"
    assert payload["version"] == 2
    assert len(payload["samples"]) == 3
    assert payload["samples"][0]["source"] == "failed lookups"
    assert proposal.metadata["adapter_source"] == "failed lookups"
    assert proposal.metadata["added"] == 3


def test_propose_attaches_prediction(tmp_datasets, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _seed_failed_lookup_traces(raw_dir, n=2)
    monkeypatch.setattr(
        "harness.proposer.dataset.mining.failed_lookups._DEFAULT_TRACES_RAW",
        raw_dir,
    )
    _seed_failed_lookups_dataset("rag-gaps", tmp_datasets)

    proposer = DatasetProposer(embedder=_MockEmbedder())
    proposal = proposer.propose("rag-gaps")

    assert proposal.prediction is not None
    assert proposal.prediction.rubric == "coverage_gap_score"
    assert "rag-gaps" in proposal.prediction.rationale


def test_propose_with_assigner_computes_gap_scores(tmp_datasets, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _seed_failed_lookup_traces(raw_dir, n=4)
    monkeypatch.setattr(
        "harness.proposer.dataset.mining.failed_lookups._DEFAULT_TRACES_RAW",
        raw_dir,
    )
    _seed_failed_lookups_dataset("rag-gaps", tmp_datasets)

    assigner = KMeansClusterAssigner(
        centroids=np.asarray([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]),
    )
    proposer = DatasetProposer(embedder=_MockEmbedder(), assigner=assigner)
    proposal = proposer.propose("rag-gaps")

    meta = proposal.metadata
    assert meta["gap_score_before"] is not None
    assert meta["gap_score_after"] is not None


def test_propose_manual_dataset_raises_no_adapter(tmp_datasets):
    # 'manual' source has no adapter
    payload = {
        "version": 1, "name": "manual-ds", "desc": "", "source": "manual",
        "sourceType": "manual", "use": ["Eval"], "owner": "human",
        "growing": False, "embedder_model": "test", "embedding_dim": 4,
        "samples": [], "history": [], "metadata": {},
    }
    save_dataset(payload, datasets_dir=tmp_datasets)

    proposer = DatasetProposer(embedder=_MockEmbedder())
    with pytest.raises(NoAdapterError):
        proposer.propose("manual-ds")


def test_propose_feedback_signals_raises_no_adapter(tmp_datasets):
    payload = {
        "version": 1, "name": "fb", "desc": "", "source": "feedback signals",
        "sourceType": "auto", "use": ["Eval"], "owner": "agent",
        "growing": True, "embedder_model": "test", "embedding_dim": 4,
        "samples": [], "history": [], "metadata": {},
    }
    save_dataset(payload, datasets_dir=tmp_datasets)

    proposer = DatasetProposer(embedder=_MockEmbedder())
    with pytest.raises(NoAdapterError):
        proposer.propose("fb")


def test_propose_empty_traces_raises_nothing_to_add(tmp_datasets, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()  # empty dir
    monkeypatch.setattr(
        "harness.proposer.dataset.mining.failed_lookups._DEFAULT_TRACES_RAW",
        raw_dir,
    )
    _seed_failed_lookups_dataset("rag-gaps", tmp_datasets)

    proposer = DatasetProposer(embedder=_MockEmbedder())
    with pytest.raises(NothingToAddError):
        proposer.propose("rag-gaps")


def test_propose_source_override(tmp_datasets, tmp_path, monkeypatch):
    """source_override picks a different adapter than dataset.metadata.source."""
    raw_dir = tmp_path / "raw"
    _seed_failed_lookup_traces(raw_dir, n=2)
    monkeypatch.setattr(
        "harness.proposer.dataset.mining.failed_lookups._DEFAULT_TRACES_RAW",
        raw_dir,
    )
    # Dataset was created with source='manual' (no adapter), but we override
    payload = {
        "version": 1, "name": "x", "desc": "", "source": "manual",
        "sourceType": "manual", "use": ["Eval"], "owner": "human",
        "growing": False, "embedder_model": "test", "embedding_dim": 4,
        "samples": [], "history": [], "metadata": {},
    }
    save_dataset(payload, datasets_dir=tmp_datasets)

    proposer = DatasetProposer(embedder=_MockEmbedder())
    proposal = proposer.propose("x", source_override="failed lookups")
    assert proposal.metadata["adapter_source"] == "failed lookups"
    assert proposal.metadata["added"] == 2


def test_propose_respects_max_additions(tmp_datasets, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _seed_failed_lookup_traces(raw_dir, n=10)
    monkeypatch.setattr(
        "harness.proposer.dataset.mining.failed_lookups._DEFAULT_TRACES_RAW",
        raw_dir,
    )
    _seed_failed_lookups_dataset("rag-gaps", tmp_datasets)

    proposer = DatasetProposer(
        embedder=_MockEmbedder(),
        cfg=DatasetProposerConfig(max_additions=3),
    )
    proposal = proposer.propose("rag-gaps")
    assert proposal.metadata["added"] == 3
