"""Tests for harness.proposer.dataset.coverage.cluster_gaps."""

from __future__ import annotations

import numpy as np
import pytest

from harness.proposer.dataset.coverage import cluster_gaps
from router.core.clustering import KMeansClusterAssigner
from router.data.dataset import Dataset, DatasetMetadata, DatasetSample


def _meta() -> DatasetMetadata:
    return DatasetMetadata(
        name="test",
        desc="",
        source="manual",
        sourceType="manual",
        use=["Eval"],
        owner="human",
        growing=False,
        embedder_model="test",
        embedding_dim=4,
    )


def _sample(id_: str, emb: list[float]) -> DatasetSample:
    return DatasetSample(
        id=id_,
        prompt=f"p {id_}",
        ground_truth="",
        tag=None,
        trace_id=None,
        added_at="2026-05-09T18:30:00Z",
        source="manual",
        embedding=emb,
    )


def _assigner(centroids: list[list[float]]) -> KMeansClusterAssigner:
    return KMeansClusterAssigner(centroids=np.asarray(centroids, dtype=float))


def _dataset(samples: list[DatasetSample]) -> Dataset:
    return Dataset(
        metadata=_meta(),
        version=1,
        samples=samples,
        history=[],
        created_at="2026-05-09T18:30:00Z",
        extra={},
    )


def test_cold_start_no_assigner_returns_none():
    ds = _dataset([_sample("s1", [1.0, 0.0, 0.0, 0.0])])
    assert cluster_gaps(ds, assigner=None) is None


def test_empty_dataset_returns_zero_gaps():
    assigner = _assigner([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    ds = _dataset([])
    report = cluster_gaps(ds, assigner=assigner)
    assert report is not None
    assert report.total_samples == 0
    assert report.expected_per_cluster == 0
    assert report.gap_score == 0.0


def test_uniform_distribution_gap_score_zero():
    assigner = _assigner([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    ds = _dataset([
        _sample("a", [1.0, 0.0, 0.0, 0.0]),
        _sample("b", [1.0, 0.0, 0.0, 0.0]),
        _sample("c", [0.0, 1.0, 0.0, 0.0]),
        _sample("d", [0.0, 1.0, 0.0, 0.0]),
    ])
    report = cluster_gaps(ds, assigner=assigner)
    assert report is not None
    assert report.cluster_distribution == {0: 2, 1: 2}
    assert report.gap_score == 0.0
    assert report.under_covered_clusters() == []


def test_skewed_distribution_gap_score_above_zero():
    assigner = _assigner([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    # All 4 land in cluster 0; cluster 1 empty
    ds = _dataset([
        _sample(f"s{i}", [1.0, 0.0, 0.0, 0.0]) for i in range(4)
    ])
    report = cluster_gaps(ds, assigner=assigner)
    assert report is not None
    assert report.cluster_distribution == {0: 4, 1: 0}
    # expected_per_cluster = ceil(4/2) = 2, so cluster 1 has gap=2
    assert report.expected_per_cluster == 2
    assert report.gap_score > 0.0
    assert report.under_covered_clusters() == [1]


def test_under_covered_sorted_biggest_gap_first():
    assigner = _assigner([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    # 6 samples → expected = 2 per cluster. Distribute as (4, 1, 1) →
    # gaps (0, 1, 1) which are equal — order by cluster_id tie-break.
    ds = _dataset([
        _sample("a", [1.0, 0.0, 0.0, 0.0]),
        _sample("b", [1.0, 0.0, 0.0, 0.0]),
        _sample("c", [1.0, 0.0, 0.0, 0.0]),
        _sample("d", [1.0, 0.0, 0.0, 0.0]),
        _sample("e", [0.0, 1.0, 0.0, 0.0]),
        _sample("f", [0.0, 0.0, 1.0, 0.0]),
    ])
    report = cluster_gaps(ds, assigner=assigner)
    assert report is not None
    under = report.under_covered_clusters()
    assert set(under) == {1, 2}
