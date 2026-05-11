"""Tests for router/feedback/drift_detector.py."""

from __future__ import annotations

import numpy as np
import pytest

from router.core.clustering import KMeansClusterAssigner
from router.feedback.drift_detector import DriftDetector, DriftReport
from router.feedback.incremental import IncrementalPsiUpdater


def _assigner_with_centroids() -> KMeansClusterAssigner:
    """3 centroids in 4-D, well-separated."""
    centroids = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
        ]
    )
    return KMeansClusterAssigner(centroids)


def test_drift_under_baseline_no_reclustering():
    """Embeddings near centroids → drift_ratio ≈ 1, needs_reclustering False."""
    assigner = _assigner_with_centroids()
    # Baseline = 0.1 (small distance). Embeddings within ~0.05 of centroids
    # should NOT cross drift_threshold (1.5 * baseline = 0.15) or
    # outlier_cutoff (3 * 0.1 = 0.3).
    detector = DriftDetector(assigner, baseline_distance=0.1)
    rng = np.random.default_rng(seed=0)
    near = np.tile(assigner.centroids, (10, 1))  # 30 embeddings exactly at centroids
    near = near + rng.normal(0.0, 0.02, near.shape)
    report = detector.check(near)
    assert isinstance(report, DriftReport)
    assert report.drift_ratio < 1.5
    assert report.outlier_fraction < 0.2
    assert report.needs_reclustering is False
    assert report.num_embeddings == 30


def test_drift_above_threshold_triggers_reclustering():
    """Embeddings far from centroids → drift_ratio > 1.5, needs_reclustering True."""
    assigner = _assigner_with_centroids()
    detector = DriftDetector(assigner, baseline_distance=0.1)
    # Embeddings far from any centroid: pile them at (5, 5, 5, 0) which is
    # ~8.6 from each centroid — well above baseline=0.1.
    far = np.tile(np.array([5.0, 5.0, 5.0, 0.0]), (20, 1))
    report = detector.check(far)
    assert report.drift_ratio > 1.5
    assert report.outlier_fraction > 0.2
    assert report.needs_reclustering is True


def test_drift_outlier_fraction_picks_up_partial_drift():
    """Mix half-near + half-far → outlier_fraction reflects far fraction."""
    assigner = _assigner_with_centroids()
    detector = DriftDetector(assigner, baseline_distance=0.1)
    near = np.tile(assigner.centroids[0], (10, 1))
    far = np.tile(np.array([5.0, 5.0, 5.0, 0.0]), (10, 1))
    mix = np.concatenate([near, far], axis=0)
    report = detector.check(mix)
    # Half the points are far — outlier fraction at least 0.5.
    assert report.outlier_fraction >= 0.4
    assert report.needs_reclustering is True


def test_drift_baseline_auto_set_on_first_check():
    """When baseline_distance is None, first check() sets it to that batch's avg."""
    assigner = _assigner_with_centroids()
    detector = DriftDetector(assigner, baseline_distance=None)
    rng = np.random.default_rng(seed=1)
    embeddings = np.tile(assigner.centroids[0], (15, 1))
    embeddings = embeddings + rng.normal(0.0, 0.05, embeddings.shape)
    report = detector.check(embeddings)
    assert detector.baseline_distance is not None
    assert detector.baseline_distance > 0
    # drift_ratio == 1.0 when baseline equals current batch avg.
    assert pytest.approx(report.drift_ratio, abs=0.001) == 1.0


def test_drift_update_baseline():
    """update_baseline replaces the stored baseline."""
    assigner = _assigner_with_centroids()
    detector = DriftDetector(assigner, baseline_distance=0.1)
    detector.update_baseline(0.5)
    assert detector.baseline_distance == 0.5


def test_drift_empty_input_returns_zeroed_report():
    """check([]) returns a DriftReport with num_embeddings=0 and no recluster flag."""
    assigner = _assigner_with_centroids()
    detector = DriftDetector(assigner, baseline_distance=0.1)
    report = detector.check(np.empty((0, 4)))
    assert report.num_embeddings == 0
    assert report.needs_reclustering is False


# --- incremental.py is a stub — confirm it shouts ---


def test_incremental_psi_updater_stub_raises():
    with pytest.raises(NotImplementedError) as exc_info:
        IncrementalPsiUpdater()
    assert "deferred" in str(exc_info.value).lower()
