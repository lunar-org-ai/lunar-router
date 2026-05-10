"""Tests for router/feedback/trace_to_training.py."""

from __future__ import annotations

import numpy as np
import pytest

from router.feedback.trace_to_training import (
    ProductionPsiUpdate,
    TraceRecord,
    TraceToTraining,
)
from router.models.llm_profile import LLMProfile


def _trace(
    request_id: str,
    model: str,
    cluster: int,
    is_error: bool = False,
    latency_ms: float = 100.0,
    cost_usd: float = 0.0,
) -> TraceRecord:
    return TraceRecord(
        request_id=request_id,
        selected_model=model,
        cluster_id=cluster,
        is_error=is_error,
        latency_ms=latency_ms,
        total_cost_usd=cost_usd,
    )


def test_add_traces_groups_by_cluster_and_model():
    """50 traces across 3 models × 4 clusters → 3 ProductionPsiUpdate of length-4."""
    conv = TraceToTraining(num_clusters=4)

    rng = np.random.default_rng(seed=42)
    for i in range(50):
        m = f"m{i % 3}"
        c = int(rng.integers(0, 4))
        conv.add_trace(_trace(f"r{i}", m, c, is_error=(i % 7 == 0)))

    updates = conv.compute_psi_updates()
    assert len(updates) == 3
    for u in updates:
        assert isinstance(u, ProductionPsiUpdate)
        assert u.psi_vector.shape == (4,)
        assert u.sample_counts.shape == (4,)
        assert u.total_traces == int(u.sample_counts.sum())
        assert 0.0 <= u.error_rate <= 1.0


def test_psi_update_skips_cluster_minus_one():
    """Records with cluster_id=-1 must NOT land in psi_vector indices."""
    conv = TraceToTraining(num_clusters=3)
    conv.add_trace(_trace("r1", "m", -1, is_error=True))      # skipped
    conv.add_trace(_trace("r2", "m", 0, is_error=False))
    conv.add_trace(_trace("r3", "m", 0, is_error=True))
    updates = conv.compute_psi_updates()
    assert len(updates) == 1
    u = updates[0]
    # Only 2 valid traces accounted for.
    assert u.total_traces == 2
    # cluster 0 has 1 error / 2 → 0.5
    assert pytest.approx(u.psi_vector[0]) == 0.5


def test_latency_threshold_counts_as_error():
    """Latency above latency_threshold_ms increments the error count."""
    conv = TraceToTraining(num_clusters=2, latency_threshold_ms=50.0)
    conv.add_trace(_trace("r1", "m", 0, is_error=False, latency_ms=10.0))   # ok
    conv.add_trace(_trace("r2", "m", 0, is_error=False, latency_ms=200.0))  # latency-error
    updates = conv.compute_psi_updates()
    u = updates[0]
    # 1 error out of 2 in cluster 0.
    assert pytest.approx(u.psi_vector[0]) == 0.5


def test_quality_flags_mark_error():
    """External quality_flags[request_id]=True flips a trace to an error."""
    conv = TraceToTraining(num_clusters=2, quality_flags={"bad-1": True})
    conv.add_trace(_trace("bad-1", "m", 0, is_error=False))
    conv.add_trace(_trace("good-1", "m", 0, is_error=False))
    u = conv.compute_psi_updates()[0]
    assert pytest.approx(u.psi_vector[0]) == 0.5  # 1/2


def test_blend_with_profiles_respects_alpha():
    """alpha=0 → benchmark; alpha=1 → production; alpha=0.3 → blended."""
    bench = LLMProfile(
        model_id="m",
        psi_vector=np.array([0.0, 0.0, 0.0]),
        cost_per_1k_tokens=0.001,
        num_validation_samples=30,
        cluster_sample_counts=np.array([10, 10, 10]),
    )

    conv = TraceToTraining(num_clusters=3)
    # Production: 100% error in cluster 0 only.
    conv.add_trace(_trace("r1", "m", 0, is_error=True))
    conv.add_trace(_trace("r2", "m", 0, is_error=True))

    out0 = conv.blend_with_profiles([bench], alpha=0.0)
    assert np.allclose(out0[0].psi_vector, [0.0, 0.0, 0.0])

    out1 = conv.blend_with_profiles([bench], alpha=1.0)
    # Production fills empty clusters with the global error rate (1.0 here).
    assert pytest.approx(out1[0].psi_vector[0]) == 1.0

    out_mid = conv.blend_with_profiles([bench], alpha=0.3)
    # 0.3 * 1.0 + 0.7 * 0.0 = 0.3 in cluster 0
    assert pytest.approx(out_mid[0].psi_vector[0]) == 0.3
    assert out_mid[0].metadata["production_blended"] is True
    assert out_mid[0].metadata["blend_alpha"] == 0.3


def test_blend_passes_through_models_without_production_data():
    """Profile for a model with no production traces is returned unchanged."""
    p = LLMProfile(
        model_id="lonely",
        psi_vector=np.array([0.5, 0.5]),
        cost_per_1k_tokens=0.001,
        num_validation_samples=10,
        cluster_sample_counts=np.array([5, 5]),
    )
    conv = TraceToTraining(num_clusters=2)  # no traces added
    out = conv.blend_with_profiles([p], alpha=0.3)
    assert out[0] is p


def test_reset_clears_state():
    """After reset() compute_psi_updates returns an empty list."""
    conv = TraceToTraining(num_clusters=2)
    conv.add_trace(_trace("r1", "m", 0))
    assert len(conv.compute_psi_updates()) == 1
    conv.reset()
    assert conv.compute_psi_updates() == []
