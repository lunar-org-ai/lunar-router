"""Tests for router/evaluation/metrics."""

from __future__ import annotations

import pytest

from router.evaluation.metrics import (
    RoutingMetrics,
    compute_apgr,
    compute_auroc,
    compute_cpt,
    compute_pgr_at_savings,
    compute_win_rate,
)


# --- AUROC ---


def test_auroc_returns_half_on_uniform_scores():
    """All scores equal → no separation → AUROC = 0.5 (random)."""
    scores = [0.0] * 10
    labels = [True, False] * 5
    assert compute_auroc(scores, labels) == pytest.approx(0.5, abs=1e-9)


def test_auroc_returns_one_on_perfect_separation():
    """Scores ordered by labels → AUROC = 1.0."""
    scores = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    labels = [True, True, True, True, True, False, False, False, False, False]
    assert compute_auroc(scores, labels) == pytest.approx(1.0, abs=1e-9)


def test_auroc_returns_zero_on_inverted_separation():
    """All-positive labels first but scores flipped → AUROC = 0.0."""
    scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    labels = [True, True, True, True, True, False, False, False, False, False]
    assert compute_auroc(scores, labels) == pytest.approx(0.0, abs=1e-9)


def test_auroc_handles_degenerate_labels():
    """All True or all False → 0.5 (undefined ROC)."""
    assert compute_auroc([1.0, 2.0, 3.0], [True, True, True]) == 0.5
    assert compute_auroc([1.0, 2.0, 3.0], [False, False, False]) == 0.5


def test_auroc_handles_empty():
    assert compute_auroc([], []) == 0.5


# --- APGR ---


def test_apgr_zero_when_router_matches_weak():
    assert compute_apgr(0.5, 0.5, 0.9) == pytest.approx(0.0)


def test_apgr_one_when_router_matches_strong():
    assert compute_apgr(0.9, 0.5, 0.9) == pytest.approx(1.0)


def test_apgr_above_one_when_router_beats_strong():
    assert compute_apgr(0.95, 0.5, 0.9) > 1.0


def test_apgr_when_strong_eq_weak():
    """Quality gap = 0 → degenerate. Returns 1.0 if router >= strong."""
    assert compute_apgr(0.6, 0.5, 0.5) == 1.0
    assert compute_apgr(0.4, 0.5, 0.5) == 0.0


# --- CPT ---


def test_cpt_basic():
    """Find min strong-fraction to reach 95% of [weak=0, strong=1] → 0.95 quality."""
    pareto = [(0.50, 0.0), (0.80, 0.3), (0.96, 0.5), (1.0, 1.0)]
    cpt = compute_cpt(pareto, quality_target_pct=0.95, quality_strong=1.0, quality_weak=0.0)
    assert cpt == pytest.approx(0.5)


def test_cpt_unreachable_returns_none():
    pareto = [(0.50, 0.0), (0.60, 1.0)]
    cpt = compute_cpt(pareto, quality_target_pct=0.95, quality_strong=1.0, quality_weak=0.0)
    assert cpt is None


# --- PGR at savings ---


def test_pgr_at_savings_basic():
    """At 50% savings (max strong frac = 0.5), best quality from points with cost <= 0.5."""
    pareto = [(0.50, 0.0), (0.70, 0.3), (0.85, 0.5), (1.0, 1.0)]
    pgr = compute_pgr_at_savings(pareto, savings_target=0.5, quality_strong=1.0, quality_weak=0.5)
    # best quality at cost <= 0.5 is 0.85; APGR(0.85, 0.5, 1.0) = (0.85-0.5)/0.5 = 0.7
    assert pgr == pytest.approx(0.7)


def test_pgr_at_savings_returns_none_when_no_pareto_points_eligible():
    """All points have cost > max_cost → returns None."""
    pareto = [(0.5, 0.9), (0.6, 1.0)]
    pgr = compute_pgr_at_savings(pareto, savings_target=0.5, quality_strong=1.0, quality_weak=0.0)
    assert pgr is None


# --- win rate ---


def test_win_rate_basic():
    assert compute_win_rate([True, True, False, True, False]) == pytest.approx(0.6)


def test_win_rate_with_explicit_total():
    assert compute_win_rate([True, True], total=10) == pytest.approx(0.2)


def test_win_rate_empty():
    assert compute_win_rate([]) == 0.0


# --- RoutingMetrics dataclass ---


def test_routing_metrics_to_dict_round_trip_keys():
    rm = RoutingMetrics(
        auroc=0.7,
        apgr=0.5,
        win_rate=0.8,
        cpt_50=0.1,
        cpt_75=0.2,
        cpt_90=0.4,
        cpt_95=0.6,
        pgr_at_25_savings=0.5,
        pgr_at_50_savings=0.4,
        pgr_at_75_savings=0.2,
        quality_strong=0.9,
        quality_weak=0.5,
        strong_model="sonnet",
        weak_model="haiku",
        num_samples=100,
    )
    d = rm.to_dict()
    assert d["auroc"] == pytest.approx(0.7)
    assert d["strong_model"] == "sonnet"
    assert d["num_samples"] == 100
    assert "AUROC" in rm.summary()
