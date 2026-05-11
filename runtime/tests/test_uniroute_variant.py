"""Tests for techniques.routing.impl._UniRouteStage.

Cold-start fallback + happy path with a synthesized router_config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runtime.embedder_pool import reset_pool
from runtime.protocols import Context
from techniques.routing.impl import RoutingTechnique


def _embedder_with_blobs():
    """Tiny fake embedder so we don't load MiniLM."""
    from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder

    return PromptEmbedder(MockEmbeddingProvider(dimension=8), cache_enabled=False)


def _seed_router_config(versions_dir: Path, *, version: int = 1) -> None:
    """Drop a synthetic router_config artifact + sidecar centroids."""
    centroids = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    np.savez(
        versions_dir / f"router_config_v{version}_centroids.npz",
        type="kmeans",
        centroids=centroids,
    )
    payload = {
        "version": version,
        "k": 2,
        "centroids": None,
        "model_psi": {
            "haiku": {
                "psi_vector": [0.1, 0.5],
                "cost_per_1k_tokens": 0.001,
                "cluster_sample_counts": [10, 10],
            },
            "sonnet": {
                "psi_vector": [0.5, 0.1],
                "cost_per_1k_tokens": 0.003,
                "cluster_sample_counts": [10, 10],
            },
        },
        "cost_weight": 0.0,
        "embedder_model": "test",
        "embedding_dim": 8,
        "fitted_from": {"source": "test"},
        "drift_baseline": 0.5,
        "metadata": {"phase": "test"},
    }
    (versions_dir / f"router_config_v{version}.json").write_text(json.dumps(payload))
    (versions_dir / "router_config_current.txt").write_text(str(version))


def test_uniroute_cold_start_falls_back_to_default(tmp_path: Path, monkeypatch):
    """No router_config_current → routing.model = knobs.small + fallback_reason."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    reset_pool()

    stage = RoutingTechnique().compile("uniroute", knobs={"small": "claude-haiku-4-5"})
    ctx = Context(request="hi")
    out = stage.execute(ctx)

    assert out.routing is not None
    assert out.routing.model == "claude-haiku-4-5"
    assert out.routing.decision is not None
    assert out.routing.decision["cold_start"] is True
    assert "router_not_initialized" in out.routing.decision["fallback_reason"]


def test_uniroute_happy_path_returns_full_decision(tmp_path: Path, monkeypatch):
    """With a fitted config, routing.decision has the full UniRoute payload."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    _seed_router_config(tmp_path, version=1)
    reset_pool()

    # Inject our fake embedder so we don't load MiniLM.
    from runtime.embedder_pool import get_pool

    pool = get_pool()
    pool._embedder = _embedder_with_blobs()

    stage = RoutingTechnique().compile("uniroute", knobs={"small": "claude-haiku-4-5"})
    ctx = Context(request="what is your refund policy?")
    out = stage.execute(ctx)

    assert out.routing is not None
    assert out.routing.model in {"haiku", "sonnet"}
    assert out.routing.decision is not None
    assert out.routing.decision["cold_start"] is False
    assert out.routing.decision["fallback_reason"] is None
    assert "selected_model" in out.routing.decision
    assert "all_scores" in out.routing.decision
    assert "cluster_id" in out.routing.decision
    assert isinstance(out.routing.decision["cluster_probabilities"], list)
    reset_pool()


def test_small_first_unchanged_no_decision_field(tmp_path: Path):
    """Existing variant 'small_first' keeps its old behavior — decision is None."""
    stage = RoutingTechnique().compile(
        "small_first", knobs={"small": "claude-haiku-4-5"}
    )
    out = stage.execute(Context(request="hi"))
    assert out.routing is not None
    assert out.routing.model == "claude-haiku-4-5"
    assert out.routing.decision is None  # untouched by small_first


def test_routing_technique_rejects_unknown_variant():
    with pytest.raises(ValueError) as exc_info:
        RoutingTechnique().compile("ghost", knobs={})
    assert "ghost" in str(exc_info.value)
