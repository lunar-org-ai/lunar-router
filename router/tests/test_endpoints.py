"""Integration tests for /router/config and /router/decide.

Uses FastAPI TestClient. Synthesizes router_config artifacts in tmp_path and
monkey-patches the embedder to a fake so tests don't pull MiniLM weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_versions(tmp_path: Path, monkeypatch):
    """Redirect router/config_io to a temp versions/ for each test."""
    versions = tmp_path / "versions"
    versions.mkdir()
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", versions)
    return versions


@pytest.fixture
def fake_embedder(monkeypatch):
    """Force runtime/server.py to use a deterministic mock embedder."""
    from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder

    fake = PromptEmbedder(MockEmbeddingProvider(dimension=8), cache_enabled=False)
    monkeypatch.setattr("runtime.server._router_embedder", fake)
    return fake


@pytest.fixture
def client():
    from runtime.server import app

    return TestClient(app)


# --- helpers ---


def _write_synthetic_config(versions_dir: Path, version: int = 1) -> Path:
    """Drop a hand-rolled router_config_v<n>.json + sidecar centroids + pointer."""
    centroids = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    np.savez(versions_dir / f"router_config_v{version}_centroids.npz", type="kmeans", centroids=centroids)

    payload = {
        "version": version,
        "k": 2,
        "centroids": None,
        "model_psi": {
            "haiku": {"psi_vector": [0.1, 0.5], "cost_per_1k_tokens": 0.001},
            "sonnet": {"psi_vector": [0.5, 0.1], "cost_per_1k_tokens": 0.003},
        },
        "cost_weight": 0.0,
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 8,
        "created_at": "2026-05-09T18:00:00Z",
        "fitted_from": {"source": "synthetic"},
        "metadata": {"phase": "test"},
    }
    json_path = versions_dir / f"router_config_v{version}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # Pointer: try symlink, fall back to .txt.
    ptr = versions_dir / "router_config_current"
    try:
        import os

        os.symlink(json_path.name, ptr)
    except OSError:
        (versions_dir / "router_config_current.txt").write_text(str(version))
    return json_path


# --- GET /router/config ---


def test_get_router_config_cold_start_returns_200(client, tmp_versions):
    """No config exists → 200 with cold_start=True."""
    res = client.get("/router/config")
    assert res.status_code == 200
    body = res.json()
    assert body["cold_start"] is True
    assert body["version"] is None
    assert body["k"] == 0
    assert body["model_count"] == 0
    assert body["embedder_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert body["embedding_dim"] == 384


def test_get_router_config_fitted(client, tmp_versions):
    """Synthesized config → 200 with full metadata."""
    _write_synthetic_config(tmp_versions, version=1)
    res = client.get("/router/config")
    assert res.status_code == 200
    body = res.json()
    assert body["cold_start"] is False
    assert body["version"] == 1
    assert body["k"] == 2
    assert body["model_count"] == 2
    assert body["last_fit_at"] == "2026-05-09T18:00:00Z"
    assert body["fitted_from"] == {"source": "synthetic"}


# --- POST /router/decide ---


def test_post_router_decide_cold_start_503(client, tmp_versions):
    """No config → 503 with router_cold_start detail."""
    res = client.post("/router/decide", json={"prompt": "hello"})
    assert res.status_code == 503
    assert "router_cold_start" in res.json()["detail"]


def test_post_router_decide_happy_path(client, tmp_versions, fake_embedder):
    """Synthesized config + fake embedder → full RoutingDecision shape."""
    _write_synthetic_config(tmp_versions, version=1)
    res = client.post("/router/decide", json={"prompt": "what's your refund policy?"})
    assert res.status_code == 200
    body = res.json()
    assert body["cold_start"] is False
    assert body["selected_model"] in {"haiku", "sonnet"}
    assert isinstance(body["all_scores"], dict)
    assert set(body["all_scores"].keys()) == {"haiku", "sonnet"}
    assert isinstance(body["cluster_probabilities"], list)
    assert len(body["cluster_probabilities"]) == 2
    # cluster_probabilities are valid one-hot for KMeans hard assigner.
    assert sum(body["cluster_probabilities"]) == pytest.approx(1.0)
    assert isinstance(body["reasoning"], str) and len(body["reasoning"]) > 0


def test_post_router_decide_allowed_models_filter(client, tmp_versions, fake_embedder):
    """allowed_models=['sonnet'] forces sonnet even if haiku would otherwise win."""
    _write_synthetic_config(tmp_versions, version=1)
    res = client.post(
        "/router/decide",
        json={"prompt": "hi", "allowed_models": ["sonnet"]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["selected_model"] == "sonnet"


def test_post_router_decide_unknown_allowed_model_400(client, tmp_versions, fake_embedder):
    """allowed_models referencing only unknown IDs → 400."""
    _write_synthetic_config(tmp_versions, version=1)
    res = client.post(
        "/router/decide",
        json={"prompt": "hi", "allowed_models": ["nope"]},
    )
    assert res.status_code == 400


def test_post_router_decide_validation_422(client):
    """Missing 'prompt' → 422 from Pydantic."""
    res = client.post("/router/decide", json={"allowed_models": ["a"]})
    assert res.status_code == 422
