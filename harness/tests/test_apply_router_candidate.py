"""Tests for harness.executor.promote.apply_router_candidate.

Verifies atomic write + sidecar centroids + current-pointer flip + the
drift_baseline persisted in the JSON survives a load round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harness.executor.promote import apply_router_candidate


def _make_payload(version: int = 1, k: int = 3) -> dict:
    return {
        "version": version,
        "k": k,
        "centroids": np.eye(k, 8).tolist(),
        "model_psi": {
            "haiku": {
                "psi_vector": [0.1] * k,
                "cost_per_1k_tokens": 0.001,
                "cluster_sample_counts": [10] * k,
                "metadata": {},
            }
        },
        "cost_weight": 0.0,
        "embedder_model": "test",
        "embedding_dim": 8,
        "min_corpus_size": 200,
        "created_at": "2026-05-09T18:00:00Z",
        "fitted_from": {"source": "test"},
        "drift_baseline": 0.4242,
        "metadata": {"phase": "P15.3.7", "stage": "test"},
    }


def test_apply_writes_json_and_sidecar(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    payload = _make_payload(version=1, k=3)

    json_path, npz_path = apply_router_candidate(payload, versions_dir=tmp_path)
    assert json_path.exists()
    assert json_path.name == "router_config_v1.json"
    assert npz_path is not None and npz_path.exists()
    assert npz_path.name == "router_config_v1_centroids.npz"


def test_apply_strips_centroids_from_inline_json(tmp_path: Path, monkeypatch):
    """The big centroid array shouldn't be duplicated in the JSON; it lives
    only in the sidecar .npz."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    payload = _make_payload(version=2, k=3)

    json_path, _ = apply_router_candidate(payload, versions_dir=tmp_path)
    body = json.loads(json_path.read_text())
    assert body["centroids"] is None
    # sidecar has the data instead
    npz = np.load(tmp_path / "router_config_v2_centroids.npz")
    assert "centroids" in npz
    assert npz["centroids"].shape == (3, 8)


def test_apply_persists_drift_baseline(tmp_path: Path, monkeypatch):
    """The drift_baseline arrives via the candidate payload and must survive
    the JSON round-trip — P15.3.4's risk fix."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    payload = _make_payload(version=3, k=3)
    apply_router_candidate(payload, versions_dir=tmp_path)

    body = json.loads((tmp_path / "router_config_v3.json").read_text())
    assert body["drift_baseline"] == pytest.approx(0.4242)


def test_apply_flips_current_pointer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    payload = _make_payload(version=7, k=3)
    apply_router_candidate(payload, versions_dir=tmp_path)

    # Either a symlink or .txt fallback should now point at v7.
    sym = tmp_path / "router_config_current"
    txt = tmp_path / "router_config_current.txt"
    assert sym.exists() or txt.exists()
    if sym.is_symlink():
        target = sym.resolve().name
    else:
        target = f"router_config_v{txt.read_text().strip()}.json"
    assert target == "router_config_v7.json"


def test_apply_round_trip_via_load_current_config(tmp_path: Path, monkeypatch):
    """Write → load_current_config → centroids/registry/lambda all readable."""
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", tmp_path)
    payload = _make_payload(version=4, k=3)
    apply_router_candidate(payload, versions_dir=tmp_path)

    from router.config_io import load_current_config

    assigner, registry, lam = load_current_config(versions_dir=tmp_path)
    assert assigner.num_clusters == 3
    assert "haiku" in registry
    assert lam == pytest.approx(0.0)
