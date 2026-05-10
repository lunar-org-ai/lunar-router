"""Tests for PUT /router/config — manual λ override via record_manual_router_change.

Verifies AHE alignment: edit emits Lesson(kind="router_config",
proposal_source="human"), bumps version, and the resulting config is
loadable end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_versions(tmp_path: Path, monkeypatch):
    versions = tmp_path / "versions"
    versions.mkdir()
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", versions)
    return versions


@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch):
    """Redirect ledger writes to a tmp dir so test artifacts don't pollute the repo.

    write_entry / write_lesson capture their default dirs at function-definition
    time, so monkeypatching the module-level constants doesn't reach them.
    We patch ``__defaults__`` instead — the cleanest way to inject into
    pre-existing functions without touching ledger/writer.py.
    """
    import ledger.writer as lw

    entries = tmp_path / "ledger" / "entries"
    lessons = tmp_path / "ledger" / "lessons"
    entries.mkdir(parents=True)
    lessons.mkdir(parents=True)

    monkeypatch.setattr("ledger.writer.ENTRIES_DIR", entries)
    monkeypatch.setattr("ledger.writer.LESSONS_DIR", lessons)

    # write_entry's last positional default is entries_dir.
    we_defaults = list(lw.write_entry.__defaults__ or ())
    if we_defaults:
        we_defaults[-1] = entries
    monkeypatch.setattr(lw.write_entry, "__defaults__", tuple(we_defaults))

    # write_lesson's last positional default is lessons_dir.
    wl_defaults = list(lw.write_lesson.__defaults__ or ())
    if wl_defaults:
        wl_defaults[-1] = lessons
    monkeypatch.setattr(lw.write_lesson, "__defaults__", tuple(wl_defaults))

    return tmp_path / "ledger"


@pytest.fixture
def client():
    from runtime.server import app

    return TestClient(app)


def _seed(versions_dir: Path, version: int = 1, cost_weight: float = 0.0):
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
            }
        },
        "cost_weight": cost_weight,
        "embedder_model": "test",
        "embedding_dim": 8,
        "fitted_from": {"source": "test"},
        "drift_baseline": 0.5,
        "metadata": {"phase": "test"},
    }
    (versions_dir / f"router_config_v{version}.json").write_text(json.dumps(payload))
    (versions_dir / "router_config_current.txt").write_text(str(version))


def test_put_cold_start_returns_409(client, tmp_versions):
    """No current config → 409 with explanatory detail."""
    res = client.put("/router/config", json={"cost_weight": 0.5})
    assert res.status_code == 409
    assert "doesn't exist yet" in res.json()["detail"]


def test_put_no_fields_returns_400(client, tmp_versions):
    _seed(tmp_versions, version=1, cost_weight=0.0)
    res = client.put("/router/config", json={})
    assert res.status_code == 400
    assert "no fields" in res.json()["detail"]


def test_put_bumps_version_and_writes_config(client, tmp_versions, tmp_ledger):
    _seed(tmp_versions, version=3, cost_weight=0.0)

    res = client.put("/router/config", json={"cost_weight": 0.42})
    assert res.status_code == 200, res.json()
    body = res.json()
    assert body["version"] == 4
    assert body["lesson_id"].startswith("L-")
    assert body["config"]["cost_weight"] == pytest.approx(0.42)
    assert body["config"]["cold_start"] is False

    # On disk: v4 JSON exists, v4 sidecar, current pointer flipped to 4.
    assert (tmp_versions / "router_config_v4.json").exists()
    sym = tmp_versions / "router_config_current"
    txt = tmp_versions / "router_config_current.txt"
    assert sym.exists() or txt.exists()
    if sym.is_symlink():
        target_name = sym.resolve().name
    else:
        target_name = f"router_config_v{txt.read_text().strip()}.json"
    assert target_name == "router_config_v4.json"


def test_put_emits_lesson_with_human_source(client, tmp_versions, tmp_ledger):
    _seed(tmp_versions, version=1, cost_weight=0.0)

    res = client.put("/router/config", json={"cost_weight": 0.5})
    assert res.status_code == 200
    lesson_id = res.json()["lesson_id"]

    # Lesson on disk has the right shape.
    lesson_path = tmp_ledger / "lessons" / f"{lesson_id}.json"
    assert lesson_path.exists()
    lesson = json.loads(lesson_path.read_text())
    assert lesson["kind"] == "router_config"
    assert lesson["proposal_source"] == "human"
    assert lesson["status"] == "approved"
    assert lesson["version"] == "2"
    assert "λ" in lesson["title"] or "lambda" in lesson["title"].lower()


def test_put_persists_lambda_across_load(client, tmp_versions, tmp_ledger):
    """After PUT, load_current_config returns the new λ."""
    _seed(tmp_versions, version=1, cost_weight=0.0)

    res = client.put("/router/config", json={"cost_weight": 0.7})
    assert res.status_code == 200

    from router.config_io import load_current_config

    _, _, lam = load_current_config(versions_dir=tmp_versions)
    assert lam == pytest.approx(0.7)
