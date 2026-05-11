"""Tests for the P15.4.2 /datasets endpoints.

Covers GET (list, detail, health), POST/PUT/DELETE manual flows, and
verifies that POST + PUT emit Lessons with `kind="dataset"` and
`proposal_source="human"`.

Ledger writers + datasets dir are patched per-test so we don't pollute
the real repo state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_datasets(tmp_path: Path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch):
    """Redirect ledger writes so test artifacts don't pollute the repo.

    write_entry / write_lesson capture their default dirs at function-def
    time, so we patch __defaults__ — same trick as test_put_router_config.
    """
    import ledger.writer as lw

    entries = tmp_path / "ledger" / "entries"
    lessons = tmp_path / "ledger" / "lessons"
    entries.mkdir(parents=True)
    lessons.mkdir(parents=True)
    monkeypatch.setattr("ledger.writer.ENTRIES_DIR", entries)
    monkeypatch.setattr("ledger.writer.LESSONS_DIR", lessons)

    we_defaults = list(lw.write_entry.__defaults__ or ())
    if we_defaults:
        we_defaults[-1] = entries
    monkeypatch.setattr(lw.write_entry, "__defaults__", tuple(we_defaults))

    wl_defaults = list(lw.write_lesson.__defaults__ or ())
    if wl_defaults:
        wl_defaults[-1] = lessons
    monkeypatch.setattr(lw.write_lesson, "__defaults__", tuple(wl_defaults))
    return tmp_path / "ledger"


@pytest.fixture
def client():
    from runtime.server import app
    return TestClient(app)


def _seed_goldens(tmp_datasets: Path, n_samples: int = 3) -> dict:
    """Drop a `goldens` dataset directly to disk (skip the embedder cost)."""
    from router.data.dataset_io import save_dataset

    payload = {
        "version": 1,
        "name": "goldens",
        "desc": "Eval suite goldens. Migrated.",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "created_at": "2026-05-09T18:30:00Z",
        "embedder_model": "test",
        "embedding_dim": 4,
        "samples": [
            {
                "id": f"smp_{i:04d}",
                "prompt": f"prompt {i}",
                "ground_truth": f"gt {i}",
                "tag": "policy" if i % 2 == 0 else None,
                "trace_id": None,
                "added_at": "2026-05-09T18:30:00Z",
                "source": "manual",
                "embedding": [0.1 * i, 0.2, 0.3, 0.4],
            }
            for i in range(n_samples)
        ],
        "history": [
            {"when": "2026-05-09T18:30:00Z", "what": "Migrated from evals/golden/*.yaml."}
        ],
        "metadata": {"phase": "P15.4.1"},
    }
    save_dataset(payload, datasets_dir=tmp_datasets)
    return payload


# ---------------------------------------------------------------------------
# GET /datasets (list)
# ---------------------------------------------------------------------------


def test_list_cold_start_returns_empty(client, tmp_datasets):
    res = client.get("/datasets")
    assert res.status_code == 200
    assert res.json() == []


def test_list_returns_seeded_dataset(client, tmp_datasets):
    _seed_goldens(tmp_datasets, n_samples=8)
    res = client.get("/datasets")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "goldens"
    assert row["name"] == "goldens"
    assert row["size"] == 8
    assert row["owner"] == "human"
    assert row["sourceType"] == "manual"
    assert row["use"] == ["Eval"]
    assert row["growing"] is False
    # 'fresh' is relative-time; "just now" / "Xm" / "Xh" / "—" all acceptable
    assert isinstance(row["fresh"], str)
    # No embedding leak — list view doesn't include samples
    assert "samples" not in row
    assert "history" not in row


def test_list_filters_by_use(client, tmp_datasets):
    _seed_goldens(tmp_datasets)
    # use=Distill won't match the Eval-only goldens
    res = client.get("/datasets?use=Distill")
    assert res.status_code == 200
    assert res.json() == []
    res = client.get("/datasets?use=Eval")
    assert len(res.json()) == 1


def test_list_filters_by_owner_and_sourceType(client, tmp_datasets):
    _seed_goldens(tmp_datasets)
    assert client.get("/datasets?owner=agent").json() == []
    assert len(client.get("/datasets?owner=human").json()) == 1
    assert client.get("/datasets?sourceType=auto").json() == []
    assert len(client.get("/datasets?sourceType=manual").json()) == 1


# ---------------------------------------------------------------------------
# GET /datasets/{name}
# ---------------------------------------------------------------------------


def test_get_detail_returns_samples_and_history(client, tmp_datasets):
    _seed_goldens(tmp_datasets, n_samples=3)
    res = client.get("/datasets/goldens")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "goldens"
    assert len(body["samples"]) == 3
    # No embeddings leak to UI
    assert "embedding" not in body["samples"][0]
    assert body["samples"][0]["preview"] == "prompt 0"
    assert body["samples"][0]["tag"] == "policy"
    assert body["samples"][1]["tag"] is None
    # History rendered with relative-time
    assert len(body["history"]) >= 1
    assert isinstance(body["history"][0]["when"], str)


def test_get_detail_404(client, tmp_datasets):
    res = client.get("/datasets/ghost")
    assert res.status_code == 404
    assert "dataset_not_found" in res.json()["detail"]


def test_get_detail_truncates_long_prompts(client, tmp_datasets):
    """Preview is capped at 200 chars + ellipsis."""
    from router.data.dataset_io import save_dataset

    long_prompt = "x" * 500
    payload = {
        "version": 1,
        "name": "long",
        "desc": "",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
        "embedder_model": "test",
        "embedding_dim": 4,
        "samples": [{
            "id": "s1", "prompt": long_prompt, "ground_truth": "", "tag": None,
            "trace_id": None, "added_at": "2026-05-09T18:30:00Z",
            "source": "manual", "embedding": [0.1, 0.2, 0.3, 0.4],
        }],
        "history": [],
        "metadata": {},
    }
    save_dataset(payload, datasets_dir=tmp_datasets)
    body = client.get("/datasets/long").json()
    assert len(body["samples"][0]["preview"]) <= 200
    assert body["samples"][0]["preview"].endswith("…")


# ---------------------------------------------------------------------------
# GET /datasets/{name}/health
# ---------------------------------------------------------------------------


def test_get_health_returns_payload(client, tmp_datasets):
    _seed_goldens(tmp_datasets, n_samples=5)
    res = client.get("/datasets/goldens/health")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "goldens"
    assert body["size"] == 5
    # cluster_distribution is empty because router_config is cold-start in tests
    assert body["cluster_distribution"] == {}
    assert body["coverage_gap_score"] is None
    assert body["last_curation_at"] is not None


def test_get_health_404(client, tmp_datasets):
    res = client.get("/datasets/ghost/health")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# POST /datasets
# ---------------------------------------------------------------------------


def test_post_creates_dataset(client, tmp_datasets, tmp_ledger):
    body = {
        "name": "edge-cases",
        "desc": "Hard cases.",
        "source": "manual",
        "sourceType": "manual",
        "use": ["Eval"],
        "owner": "human",
        "growing": False,
    }
    res = client.post("/datasets", json=body)
    assert res.status_code == 201, res.json()
    out = res.json()
    assert out["name"] == "edge-cases"
    assert out["size"] == 0
    assert out["owner"] == "human"
    assert out["sourceType"] == "manual"

    # Subsequent GET shows it
    listing = client.get("/datasets").json()
    assert any(d["name"] == "edge-cases" for d in listing)

    # Lesson was written with the right shape
    lessons = list((tmp_ledger / "lessons").glob("*.json"))
    assert len(lessons) >= 1
    lesson = json.loads(lessons[0].read_text())
    assert lesson["kind"] == "dataset"
    assert lesson["proposal_source"] == "human"
    assert lesson["status"] == "approved"
    assert "edge-cases" in lesson["title"]


def test_post_rejects_duplicate(client, tmp_datasets, tmp_ledger):
    _seed_goldens(tmp_datasets)
    res = client.post("/datasets", json={"name": "goldens", "use": ["Eval"]})
    assert res.status_code == 409
    assert "dataset_already_exists" in res.json()["detail"]


def test_post_rejects_invalid_name(client, tmp_datasets, tmp_ledger):
    res = client.post("/datasets", json={"name": "Bad Name!", "use": ["Eval"]})
    assert res.status_code == 400


def test_post_rejects_invalid_use(client, tmp_datasets, tmp_ledger):
    res = client.post("/datasets", json={"name": "edge", "use": ["NotAType"]})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# PUT /datasets/{name}
# ---------------------------------------------------------------------------


def test_put_updates_desc_and_bumps_version(client, tmp_datasets, tmp_ledger):
    _seed_goldens(tmp_datasets, n_samples=3)
    res = client.put("/datasets/goldens", json={"desc": "new desc"})
    assert res.status_code == 200, res.json()
    out = res.json()
    assert out["desc"] == "new desc"

    # v2 file exists; current pointer flipped
    assert (tmp_datasets / "goldens" / "v2.json").exists()
    detail = client.get("/datasets/goldens").json()
    assert detail["desc"] == "new desc"

    # Lesson emitted
    lessons = [json.loads(p.read_text()) for p in (tmp_ledger / "lessons").glob("*.json")]
    edits = [l for l in lessons if "Edited" in l.get("title", "")]
    assert len(edits) == 1
    assert edits[0]["kind"] == "dataset"
    assert edits[0]["proposal_source"] == "human"


def test_put_404_on_unknown_dataset(client, tmp_datasets):
    res = client.put("/datasets/ghost", json={"desc": "x"})
    assert res.status_code == 404


def test_put_no_fields_returns_400(client, tmp_datasets):
    _seed_goldens(tmp_datasets)
    res = client.put("/datasets/goldens", json={})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# GET /datasets/{name}/export
# ---------------------------------------------------------------------------


def test_export_streams_ndjson(client, tmp_datasets):
    """Export returns one NDJSON line per sample with full payload (no truncation)."""
    _seed_goldens(tmp_datasets, n_samples=3)
    res = client.get("/datasets/goldens/export")
    assert res.status_code == 200
    assert "ndjson" in res.headers["content-type"]
    assert "goldens.jsonl" in res.headers["content-disposition"]
    lines = [l for l in res.text.strip().split("\n") if l]
    assert len(lines) == 3
    first = json.loads(lines[0])
    # Full payload — no preview truncation, embeddings included
    assert first["prompt"] == "prompt 0"
    assert "embedding" in first
    assert isinstance(first["embedding"], list)
    assert first["embedding"][0] == pytest.approx(0.0)


def test_export_404_on_unknown(client, tmp_datasets):
    res = client.get("/datasets/ghost/export")
    assert res.status_code == 404


def test_export_404_on_soft_deleted(client, tmp_datasets):
    _seed_goldens(tmp_datasets)
    client.delete("/datasets/goldens")
    res = client.get("/datasets/goldens/export")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /datasets/{name}
# ---------------------------------------------------------------------------


def test_delete_soft_deletes(client, tmp_datasets):
    _seed_goldens(tmp_datasets)
    res = client.delete("/datasets/goldens")
    assert res.status_code == 204

    # Not listed anymore
    assert client.get("/datasets").json() == []
    # But v1.json still on disk (rollback-safe)
    assert (tmp_datasets / "goldens" / "v1.json").exists()


def test_delete_404_on_unknown(client, tmp_datasets):
    res = client.delete("/datasets/ghost")
    assert res.status_code == 404
