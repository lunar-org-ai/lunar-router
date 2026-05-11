"""Tests for P16.2 — cost estimation + CSAT feedback signals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# runtime/cost.py
# ---------------------------------------------------------------------------


def test_estimate_cost_empty_inputs_returns_zero():
    from runtime.cost import estimate_cost
    assert estimate_cost(None, None) == (0, 0, 0.0)
    assert estimate_cost("", "") == (0, 0, 0.0)


def test_estimate_cost_known_model():
    """Known model + sane char counts → non-zero estimate."""
    from runtime.cost import estimate_cost
    ti, to, c = estimate_cost(
        "What is your refund policy?",  # 27 chars → ~6 tokens
        "30-day refund.",                # 14 chars → ~3 tokens
        model="claude-haiku-4-5",        # $0.001/1k
    )
    assert ti >= 1
    assert to >= 1
    assert 0.0 < c < 0.001  # tiny but non-zero


def test_estimate_cost_unknown_model_falls_back_default():
    """Unknown model name → default rate, no exception."""
    from runtime.cost import estimate_cost
    ti, to, c = estimate_cost("hi", "world", model="nonexistent-model")
    assert ti >= 1 and to >= 1 and c >= 0.0


def test_estimate_cost_more_expensive_model_costs_more():
    from runtime.cost import estimate_cost
    _, _, haiku = estimate_cost("a" * 1000, "b" * 1000, model="claude-haiku-4-5")
    _, _, opus = estimate_cost("a" * 1000, "b" * 1000, model="claude-opus-4-7")
    assert opus > haiku


# ---------------------------------------------------------------------------
# runtime/store/feedback.py
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_feedback(tmp_path: Path, monkeypatch):
    root = tmp_path / "feedback"
    monkeypatch.setattr("runtime.store.feedback._FEEDBACK_ROOT", root)
    return root


def test_feedback_write_and_read(tmp_feedback):
    from runtime.store.feedback import write_feedback, list_feedback_for_trace

    row = write_feedback("t1", 5, "great", root=tmp_feedback)
    assert row["trace_id"] == "t1"
    assert row["score"] == 5
    rows = list_feedback_for_trace("t1", root=tmp_feedback)
    assert len(rows) == 1
    assert rows[0]["comment"] == "great"


def test_feedback_score_validation(tmp_feedback):
    from runtime.store.feedback import write_feedback
    for bad in (0, 6, -1, "5", 3.5):
        with pytest.raises(ValueError):
            write_feedback("t1", bad, root=tmp_feedback)


def test_feedback_csat_aggregation(tmp_feedback):
    from runtime.store.feedback import csat_for_window, write_feedback
    write_feedback("t1", 5, root=tmp_feedback)
    write_feedback("t2", 4, root=tmp_feedback)
    write_feedback("t3", 3, root=tmp_feedback)
    assert csat_for_window(window_days=7, root=tmp_feedback) == 4.0


def test_feedback_csat_empty_returns_none(tmp_feedback):
    from runtime.store.feedback import csat_for_window
    assert csat_for_window(root=tmp_feedback) is None


def test_feedback_multiple_rows_per_trace(tmp_feedback):
    """A trace can be rated multiple times — every row persists."""
    from runtime.store.feedback import list_feedback_for_trace, write_feedback
    write_feedback("t1", 3, root=tmp_feedback)
    write_feedback("t1", 5, root=tmp_feedback)
    rows = list_feedback_for_trace("t1", root=tmp_feedback)
    assert len(rows) == 2
    assert [r["score"] for r in rows] == [3, 5]


# ---------------------------------------------------------------------------
# POST /traces/{id}/feedback
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from runtime.server import app
    return TestClient(app)


def test_post_feedback_404_on_unknown(client, tmp_feedback):
    r = client.post("/traces/ghost-id/feedback", json={"score": 5})
    assert r.status_code == 404


def test_post_feedback_400_on_bad_score(client, tmp_feedback, monkeypatch):
    """Bad score returns 400 with explanatory detail."""
    # Patch get_trace to claim trace exists so we hit the score validation path
    monkeypatch.setattr(
        "runtime.store.traces.get_trace",
        lambda tid: {"trace_id": tid, "request": "x"},
    )
    r = client.post("/traces/t1/feedback", json={"score": 9})
    assert r.status_code == 400
    assert "[1, 5]" in r.json()["detail"]


def test_post_feedback_201_returns_row(client, tmp_feedback, monkeypatch):
    monkeypatch.setattr(
        "runtime.store.traces.get_trace",
        lambda tid: {"trace_id": tid, "request": "x"},
    )
    r = client.post("/traces/t1/feedback", json={"score": 4, "comment": "ok"})
    assert r.status_code == 201
    body = r.json()
    assert body["trace_id"] == "t1"
    assert body["score"] == 4
    assert body["comment"] == "ok"
    assert body["n_total"] == 1
    assert "at" in body


def test_get_feedback_list(client, tmp_feedback, monkeypatch):
    monkeypatch.setattr(
        "runtime.store.traces.get_trace",
        lambda tid: {"trace_id": tid, "request": "x"},
    )
    client.post("/traces/t1/feedback", json={"score": 3})
    client.post("/traces/t1/feedback", json={"score": 5})
    r = client.get("/traces/t1/feedback")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert [row["score"] for row in rows] == [3, 5]


# ---------------------------------------------------------------------------
# /metrics/overview — surfaces both signals
# ---------------------------------------------------------------------------


def test_metrics_overview_includes_cost_and_csat_keys(client, tmp_feedback):
    """Whether populated or null, keys must exist so the UI can render."""
    r = client.get("/metrics/overview")
    assert r.status_code == 200
    body = r.json()
    assert "avg_cost_usd" in body
    assert "csat" in body


def test_metrics_overview_csat_after_feedback(client, tmp_feedback, monkeypatch):
    """Post 2 ratings → csat aggregates them."""
    monkeypatch.setattr(
        "runtime.store.traces.get_trace",
        lambda tid: {"trace_id": tid, "request": "x"},
    )
    client.post("/traces/t1/feedback", json={"score": 4})
    client.post("/traces/t2/feedback", json={"score": 5})
    body = client.get("/metrics/overview").json()
    assert body["csat"] == 4.5


# ---------------------------------------------------------------------------
# ExecutionRecord — cost fields populated by executor
# ---------------------------------------------------------------------------


def test_execution_record_carries_cost_fields():
    """PipelineExecutor.run() populates tokens_in/out + cost_usd."""
    from runtime.compiler.builder import compile_agent
    from runtime.compiler.loader import load_agent
    from runtime.executor.pipeline import PipelineExecutor

    cfg = load_agent("agent/agent.yaml")
    pipeline = compile_agent(cfg)
    executor = PipelineExecutor(pipeline)
    _, record = executor.run("What is your refund policy?")
    assert record.tokens_in >= 1
    assert record.tokens_out >= 1
    assert record.cost_usd > 0.0
