"""Tests for router/feedback/store_adapter.py.

Synthesizes a tiny JSONL partition in tmp_path matching the real schema
verified live during P15.3.4 design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from router.core.clustering import KMeansClusterAssigner
from router.core.embeddings import MockEmbeddingProvider, PromptEmbedder
from router.feedback.store_adapter import iter_traces_since


def _row(
    trace_id: str,
    request: str,
    *,
    timestamp: str,
    routing_model: str | None = "claude-haiku-4-5",
    duration_ms: float = 100.0,
    error_in_stage: str | None = None,
) -> dict:
    """Construct a JSONL row matching the live schema."""
    stages = [
        {
            "stage": "retrieve",
            "technique": "rag",
            "variant": "hybrid",
            "duration_ms": 0.04,
            "docs_in": 0,
            "docs_out": 18,
            "response_set": False,
            "routing_model": None,
            "error": "retrieve-failed" if error_in_stage == "retrieve" else None,
        },
        {
            "stage": "route",
            "technique": "routing",
            "variant": "small_first",
            "duration_ms": 0.003,
            "docs_in": 4,
            "docs_out": 4,
            "response_set": False,
            "routing_model": routing_model,
            "error": None,
        },
        {
            "stage": "generate",
            "technique": "prompt_strategies",
            "variant": "direct",
            "duration_ms": 0.007,
            "docs_in": 4,
            "docs_out": 4,
            "response_set": True,
            "routing_model": routing_model,
            "error": "generate-failed" if error_in_stage == "generate" else None,
        },
    ]
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "request": request,
        "response": f"resp for {trace_id}",
        "duration_ms": duration_ms,
        "stages": stages,
    }


def _seed_partitions(root: Path, dates: list[str], rows_per_date: dict[str, list[dict]]) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for date in dates:
        path = raw / f"{date}.jsonl"
        with path.open("w") as f:
            for r in rows_per_date.get(date, []):
                f.write(json.dumps(r) + "\n")
    return raw


def test_iter_traces_cold_start(tmp_path: Path):
    """embedder=None, assigner=None → records with cluster_id=-1."""
    raw = _seed_partitions(
        tmp_path,
        dates=["2026-05-09"],
        rows_per_date={
            "2026-05-09": [
                _row("t1", "ping", timestamp="2026-05-09T10:00:00Z"),
                _row("t2", "pong", timestamp="2026-05-09T10:01:00Z"),
            ],
        },
    )
    out = list(iter_traces_since(traces_root=raw))
    assert len(out) == 2
    for r in out:
        assert r.cluster_id == -1
        assert r.selected_model == "claude-haiku-4-5"
        assert r.is_error is False
    assert out[0].input_text == "ping"


def test_iter_traces_with_assigner(tmp_path: Path):
    """With embedder + assigner: cluster_id ∈ [0, K)."""
    raw = _seed_partitions(
        tmp_path,
        dates=["2026-05-09"],
        rows_per_date={
            "2026-05-09": [
                _row(f"t{i}", f"prompt-{i}", timestamp="2026-05-09T10:00:00Z")
                for i in range(5)
            ]
        },
    )

    embedder = PromptEmbedder(MockEmbeddingProvider(dimension=8), cache_enabled=False)
    rng = np.random.default_rng(seed=7)
    centroids = rng.standard_normal((4, 8))
    assigner = KMeansClusterAssigner(centroids)

    out = list(iter_traces_since(
        traces_root=raw, embedder=embedder, assigner=assigner
    ))
    assert len(out) == 5
    for r in out:
        assert 0 <= r.cluster_id < 4


def test_iter_traces_skips_unattributed(tmp_path: Path):
    """A row with no routing_model in any stage is skipped."""
    raw = _seed_partitions(
        tmp_path,
        dates=["2026-05-09"],
        rows_per_date={
            "2026-05-09": [
                _row("t1", "ping", timestamp="2026-05-09T10:00:00Z"),
                _row("t2", "pong", timestamp="2026-05-09T10:01:00Z", routing_model=None),
            ],
        },
    )
    out = list(iter_traces_since(traces_root=raw))
    assert [r.request_id for r in out] == ["t1"]


def test_iter_traces_marks_errors(tmp_path: Path):
    """A row with a stage error → TraceRecord.is_error=True + error_category set."""
    raw = _seed_partitions(
        tmp_path,
        dates=["2026-05-09"],
        rows_per_date={
            "2026-05-09": [
                _row("t1", "ok", timestamp="2026-05-09T10:00:00Z"),
                _row("t2", "broken", timestamp="2026-05-09T10:01:00Z", error_in_stage="generate"),
            ],
        },
    )
    out = list(iter_traces_since(traces_root=raw))
    by_id = {r.request_id: r for r in out}
    assert by_id["t1"].is_error is False
    assert by_id["t1"].error_category is None
    assert by_id["t2"].is_error is True
    assert by_id["t2"].error_category == "generate-failed"


def test_iter_traces_filters_by_date_window(tmp_path: Path):
    """since_iso/until_iso skips partition files outside the window."""
    raw = _seed_partitions(
        tmp_path,
        dates=["2026-05-07", "2026-05-08", "2026-05-09"],
        rows_per_date={
            "2026-05-07": [_row("d7", "p", timestamp="2026-05-07T10:00:00Z")],
            "2026-05-08": [_row("d8", "p", timestamp="2026-05-08T10:00:00Z")],
            "2026-05-09": [_row("d9", "p", timestamp="2026-05-09T10:00:00Z")],
        },
    )
    out = list(iter_traces_since(since_iso="2026-05-08", traces_root=raw))
    ids = sorted(r.request_id for r in out)
    assert ids == ["d8", "d9"]

    out2 = list(iter_traces_since(
        since_iso="2026-05-07", until_iso="2026-05-09", traces_root=raw
    ))
    ids2 = sorted(r.request_id for r in out2)
    assert ids2 == ["d7", "d8"]


def test_iter_traces_handles_missing_request(tmp_path: Path):
    """Rows without a 'request' field are skipped silently."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2026-05-09.jsonl").write_text(
        json.dumps({"trace_id": "broken", "stages": []}) + "\n"
        + json.dumps(_row("ok", "p", timestamp="2026-05-09T10:00:00Z")) + "\n"
    )
    out = list(iter_traces_since(traces_root=raw))
    assert [r.request_id for r in out] == ["ok"]


def test_iter_traces_skips_malformed_lines(tmp_path: Path):
    """Bad JSON lines are skipped, good lines pass through."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2026-05-09.jsonl").write_text(
        "{not valid json\n"
        + json.dumps(_row("ok", "p", timestamp="2026-05-09T10:00:00Z")) + "\n"
    )
    out = list(iter_traces_since(traces_root=raw))
    assert [r.request_id for r in out] == ["ok"]
