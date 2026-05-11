"""Tests for P16.3 — flag persistence + auto-flag rule + feedback_signals adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_flags(tmp_path: Path, monkeypatch):
    root = tmp_path / "flags"
    monkeypatch.setattr("runtime.store.flags._FLAGS_ROOT", root)
    return root


@pytest.fixture
def tmp_feedback(tmp_path: Path, monkeypatch):
    root = tmp_path / "feedback"
    monkeypatch.setattr("runtime.store.feedback._FEEDBACK_ROOT", root)
    return root


@pytest.fixture
def client():
    from runtime.server import app
    return TestClient(app)


@pytest.fixture
def mock_get_trace(monkeypatch):
    """Patch traces_store.get_trace so endpoints don't require real trace JSONLs."""
    monkeypatch.setattr(
        "runtime.store.traces.get_trace",
        lambda tid: {"trace_id": tid, "request": f"prompt for {tid}"},
    )


# ---------------------------------------------------------------------------
# runtime/store/flags.py
# ---------------------------------------------------------------------------


def test_flag_write_and_read(tmp_flags):
    from runtime.store.flags import write_flag, is_flagged
    write_flag("t1", reason="bad", root=tmp_flags)
    assert is_flagged("t1", root=tmp_flags) is True
    assert is_flagged("t2", root=tmp_flags) is False


def test_flag_unflag_lifecycle(tmp_flags):
    from runtime.store.flags import write_flag, is_flagged
    write_flag("t1", root=tmp_flags)
    assert is_flagged("t1", root=tmp_flags) is True
    write_flag("t1", source="unflag", root=tmp_flags)
    assert is_flagged("t1", root=tmp_flags) is False


def test_flag_source_validation(tmp_flags):
    from runtime.store.flags import write_flag
    with pytest.raises(ValueError):
        write_flag("t1", source="bogus", root=tmp_flags)


def test_flagged_trace_ids_excludes_unflagged(tmp_flags):
    from runtime.store.flags import write_flag, flagged_trace_ids
    write_flag("t1", root=tmp_flags)
    write_flag("t2", root=tmp_flags)
    write_flag("t3", source="csat_low", root=tmp_flags)
    write_flag("t1", source="unflag", root=tmp_flags)
    flagged = flagged_trace_ids(root=tmp_flags)
    assert flagged == {"t2", "t3"}


def test_flag_history_preserved(tmp_flags):
    """Flag/unflag/reflag — all 3 rows persist."""
    from runtime.store.flags import write_flag, list_flag_rows_for_trace, is_flagged
    write_flag("t1", source="csat_low", root=tmp_flags)
    write_flag("t1", source="unflag", root=tmp_flags)
    write_flag("t1", source="manual", reason="re-flagging", root=tmp_flags)
    rows = list_flag_rows_for_trace("t1", root=tmp_flags)
    assert len(rows) == 3
    assert [r["source"] for r in rows] == ["csat_low", "unflag", "manual"]
    assert is_flagged("t1", root=tmp_flags) is True


# ---------------------------------------------------------------------------
# POST/DELETE /traces/{id}/flag
# ---------------------------------------------------------------------------


def test_flag_endpoint_404_on_unknown(client, tmp_flags):
    r = client.post("/traces/ghost-id/flag", json={"reason": "x"})
    assert r.status_code == 404


def test_flag_endpoint_201_on_success(client, tmp_flags, mock_get_trace):
    r = client.post("/traces/t1/flag", json={"reason": "looks bad"})
    assert r.status_code == 201
    body = r.json()
    assert body["flagged"] is True
    assert body["last_row"]["source"] == "manual"
    assert body["last_row"]["reason"] == "looks bad"


def test_unflag_endpoint_clears_flag(client, tmp_flags, mock_get_trace):
    client.post("/traces/t1/flag", json={})
    r = client.delete("/traces/t1/flag")
    assert r.status_code == 200
    assert r.json()["flagged"] is False


def test_list_flag_endpoint_returns_history(client, tmp_flags, mock_get_trace):
    client.post("/traces/t1/flag", json={"reason": "first"})
    client.delete("/traces/t1/flag")
    client.post("/traces/t1/flag", json={"reason": "again"})
    rows = client.get("/traces/t1/flag").json()
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Auto-flag rule (CSAT ≤ 2)
# ---------------------------------------------------------------------------


def test_low_csat_auto_flags(client, tmp_flags, tmp_feedback, mock_get_trace):
    """Score ≤ 2 auto-flags the trace with source=csat_low."""
    from runtime.store import flags as flags_store
    r = client.post("/traces/t1/feedback", json={"score": 1})
    assert r.status_code == 201
    assert flags_store.is_flagged("t1") is True
    # Inspect flag row source
    rows = client.get("/traces/t1/flag").json()
    assert rows[0]["source"] == "csat_low"
    assert "CSAT score 1/5" in rows[0]["reason"]


def test_high_csat_does_not_flag(client, tmp_flags, tmp_feedback, mock_get_trace):
    from runtime.store import flags as flags_store
    client.post("/traces/t1/feedback", json={"score": 4})
    assert flags_store.is_flagged("t1") is False


def test_threshold_boundary_score_2_flags(
    client, tmp_flags, tmp_feedback, mock_get_trace,
):
    """Score == threshold (2) does fire the rule."""
    from runtime.store import flags as flags_store
    client.post("/traces/t1/feedback", json={"score": 2})
    assert flags_store.is_flagged("t1") is True


def test_auto_flag_idempotent(client, tmp_flags, tmp_feedback, mock_get_trace):
    """Multiple low-score submissions don't add multiple auto-flag rows."""
    client.post("/traces/t1/feedback", json={"score": 2})
    client.post("/traces/t1/feedback", json={"score": 1})
    rows = client.get("/traces/t1/flag").json()
    # First low score fires the auto-flag; second is a no-op (already flagged).
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# TraceSummary.flagged stamped on list/get
# ---------------------------------------------------------------------------


def test_list_traces_stamps_flagged(client, tmp_flags, monkeypatch):
    """The list endpoint surfaces server-stamped flag state per row."""
    # Inject one trace + a flag row.
    monkeypatch.setattr(
        "runtime.store.traces.available_dates", lambda: ["2026-05-11"]
    )
    monkeypatch.setattr(
        "runtime.store.traces.query_traces",
        lambda **kw: (
            [{
                "trace_id": "abc-1", "timestamp": "2026-05-11T12:00:00Z",
                "request": "p", "response": "r", "duration_ms": 1.0,
                "success": True, "error": None, "agent_version": "v1",
                "n_stages": 1, "routing_model": "claude-haiku-4-5",
                "session_id": None, "n_turns": 1,
                "tokens_in": 1, "tokens_out": 2, "cost_usd": 0.0001,
            }],
            1,
        ),
    )

    from runtime.store import flags as flags_store
    flags_store.write_flag("abc-1", reason="test", source="manual")

    r = client.get("/traces?date=2026-05-11&limit=10")
    assert r.status_code == 200
    items = r.json()["items"]
    assert items[0]["flagged"] is True


# ---------------------------------------------------------------------------
# feedback_signals adapter
# ---------------------------------------------------------------------------


class _MockEmbedder:
    def embed(self, prompt: str) -> list[float]:
        s = sum(ord(c) for c in prompt)
        return [(s % 7) / 10.0, (s % 11) / 10.0, (s % 13) / 10.0, (s % 17) / 10.0]


def test_feedback_signals_yields_low_score_candidates(tmp_feedback):
    """Low-CSAT feedback rows produce DatasetSample candidates."""
    from harness.proposer.dataset.mining.feedback_signals import iter_candidates
    from runtime.store.feedback import write_feedback

    write_feedback("t1", 2, root=tmp_feedback)
    write_feedback("t2", 5, root=tmp_feedback)  # too high — skipped
    write_feedback("t3", 1, root=tmp_feedback)

    fake_traces = {
        "t1": {"trace_id": "t1", "request": "the agent ignored me"},
        "t2": {"trace_id": "t2", "request": "great service"},
        "t3": {"trace_id": "t3", "request": "wrong refund amount"},
    }

    samples = list(iter_candidates(
        embedder=_MockEmbedder(),
        feedback_root=tmp_feedback,
        get_trace=fake_traces.get,
    ))
    by_tid = {s.trace_id: s for s in samples}
    assert set(by_tid) == {"t1", "t3"}
    assert by_tid["t1"].tag == "csat_2"
    assert by_tid["t3"].tag == "csat_1"
    assert by_tid["t1"].source == "feedback signals"
    assert len(by_tid["t1"].embedding) == 4


def test_feedback_signals_latest_score_wins(tmp_feedback):
    """If a trace gets re-rated, the latest score determines inclusion."""
    from harness.proposer.dataset.mining.feedback_signals import iter_candidates
    from runtime.store.feedback import write_feedback

    write_feedback("t1", 1, root=tmp_feedback)  # bad
    write_feedback("t1", 5, root=tmp_feedback)  # corrected — should NOT mine

    fake_traces = {"t1": {"trace_id": "t1", "request": "p"}}
    samples = list(iter_candidates(
        embedder=_MockEmbedder(),
        feedback_root=tmp_feedback,
        get_trace=fake_traces.get,
    ))
    assert samples == []


def test_feedback_signals_dedup_against_existing(tmp_feedback):
    """Existing IDs are filtered."""
    from harness.proposer.dataset.mining.base import prompt_hash
    from harness.proposer.dataset.mining.feedback_signals import iter_candidates
    from runtime.store.feedback import write_feedback

    write_feedback("t1", 1, root=tmp_feedback)
    fake_traces = {"t1": {"trace_id": "t1", "request": "skip me"}}

    sid = prompt_hash("skip me", "csat_1")
    samples = list(iter_candidates(
        embedder=_MockEmbedder(),
        feedback_root=tmp_feedback,
        get_trace=fake_traces.get,
        existing={sid},
    ))
    assert samples == []


def test_feedback_signals_threshold_override(tmp_feedback):
    """Custom threshold lets callers tune severity."""
    from harness.proposer.dataset.mining.feedback_signals import iter_candidates
    from runtime.store.feedback import write_feedback

    write_feedback("t1", 3, root=tmp_feedback)
    fake_traces = {"t1": {"trace_id": "t1", "request": "p"}}

    # Default threshold (2) excludes score=3
    assert list(iter_candidates(
        embedder=_MockEmbedder(), feedback_root=tmp_feedback,
        get_trace=fake_traces.get,
    )) == []

    # threshold=3 includes it
    samples = list(iter_candidates(
        embedder=_MockEmbedder(), feedback_root=tmp_feedback,
        get_trace=fake_traces.get, threshold=3,
    ))
    assert len(samples) == 1


def test_feedback_signals_no_longer_raises_notimplemented():
    """P15.4.3 stub previously raised NotImplementedError; P16.3 wires it."""
    from harness.proposer.dataset.mining.feedback_signals import iter_candidates

    # Doesn't raise during construction; only raises if backends are broken.
    # Empty feedback root → empty stream, no error.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        samples = list(iter_candidates(
            embedder=_MockEmbedder(),
            feedback_root=Path(d) / "nope",
            get_trace=lambda tid: None,
        ))
    assert samples == []


# ---------------------------------------------------------------------------
# End-to-end: feedback → auto-flag → mining
# ---------------------------------------------------------------------------


def test_e2e_low_csat_flows_to_dataset_candidate(
    client, tmp_flags, tmp_feedback, mock_get_trace,
):
    """Operator rates ★, server auto-flags, mining adapter surfaces it."""
    from harness.proposer.dataset.mining.feedback_signals import iter_candidates
    from runtime.store import flags as flags_store

    # 1. User rates a trace 1 star
    r = client.post("/traces/t-bad/feedback", json={"score": 1, "comment": "wrong"})
    assert r.status_code == 201

    # 2. Trace is now flagged (auto-flag rule)
    assert flags_store.is_flagged("t-bad") is True

    # 3. Mining adapter picks it up
    samples = list(iter_candidates(
        embedder=_MockEmbedder(),
        feedback_root=tmp_feedback,
        get_trace=lambda tid: {"trace_id": tid, "request": "prompt for t-bad"},
    ))
    assert len(samples) == 1
    assert samples[0].trace_id == "t-bad"
    assert samples[0].tag == "csat_1"
    assert samples[0].source == "feedback signals"
