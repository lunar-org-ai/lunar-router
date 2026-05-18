"""Tests for harness.wakeup.runner.run_wakeup.

We don't hit the live brain. Instead we inject a fake ``introspect_fn`` and
verify the three exit paths (proposed / skipped / blocked) all persist a
decision artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from harness.wakeup.runner import WakeupOutcome, run_wakeup


@dataclass
class _FakeIntrospect:
    response: str = ""
    tool_calls: list = field(default_factory=list)
    success: bool = True


@dataclass
class _FakeToolCall:
    tool: str
    input: dict
    output_preview: str


@pytest.fixture
def tmp_decisions(tmp_path: Path, monkeypatch):
    """Patch DECISIONS_DIR + write_decision __kwdefaults__ so artifacts land in tmp."""
    import ledger.writer as lw

    decisions = tmp_path / "ledger" / "decisions"
    decisions.mkdir(parents=True)
    monkeypatch.setattr("ledger.writer.DECISIONS_DIR", decisions)

    # decisions_dir is a keyword-only arg → lives in __kwdefaults__, not __defaults__.
    new_kwdefaults = dict(lw.write_decision.__kwdefaults__ or {})
    new_kwdefaults["decisions_dir"] = decisions
    monkeypatch.setattr(lw.write_decision, "__kwdefaults__", new_kwdefaults)
    return decisions


@pytest.fixture
def fake_brain_available(monkeypatch):
    """Pretend ANTHROPIC_API_KEY is set so transport detection passes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-fake-key")


def test_skipped_outcome_persists_rationale(tmp_decisions, fake_brain_available):
    fake = _FakeIntrospect(
        response=(
            "Drift is 0.04, well below baseline. Most recent eval was 6 hours "
            "ago and win_rate is 0.78. Skipping retrain."
        ),
        tool_calls=[],
    )
    outcome = run_wakeup(introspect_fn=lambda _p: fake)
    assert isinstance(outcome, WakeupOutcome)
    assert outcome.action == "skipped"
    assert "Skipping retrain" in outcome.rationale
    # Decision artifact written.
    files = list(tmp_decisions.glob("router_wakeup_*.json"))
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body["kind"] == "router_wakeup"
    assert body["payload"]["action"] == "skipped"


def test_proposed_outcome_extracts_lesson_id(tmp_decisions, fake_brain_available):
    """When the model invokes propose_router_retrain and the tool output
    carries a lesson_id, the outcome reflects that."""
    fake = _FakeIntrospect(
        response="Drift looks elevated; I proposed a retrain.",
        tool_calls=[
            _FakeToolCall(
                tool="propose_router_retrain",
                input={"rationale": "drift up"},
                output_preview='{"action":"queued","lesson_id":"L-20260510-130000-abcd"}',
            )
        ],
    )
    outcome = run_wakeup(introspect_fn=lambda _p: fake)
    assert outcome.action == "proposed"
    assert outcome.lesson_id == "L-20260510-130000-abcd"
    files = list(tmp_decisions.glob("router_wakeup_*.json"))
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body["payload"]["action"] == "proposed"
    assert body["payload"]["lesson_id"] == "L-20260510-130000-abcd"


def test_blocked_when_no_brain_available(tmp_decisions, monkeypatch):
    """When neither ANTHROPIC_API_KEY nor `claude` is on PATH, return blocked."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BRAIN_TRANSPORT", raising=False)
    monkeypatch.setattr("harness.brain.transport.shutil.which", lambda _: None)

    outcome = run_wakeup(introspect_fn=lambda _p: _FakeIntrospect())
    assert outcome.action == "blocked"
    assert outcome.reason == "no_brain_available"
    files = list(tmp_decisions.glob("router_wakeup_*.json"))
    assert len(files) == 1


def test_blocked_when_introspect_raises(tmp_decisions, fake_brain_available):
    def boom(_prompt):
        raise RuntimeError("connection refused")

    outcome = run_wakeup(introspect_fn=boom)
    assert outcome.action == "blocked"
    assert outcome.reason == "introspect_error"
    assert "connection refused" in outcome.rationale
