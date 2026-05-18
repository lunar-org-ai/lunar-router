"""Tests for the _run_router_pipeline branching introduced by the
P15.3 follow-ups (lib.propose_router_retrain end-to-end).

We don't load a real cache + dataset here; we monkeypatch
_critic_inputs_for_proposal to inject a stub critic that returns a
canned verdict, then exercise each Policy decision branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from experiments.types import Mutation
from harness.introspection import lib
from harness.types import CriticVerdict, Proposal


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakePolicy:
    mode: str  # "auto" | "review" | "off"

    def mode_for(self, kind: str) -> str:
        return self.mode


def _make_proposal(version: int = 1) -> Proposal:
    payload = {
        "version": version,
        "k": 2,
        "centroids": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "model_psi": {
            "haiku": {"psi_vector": [0.1, 0.5], "cost_per_1k_tokens": 0.001},
            "sonnet": {"psi_vector": [0.5, 0.1], "cost_per_1k_tokens": 0.003},
        },
        "cost_weight": 0.0,
        "embedder_model": "test",
        "embedding_dim": 8,
        "metadata": {"silhouette": 0.7},
    }
    return Proposal(
        mutations=[
            Mutation(
                file=f"versions/router_config_v{version}.json",
                path="<inline_payload>",
                value=payload,
            )
        ],
        description=f"router_config v{version}",
        source="claude_code",
    )


@pytest.fixture
def tmp_versions(tmp_path: Path, monkeypatch):
    versions = tmp_path / "versions"
    versions.mkdir()
    monkeypatch.setattr("router.config_io.VERSIONS_DIR", versions)
    return versions


@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch):
    """Patch ledger writers via __defaults__ / __kwdefaults__."""
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
def critic_passes(monkeypatch):
    """Make _critic_inputs return non-blocked + RouterCritic.verdict() pass."""
    monkeypatch.setattr(
        lib,
        "_critic_inputs_for_proposal",
        lambda proposal: {"cache": object(), "dataset": object(), "embedder": object()},
    )
    # Patch RouterCritic.verdict to return approved=True
    from harness.critics.router_critic import RouterCritic

    monkeypatch.setattr(
        RouterCritic,
        "verdict",
        lambda self, ctx: CriticVerdict(
            critic="router_quality_gate",
            approved=True,
            reason="delta_auroc=+0.1234",
            severity="info",
        ),
    )


@pytest.fixture
def critic_fails(monkeypatch):
    """Critic blocks the candidate."""
    monkeypatch.setattr(
        lib,
        "_critic_inputs_for_proposal",
        lambda proposal: {"cache": object(), "dataset": object(), "embedder": object()},
    )
    from harness.critics.router_critic import RouterCritic

    monkeypatch.setattr(
        RouterCritic,
        "verdict",
        lambda self, ctx: CriticVerdict(
            critic="router_quality_gate",
            approved=False,
            reason="delta_auroc=-0.05 (regression)",
            severity="block",
        ),
    )


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


def test_pipeline_blocked_when_cache_missing(tmp_versions, tmp_ledger):
    """Default cache path doesn't exist → blocked with cache_missing reason."""
    proposal = _make_proposal()
    out = lib._run_router_pipeline(proposal, _FakePolicy(mode="auto"), "auto")
    assert out["action"] == "blocked"
    assert "cache_missing" in out["reason"] or "cache" in out["reason"].lower()


def test_pipeline_rejected_when_critic_blocks(critic_fails, tmp_versions, tmp_ledger):
    proposal = _make_proposal()
    out = lib._run_router_pipeline(proposal, _FakePolicy(mode="auto"), "auto")
    assert out["action"] == "rejected"
    assert "regression" in out["reason"]
    assert out["lesson_id"].startswith("L-")


def test_pipeline_promoted_on_clear_win(critic_passes, tmp_versions, tmp_ledger, monkeypatch):
    """Critic passes + policy=auto → promote_router_config runs → action=promoted."""
    promoted = []

    def fake_promote(outcome):
        promoted.append(outcome)
        return 7, "L-fake-promoted-aaaa"

    # The pipeline imports `promote_router_config` from
    # ``harness.executor.promote``. The package __init__ shadows the
    # module name with a function (``promote``), so we go through
    # importlib to grab the actual module object.
    import importlib

    promote_mod = importlib.import_module("harness.executor.promote")
    monkeypatch.setattr(promote_mod, "promote_router_config", fake_promote)

    proposal = _make_proposal()
    out = lib._run_router_pipeline(proposal, _FakePolicy(mode="auto"), "auto")
    assert out["action"] == "promoted"
    assert out["lesson_id"] == "L-fake-promoted-aaaa"
    assert out["version"] == 7
    assert promoted, "promote_router_config should have been called"


def test_pipeline_queued_on_review_policy(critic_passes, tmp_versions, tmp_ledger):
    """Critic passes + policy=review → queued Lesson written, no promote."""
    proposal = _make_proposal()
    out = lib._run_router_pipeline(proposal, _FakePolicy(mode="review"), "review")
    assert out["action"] == "queued"
    assert out["lesson_id"].startswith("L-")
    assert "human approval" in out["reason"].lower()


def test_pipeline_blocked_on_off_policy(tmp_versions, tmp_ledger):
    """policy=off → action=blocked at the function-level guard, no Lesson written."""
    proposal = _make_proposal()
    out = lib._run_router_pipeline(proposal, _FakePolicy(mode="off"), "off")
    assert out["action"] == "blocked"
    assert "off" in out["reason"]
    assert out["lesson_id"] is None
