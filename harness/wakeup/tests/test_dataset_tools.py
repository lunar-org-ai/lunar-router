"""Tests for the P15.4.5 dataset MCP tools + wakeup integration.

We don't hit the live brain or the live registry — we patch:
  - DEFAULT_DATASETS_DIR (storage)
  - Policy.from_yaml (gating)
  - DatasetProposer.propose (so we can drive each branch)
  - ledger writers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest


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
    import ledger.writer as lw

    entries = tmp_path / "ledger" / "entries"
    lessons = tmp_path / "ledger" / "lessons"
    decisions = tmp_path / "ledger" / "decisions"
    for d in (entries, lessons, decisions):
        d.mkdir(parents=True)
    monkeypatch.setattr("ledger.writer.ENTRIES_DIR", entries)
    monkeypatch.setattr("ledger.writer.LESSONS_DIR", lessons)
    monkeypatch.setattr("ledger.writer.DECISIONS_DIR", decisions)

    we_kw = dict(lw.write_entry.__kwdefaults__ or {})
    if "entries_dir" in we_kw:
        we_kw["entries_dir"] = entries
        monkeypatch.setattr(lw.write_entry, "__kwdefaults__", we_kw)
    wl_d = list(lw.write_lesson.__defaults__ or ())
    if wl_d:
        wl_d[-1] = lessons
        monkeypatch.setattr(lw.write_lesson, "__defaults__", tuple(wl_d))
    wd_kw = dict(lw.write_decision.__kwdefaults__ or {})
    if "decisions_dir" in wd_kw:
        wd_kw["decisions_dir"] = decisions
        monkeypatch.setattr(lw.write_decision, "__kwdefaults__", wd_kw)
    return tmp_path / "ledger"


def _seed_dataset(name: str, datasets_dir: Path, *, source: str = "failed lookups") -> None:
    from router.data.dataset_io import save_dataset
    save_dataset({
        "version": 1, "name": name, "desc": "",
        "source": source, "sourceType": "auto", "use": ["Eval"],
        "owner": "agent", "growing": True,
        "embedder_model": "test", "embedding_dim": 4,
        "samples": [], "history": [], "metadata": {},
    }, datasets_dir=datasets_dir)


# ---------------------------------------------------------------------------
# dataset_health_check
# ---------------------------------------------------------------------------


def test_health_check_cold_start_returns_empty_list(tmp_datasets):
    from harness.introspection.lib import dataset_health_check
    out = dataset_health_check()
    assert out == {"datasets": []}


def test_health_check_lists_all_datasets(tmp_datasets):
    from harness.introspection.lib import dataset_health_check
    _seed_dataset("a", tmp_datasets)
    _seed_dataset("b", tmp_datasets, source="manual")
    out = dataset_health_check()
    names = {d["name"] for d in out["datasets"]}
    assert names == {"a", "b"}
    by_name = {d["name"]: d for d in out["datasets"]}
    # adapter_available reflects whether a mining adapter exists for source.
    assert by_name["a"]["adapter_available"] is True   # failed lookups
    assert by_name["b"]["adapter_available"] is False  # manual


def test_health_check_single_name_returns_snapshot(tmp_datasets):
    from harness.introspection.lib import dataset_health_check
    _seed_dataset("focus", tmp_datasets)
    snap = dataset_health_check("focus")
    assert snap["name"] == "focus"
    assert snap["version"] == 1
    assert snap["size"] == 0
    # router cold-start → gap_score is None
    assert snap["gap_score"] is None


def test_health_check_unknown_name_returns_error(tmp_datasets):
    from harness.introspection.lib import dataset_health_check
    snap = dataset_health_check("ghost")
    assert snap.get("error") == "dataset_not_found"


# ---------------------------------------------------------------------------
# propose_dataset_curation — policy gating
# ---------------------------------------------------------------------------


def test_propose_blocked_by_policy_off(monkeypatch, tmp_datasets, tmp_ledger):
    """Policy mode='off' for dataset → blocked without invoking proposer."""
    from harness.introspection import lib

    class _Pol:
        def mode_for(self, kind):
            return "off" if kind == "dataset" else "auto"

    monkeypatch.setattr("harness.approver.policy.Policy.from_yaml", lambda: _Pol())
    out = lib.propose_dataset_curation("anything")
    assert out["action"] == "blocked"
    assert "off" in out["reason"]
    assert out["lesson_id"] is None


def test_propose_blocked_when_dataset_missing(monkeypatch, tmp_datasets, tmp_ledger):
    from harness.introspection import lib

    class _Pol:
        def mode_for(self, kind):
            return "auto"

    monkeypatch.setattr("harness.approver.policy.Policy.from_yaml", lambda: _Pol())
    out = lib.propose_dataset_curation("ghost-dataset")
    assert out["action"] == "blocked"
    assert "dataset_not_found" in out["reason"]


def test_propose_blocked_when_no_adapter(monkeypatch, tmp_datasets, tmp_ledger):
    """source='manual' has no adapter → blocked with NoAdapterError reason."""
    from harness.introspection import lib

    class _Pol:
        def mode_for(self, kind):
            return "auto"
    monkeypatch.setattr("harness.approver.policy.Policy.from_yaml", lambda: _Pol())
    _seed_dataset("manual-x", tmp_datasets, source="manual")

    # Stub the embedder pool so init doesn't try to load MiniLM
    class _Emb:
        def embed(self, p): return [0.0, 0.0, 0.0, 0.0]

    class _Pool:
        def get(self): return _Emb()

    monkeypatch.setattr("runtime.embedder_pool.get_pool", lambda: _Pool())

    out = lib.propose_dataset_curation("manual-x")
    assert out["action"] == "blocked"
    assert "no_adapter" in out["reason"]


# ---------------------------------------------------------------------------
# propose_dataset_curation — end-to-end branches
# ---------------------------------------------------------------------------


def _patch_proposer_to_return(monkeypatch, prediction_delta=-0.2):
    """Make DatasetProposer.propose return a canned proposal."""
    from experiments.types import Mutation
    from harness.types import Prediction, Proposal

    def _fake_propose(self, name, *, source_override=None):
        payload = {
            "version": 2,
            "name": name,
            "desc": "",
            "source": source_override or "failed lookups",
            "sourceType": "auto",
            "use": ["Eval"],
            "owner": "agent",
            "growing": True,
            "embedder_model": "test",
            "embedding_dim": 4,
            "samples": [{
                "id": "smp_0001", "prompt": "p1", "ground_truth": "",
                "tag": "tag", "trace_id": None,
                "added_at": "2026-05-09T18:30:00Z",
                "source": "failed lookups",
                "embedding": [0.1, 0.2, 0.3, 0.4],
            }],
            "history": [],
            "metadata": {"added": 1, "gap_score_before": 0.4, "gap_score_after": 0.2},
        }
        return Proposal(
            mutations=[Mutation(file=f"datasets/{name}/v2.json",
                                path="<inline_payload>", value=payload)],
            description="test",
            source="claude_code",
            prediction=Prediction(
                rubric="coverage_gap_score",
                expected_delta=prediction_delta,
                rationale="test",
                confidence=0.5,
            ),
        )
    from harness.proposer.dataset_proposer import DatasetProposer
    monkeypatch.setattr(DatasetProposer, "propose", _fake_propose)


def _patch_pool(monkeypatch):
    class _Emb:
        def embed(self, p): return [0.0, 0.0, 0.0, 0.0]
    class _Pool:
        def get(self): return _Emb()
    monkeypatch.setattr("runtime.embedder_pool.get_pool", lambda: _Pool())


def test_propose_promotes_when_policy_auto_and_critic_passes(
    monkeypatch, tmp_datasets, tmp_ledger,
):
    """Policy auto + clean candidate → promoted, Lesson written."""
    from harness.introspection import lib

    class _Pol:
        def mode_for(self, kind): return "auto"
    monkeypatch.setattr("harness.approver.policy.Policy.from_yaml", lambda: _Pol())
    _patch_pool(monkeypatch)
    _seed_dataset("rag-gaps", tmp_datasets)
    _patch_proposer_to_return(monkeypatch)

    out = lib.propose_dataset_curation("rag-gaps", rationale="filling gaps")
    assert out["action"] == "promoted", out
    assert out["lesson_id"].startswith("L-")
    assert out["version"] == 2

    # Lesson on disk has the right shape
    lesson_files = list((tmp_ledger / "lessons").glob("*.json"))
    assert any(
        json.loads(p.read_text())["kind"] == "dataset"
        for p in lesson_files
    )


def test_propose_queued_when_policy_review(monkeypatch, tmp_datasets, tmp_ledger):
    """Policy review + clean candidate → queued for human approval."""
    from harness.introspection import lib

    class _Pol:
        def mode_for(self, kind): return "review"
    monkeypatch.setattr("harness.approver.policy.Policy.from_yaml", lambda: _Pol())
    _patch_pool(monkeypatch)
    _seed_dataset("rag-gaps", tmp_datasets)
    _patch_proposer_to_return(monkeypatch)

    out = lib.propose_dataset_curation("rag-gaps")
    assert out["action"] == "queued"
    assert out["lesson_id"].startswith("L-")
    assert out["version"] == 2

    # Queued lesson exists with status=awaiting_review
    lessons = [json.loads(p.read_text())
               for p in (tmp_ledger / "lessons").glob("*.json")]
    assert any(l.get("status") == "awaiting_review" and l.get("kind") == "dataset"
               for l in lessons)


def test_propose_rejected_when_critic_blocks(
    monkeypatch, tmp_datasets, tmp_ledger,
):
    """Critic block → rejected Lesson written, no promotion."""
    from harness.introspection import lib
    from harness.types import CriticVerdict
    from harness.critics.dataset_critic import DatasetCritic

    class _Pol:
        def mode_for(self, kind): return "auto"
    monkeypatch.setattr("harness.approver.policy.Policy.from_yaml", lambda: _Pol())
    _patch_pool(monkeypatch)
    _seed_dataset("rag-gaps", tmp_datasets)
    _patch_proposer_to_return(monkeypatch)

    monkeypatch.setattr(
        DatasetCritic, "verdict",
        lambda self, ctx: CriticVerdict(
            critic="dataset_quality_gate", approved=False,
            reason="coverage_widened", severity="block",
        ),
    )

    out = lib.propose_dataset_curation("rag-gaps")
    assert out["action"] == "rejected"
    assert "coverage_widened" in out["reason"]
    assert out["lesson_id"].startswith("L-")


# ---------------------------------------------------------------------------
# Wakeup runner — multi-target health + extraction
# ---------------------------------------------------------------------------


@dataclass
class _FakeIntrospect:
    response: str = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class _FakeToolCall:
    tool: str
    input: dict
    output_preview: str


def test_wakeup_extracts_dataset_target(tmp_datasets, tmp_ledger, monkeypatch):
    """When the brain calls propose_dataset_curation, runner records target='dataset'."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    from harness.wakeup.runner import run_wakeup

    fake = _FakeIntrospect(
        response="Curating the rag-gaps dataset.",
        tool_calls=[_FakeToolCall(
            tool="propose_dataset_curation",
            input={"name": "rag-gaps", "rationale": "filling gaps"},
            output_preview='{"action":"promoted","lesson_id":"L-20260511-200000-abcd","version":2}',
        )],
    )
    outcome = run_wakeup(introspect_fn=lambda _p: fake)
    assert outcome.action == "proposed"
    assert outcome.target == "dataset"
    assert outcome.lesson_id == "L-20260511-200000-abcd"


def test_wakeup_extracts_router_target_still_works(tmp_datasets, tmp_ledger, monkeypatch):
    """Existing router path keeps working post-P15.4.5."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    from harness.wakeup.runner import run_wakeup

    fake = _FakeIntrospect(
        response="Refit needed.",
        tool_calls=[_FakeToolCall(
            tool="propose_router_retrain",
            input={"rationale": "drift up"},
            output_preview='{"action":"queued","lesson_id":"L-20260511-201500-abcd"}',
        )],
    )
    outcome = run_wakeup(introspect_fn=lambda _p: fake)
    assert outcome.action == "proposed"
    assert outcome.target == "router"
    assert outcome.lesson_id == "L-20260511-201500-abcd"


def test_wakeup_health_snapshot_carries_both_keys(tmp_datasets, tmp_ledger, monkeypatch):
    """Decision artifact includes router AND dataset health, regardless of action."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    from harness.wakeup.runner import run_wakeup

    _seed_dataset("rag-gaps", tmp_datasets)

    fake = _FakeIntrospect(response="Skipping for now.", tool_calls=[])
    outcome = run_wakeup(introspect_fn=lambda _p: fake)
    assert outcome.action == "skipped"
    assert "router" in outcome.health_snapshot
    assert "datasets" in outcome.health_snapshot
    # The dataset block lists our seeded dataset
    dataset_block = outcome.health_snapshot["datasets"]
    assert any(d.get("name") == "rag-gaps" for d in dataset_block.get("datasets", []))


def test_wakeup_skipped_has_no_target(tmp_datasets, tmp_ledger, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    from harness.wakeup.runner import run_wakeup
    fake = _FakeIntrospect(response="Nothing to do.", tool_calls=[])
    outcome = run_wakeup(introspect_fn=lambda _p: fake)
    assert outcome.action == "skipped"
    assert outcome.target is None
