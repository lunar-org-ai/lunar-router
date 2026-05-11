"""End-to-end tests for promote_dataset: proposal → apply → Lesson."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.types import Mutation
from harness.executor.promote import apply_dataset_candidate, promote_dataset
from harness.types import CriticVerdict, LoopOutcome, Prediction, Proposal


@pytest.fixture
def tmp_datasets(tmp_path: Path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr("router.data.dataset_io.DEFAULT_DATASETS_DIR", d)
    return d


@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch):
    """Patch ledger writer paths.

    write_entry's entries_dir is kw-only → patch __kwdefaults__.
    write_lesson's lessons_dir is positional → patch __defaults__.
    """
    import ledger.writer as lw

    entries = tmp_path / "ledger" / "entries"
    lessons = tmp_path / "ledger" / "lessons"
    entries.mkdir(parents=True)
    lessons.mkdir(parents=True)
    monkeypatch.setattr("ledger.writer.ENTRIES_DIR", entries)
    monkeypatch.setattr("ledger.writer.LESSONS_DIR", lessons)

    # write_entry: entries_dir is keyword-only.
    we_kwdef = dict(lw.write_entry.__kwdefaults__ or {})
    if "entries_dir" in we_kwdef:
        we_kwdef["entries_dir"] = entries
        monkeypatch.setattr(lw.write_entry, "__kwdefaults__", we_kwdef)

    # write_lesson: lessons_dir is positional.
    wl_defaults = list(lw.write_lesson.__defaults__ or ())
    if wl_defaults:
        wl_defaults[-1] = lessons
        monkeypatch.setattr(lw.write_lesson, "__defaults__", tuple(wl_defaults))
    return tmp_path / "ledger"


def _payload(*, name="goldens", version=1, n_samples=2) -> dict:
    return {
        "version": version,
        "name": name,
        "desc": "test",
        "source": "failed lookups",
        "sourceType": "auto",
        "use": ["Eval"],
        "owner": "agent",
        "growing": True,
        "embedder_model": "test",
        "embedding_dim": 4,
        "samples": [
            {
                "id": f"smp_{i:04d}",
                "prompt": f"p{i}",
                "ground_truth": "",
                "tag": "tag",
                "trace_id": None,
                "added_at": "2026-05-09T18:30:00Z",
                "source": "failed lookups",
                "embedding": [0.1, 0.2, 0.3, 0.4],
            }
            for i in range(n_samples)
        ],
        "history": [{"when": "2026-05-09T18:30:00Z", "what": "seed"}],
        "metadata": {
            "added": n_samples,
            "gap_score_before": 0.4,
            "gap_score_after": 0.2,
            "adapter_source": "failed lookups",
        },
    }


def _outcome(payload: dict, *, with_prediction: bool = False) -> LoopOutcome:
    prediction = None
    if with_prediction:
        prediction = Prediction(
            rubric="coverage_gap_score",
            expected_delta=-0.2,  # gap shrinks from 0.4 → 0.2
            rationale="Mined 2 samples to fill gaps.",
            confidence=0.45,
        )
    proposal = Proposal(
        mutations=[Mutation(
            file=f"datasets/{payload['name']}/v{payload['version']}.json",
            path="<inline_payload>",
            value=payload,
        )],
        description="test",
        source="claude_code",
        prediction=prediction,
    )
    return LoopOutcome(
        proposal=proposal,
        candidate_id="cand_xyz",
        verdicts=[CriticVerdict(
            critic="dataset_quality_gate",
            approved=True,
            reason="added=2",
            severity="info",
        )],
        final="approved",
    )


# ---------------------------------------------------------------------------
# apply_dataset_candidate
# ---------------------------------------------------------------------------


def test_apply_writes_file_and_flips_pointer(tmp_datasets):
    payload = _payload(version=1, n_samples=3)
    json_path = apply_dataset_candidate(payload)
    assert json_path.name == "v1.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["version"] == 1
    assert len(on_disk["samples"]) == 3
    # pointer
    base = tmp_datasets / "goldens"
    assert (base / "current").exists() or (base / "current.txt").exists()


# ---------------------------------------------------------------------------
# promote_dataset
# ---------------------------------------------------------------------------


def test_promote_returns_version_and_lesson_id(tmp_datasets, tmp_ledger):
    payload = _payload(version=2, n_samples=2)
    outcome = _outcome(payload)
    new_version, lesson_id = promote_dataset(outcome)
    assert new_version == 2
    assert lesson_id.startswith("L-")


def test_promote_writes_lesson_with_kind_dataset(tmp_datasets, tmp_ledger):
    payload = _payload(version=1, n_samples=2)
    outcome = _outcome(payload)
    _new_version, lesson_id = promote_dataset(outcome)

    lesson_path = tmp_ledger / "lessons" / f"{lesson_id}.json"
    assert lesson_path.exists()
    lesson = json.loads(lesson_path.read_text())
    assert lesson["kind"] == "dataset"
    assert lesson["proposal_source"] == "claude_code"
    assert lesson["status"] == "auto_promoted"
    assert "goldens" in lesson["summary"]
    assert lesson["delta"]["added"] == 2.0
    assert lesson["delta"]["gap_score_before"] == 0.4
    assert lesson["delta"]["gap_score_after"] == 0.2


def test_promote_writes_ledger_entry(tmp_datasets, tmp_ledger):
    payload = _payload(version=1, n_samples=2)
    outcome = _outcome(payload)
    promote_dataset(outcome)

    entry_files = list((tmp_ledger / "entries").glob("*.jsonl"))
    assert len(entry_files) == 1
    lines = entry_files[0].read_text().strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["kind"] == "promote"
    assert entry["payload"]["kind"] == "dataset"
    assert entry["payload"]["name"] == "goldens"
    assert entry["payload"]["added"] == 2


def test_promote_handles_missing_metadata_gracefully(tmp_datasets, tmp_ledger):
    """No gap_score_before/after in metadata → Lesson still writes, just no gap text."""
    payload = _payload()
    payload["metadata"] = {"added": 2}  # no gap_score_*
    outcome = _outcome(payload)
    _new, lesson_id = promote_dataset(outcome)
    lesson = json.loads(
        (tmp_ledger / "lessons" / f"{lesson_id}.json").read_text()
    )
    assert "gap_score" not in lesson["summary"]


def test_promote_rejects_empty_mutations(tmp_datasets, tmp_ledger):
    proposal = Proposal(mutations=[], description="", source="auto")
    outcome = LoopOutcome(proposal=proposal, candidate_id=None, final="approved")
    with pytest.raises(ValueError, match="mutations"):
        promote_dataset(outcome)


def test_promote_rejects_non_dict_payload(tmp_datasets, tmp_ledger):
    proposal = Proposal(
        mutations=[Mutation(file="datasets/x/v1.json", path="<x>", value="not a dict")],
        description="",
        source="auto",
    )
    outcome = LoopOutcome(proposal=proposal, candidate_id=None, final="approved")
    with pytest.raises(ValueError, match="dict payload"):
        promote_dataset(outcome)


# ---------------------------------------------------------------------------
# kind_from_mutations
# ---------------------------------------------------------------------------


def test_kind_from_mutations_recognizes_datasets():
    from harness.types import kind_from_mutations
    assert kind_from_mutations(["datasets/goldens/v2.json"]) == "dataset"
    assert kind_from_mutations(["datasets/rag-gaps/v5.json"]) == "dataset"
    # Other kinds still work
    assert kind_from_mutations(["versions/router_config_v3.json"]) == "router_config"
    assert kind_from_mutations(["agent/prompts/system.md"]) == "prompt"


# ---------------------------------------------------------------------------
# AHE Pillar 3 — VerificationOutcome
# ---------------------------------------------------------------------------


def test_promote_computes_verification_when_prediction_present_cold_router(
    tmp_datasets, tmp_ledger,
):
    """With a Prediction attached but no router config fitted, verification
    is degenerate — actual_delta := expected_delta, verdict='verified'."""
    payload = _payload(version=1, n_samples=2)
    outcome = _outcome(payload, with_prediction=True)
    _new_version, lesson_id = promote_dataset(outcome)

    # outcome was mutated with the materialized verification
    assert outcome.verification is not None
    assert outcome.verification.rubric == "coverage_gap_score"
    assert outcome.verification.verdict in {"verified", "no_change", "partial", "wrong"}

    # Ledger entry has verification block
    entry_files = list((tmp_ledger / "entries").glob("*.jsonl"))
    entry = json.loads(entry_files[0].read_text().strip().split("\n")[0])
    assert "verification" in entry["payload"]
    assert entry["payload"]["verification"]["rubric"] == "coverage_gap_score"

    # Lesson summary mentions the verdict
    lesson = json.loads((tmp_ledger / "lessons" / f"{lesson_id}.json").read_text())
    assert "Prediction" in lesson["summary"]


def test_promote_no_prediction_no_verification(tmp_datasets, tmp_ledger):
    """No Prediction → no verification recorded."""
    payload = _payload(version=1, n_samples=2)
    outcome = _outcome(payload, with_prediction=False)
    _, lesson_id = promote_dataset(outcome)

    assert outcome.verification is None

    entry_files = list((tmp_ledger / "entries").glob("*.jsonl"))
    entry = json.loads(entry_files[0].read_text().strip().split("\n")[0])
    assert "verification" not in entry["payload"]

    lesson = json.loads((tmp_ledger / "lessons" / f"{lesson_id}.json").read_text())
    assert "Prediction" not in lesson["summary"]


def test_promote_does_not_overwrite_caller_supplied_verification(
    tmp_datasets, tmp_ledger,
):
    """If the loop already computed a VerificationOutcome upstream, we
    don't recompute it — caller wins."""
    from harness.types import VerificationOutcome

    payload = _payload(version=1, n_samples=2)
    outcome = _outcome(payload, with_prediction=True)
    # Caller pre-set verification with a specific verdict
    outcome.verification = VerificationOutcome(
        rubric="coverage_gap_score",
        expected_delta=-0.2,
        actual_delta=-0.5,
        direction_correct=True,
        magnitude_met=True,
        verdict="verified",
    )
    promote_dataset(outcome)
    # Same instance — not replaced
    assert outcome.verification.actual_delta == -0.5
