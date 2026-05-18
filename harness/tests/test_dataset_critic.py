"""Tests for harness.critics.dataset_critic.DatasetCritic."""

from __future__ import annotations

import pytest

from experiments.types import Mutation
from harness.critics.dataset_critic import DatasetCritic
from harness.types import CriticContext, Proposal


def _sample(id_: str = "smp_0001", prompt: str = "hello", embedding=None) -> dict:
    return {
        "id": id_,
        "prompt": prompt,
        "ground_truth": "",
        "tag": "policy",
        "trace_id": None,
        "added_at": "2026-05-09T18:30:00Z",
        "source": "manual",
        "embedding": embedding if embedding is not None else [0.1, 0.2, 0.3, 0.4],
    }


def _proposal(payload: dict) -> Proposal:
    return Proposal(
        mutations=[Mutation(
            file=f"datasets/{payload.get('name', 'x')}/v{payload.get('version', 1)}.json",
            path="<inline_payload>",
            value=payload,
        )],
        description="test",
        source="claude_code",
    )


def _ctx(payload: dict) -> CriticContext:
    return CriticContext(proposal=_proposal(payload))


def _payload(*, samples=None, name="goldens", version=2, gap_before=0.3, gap_after=0.2) -> dict:
    return {
        "version": version,
        "name": name,
        "samples": samples if samples is not None else [_sample()],
        "metadata": {
            "gap_score_before": gap_before,
            "gap_score_after": gap_after,
            "added": 1,
        },
    }


# ---------------------------------------------------------------------------
# Approve path
# ---------------------------------------------------------------------------


def test_approves_clean_candidate():
    v = DatasetCritic().verdict(_ctx(_payload()))
    assert v.approved is True
    assert "added=1" in v.reason
    assert "gap_score 0.300 → 0.200" in v.reason


def test_approves_without_coverage_data():
    """No gap scores in metadata → coverage check skipped, still approves."""
    p = _payload()
    p["metadata"] = {"added": 1}  # no gap_score_*
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is True


# ---------------------------------------------------------------------------
# Block paths
# ---------------------------------------------------------------------------


def test_blocks_missing_payload_keys():
    p = {"version": 1}  # no name, no samples
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is False
    assert "missing required key" in v.reason


def test_blocks_sample_missing_keys():
    p = _payload(samples=[{"id": "s1", "prompt": "x"}])  # missing embedding+tag
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is False
    assert "schema_invalid" in v.reason
    assert "missing keys" in v.reason


def test_blocks_empty_prompt():
    p = _payload(samples=[_sample(prompt="   ")])
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is False
    assert "empty prompt" in v.reason


def test_blocks_empty_embedding():
    p = _payload(samples=[_sample(embedding=[])])
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is False
    assert "invalid embedding" in v.reason


def test_blocks_duplicate_ids():
    p = _payload(samples=[_sample("s1"), _sample("s1")])
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is False
    assert "duplicate_ids" in v.reason


def test_blocks_size_cap_exceeded():
    samples = [_sample(f"s{i}") for i in range(11)]
    p = _payload(samples=samples)
    critic = DatasetCritic(params={"max_total_samples": 10})
    v = critic.verdict(_ctx(p))
    assert v.approved is False
    assert "size_cap_exceeded" in v.reason


def test_blocks_coverage_widened_beyond_epsilon():
    # gap_after - gap_before = 0.1, epsilon=0.01 → block
    p = _payload(gap_before=0.2, gap_after=0.3)
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is False
    assert "coverage_widened" in v.reason


def test_approves_coverage_widened_within_epsilon():
    # gap_after - gap_before = 0.005, epsilon=0.01 → approve
    p = _payload(gap_before=0.2, gap_after=0.205)
    v = DatasetCritic().verdict(_ctx(p))
    assert v.approved is True


def test_blocks_empty_proposal():
    proposal = Proposal(mutations=[], description="", source="auto")
    v = DatasetCritic().verdict(CriticContext(proposal=proposal))
    assert v.approved is False
    assert "no mutations" in v.reason


def test_blocks_non_dict_payload():
    proposal = Proposal(
        mutations=[Mutation(file="datasets/x/v1.json", path="<x>", value="not a dict")],
        description="",
        source="auto",
    )
    v = DatasetCritic().verdict(CriticContext(proposal=proposal))
    assert v.approved is False
    assert "dict payload" in v.reason


# ---------------------------------------------------------------------------
# Optional suite re-run path
# ---------------------------------------------------------------------------


def test_run_suites_without_agent_path_blocks(tmp_path):
    """run_suites=True + at least one suite pointing at the dataset, but
    no agent_path → critic blocks with a clear reason."""
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "smoke.yaml").write_text(
        "suite: smoke\n"
        "dataset: goldens\n"
        "rubrics:\n  - {name: r, type: response_nonempty}\n"
    )
    p = _payload()
    critic = DatasetCritic(params={
        "run_suites": True,
        "suites_dir": str(suites),
    })
    v = critic.verdict(_ctx(p))
    assert v.approved is False
    assert "no agent_path" in v.reason


def test_run_suites_no_matching_suites_passes_through(tmp_path):
    """run_suites=True but no suite references this dataset → critic skips
    the expensive path and approves."""
    suites = tmp_path / "suites"
    suites.mkdir()
    p = _payload(name="orphan")
    critic = DatasetCritic(params={
        "run_suites": True,
        "suites_dir": str(suites),
    })
    v = critic.verdict(_ctx(p))
    assert v.approved is True
