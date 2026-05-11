"""Tests for the P15.4.3 dataset mining adapters.

Each adapter is exercised against synthetic trace / golden fixtures in
tmp_path so the tests don't depend on (or pollute) the real repo state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.proposer.dataset.mining import (
    available_sources,
    failed_lookups,
    feedback_signals,
    flagged_traces,
    get_adapter,
    language_router,
)
from harness.proposer.dataset.mining.base import prompt_hash


class _MockEmbedder:
    """Deterministic 4-dim embedder — stable per prompt."""

    def embed(self, prompt: str) -> list[float]:
        s = sum(ord(c) for c in prompt)
        return [(s % 7) / 10.0, (s % 11) / 10.0, (s % 13) / 10.0, (s % 17) / 10.0]


def _write_trace(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _stage(name: str, **kwargs) -> dict:
    return {"stage": name, "technique": kwargs.pop("technique", name), **kwargs}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_all_four_sources():
    labels = set(available_sources())
    assert labels == {"flagged traces", "language router", "failed lookups", "feedback signals"}


def test_get_adapter_returns_none_for_manual():
    assert get_adapter("manual") is None


def test_get_adapter_returns_callable_for_known():
    fn = get_adapter("failed lookups")
    assert callable(fn)


# ---------------------------------------------------------------------------
# failed_lookups
# ---------------------------------------------------------------------------


def test_failed_lookups_picks_docs_out_zero(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {
            "trace_id": "t1",
            "request": "weird order ID format",
            "stages": [_stage("retrieve", technique="rag", docs_in=0, docs_out=0)],
        },
        {
            "trace_id": "t2",
            "request": "normal lookup",
            "stages": [_stage("retrieve", technique="rag", docs_in=0, docs_out=5)],
        },
    ])
    samples = list(failed_lookups.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
    ))
    assert len(samples) == 1
    assert samples[0].prompt == "weird order ID format"
    assert samples[0].tag == "failed_lookup"
    assert samples[0].trace_id == "t1"
    assert samples[0].source == "failed lookups"
    assert len(samples[0].embedding) == 4


def test_failed_lookups_dedup_within_call(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {
            "trace_id": "t1",
            "request": "same prompt",
            "stages": [_stage("retrieve", technique="rag", docs_out=0)],
        },
        {
            "trace_id": "t2",
            "request": "same prompt",
            "stages": [_stage("retrieve", technique="rag", docs_out=0)],
        },
    ])
    samples = list(failed_lookups.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
    ))
    assert len(samples) == 1, "duplicates should be filtered"


def test_failed_lookups_skips_existing(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {
            "trace_id": "t1",
            "request": "skip me",
            "stages": [_stage("retrieve", technique="rag", docs_out=0)],
        },
    ])
    existing = {prompt_hash("skip me", "failed_lookup")}
    samples = list(failed_lookups.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
        existing=existing,
    ))
    assert samples == []


def test_failed_lookups_respects_limit(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {
            "trace_id": f"t{i}",
            "request": f"prompt {i}",
            "stages": [_stage("retrieve", technique="rag", docs_out=0)],
        }
        for i in range(5)
    ])
    samples = list(failed_lookups.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
        limit=2,
    ))
    assert len(samples) == 2


def test_failed_lookups_date_window(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-01.jsonl", [
        {"trace_id": "old", "request": "p old", "stages": [_stage("retrieve", technique="rag", docs_out=0)]},
    ])
    _write_trace(raw_dir / "2026-05-10.jsonl", [
        {"trace_id": "new", "request": "p new", "stages": [_stage("retrieve", technique="rag", docs_out=0)]},
    ])
    samples = list(failed_lookups.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
        since_iso="2026-05-05T00:00:00Z",
    ))
    assert [s.prompt for s in samples] == ["p new"]


def test_failed_lookups_missing_dir(tmp_path: Path):
    """No traces/raw dir → empty stream, no error."""
    samples = list(failed_lookups.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=tmp_path / "nope",
    ))
    assert samples == []


# ---------------------------------------------------------------------------
# language_router
# ---------------------------------------------------------------------------


def test_language_router_metadata_lang_tag(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {"trace_id": "t1", "request": "where is my order", "metadata": {"language": "pt-BR"}, "stages": []},
        {"trace_id": "t2", "request": "where is my order", "metadata": {"language": "en"}, "stages": []},
    ])
    samples = list(language_router.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
    ))
    # t1 only (en is excluded). t2 has different tag (None vs pt-BR), so the
    # exclusion is by the classifier returning None, not by hash collision.
    assert len(samples) == 1
    assert samples[0].tag == "pt-br"


def test_language_router_heuristic_catches_non_ascii(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {"trace_id": "t1", "request": "meu pedido não chegou ainda", "stages": []},
        {"trace_id": "t2", "request": "where is my order at?", "stages": []},
        {"trace_id": "t3", "request": "¿cuándo llega mi pedido?", "stages": []},
    ])
    samples = list(language_router.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
    ))
    prompts = sorted([s.prompt for s in samples])
    assert prompts == ["meu pedido não chegou ainda", "¿cuándo llega mi pedido?"]
    assert all(s.tag == "non_en" for s in samples)
    assert all(s.source == "language router" for s in samples)


def test_language_router_letters_only(tmp_path: Path):
    """Punctuation-only or numeric-only prompts shouldn't trigger."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_trace(raw_dir / "2026-05-09.jsonl", [
        {"trace_id": "t1", "request": "123 456 789", "stages": []},
        {"trace_id": "t2", "request": "!@#$%^", "stages": []},
    ])
    samples = list(language_router.iter_candidates(
        embedder=_MockEmbedder(),
        traces_root=raw_dir,
    ))
    assert samples == []


# ---------------------------------------------------------------------------
# flagged_traces
# ---------------------------------------------------------------------------


def test_flagged_traces_promoted_goldens(tmp_path: Path):
    goldens = tmp_path / "goldens"
    goldens.mkdir()
    (goldens / "g1.yaml").write_text(
        "id: g1\n"
        "input:\n  request: \"flagged prompt about refund\"\n"
        "expected:\n  contains: [\"refund\"]\n  category: refund\n"
        "metadata:\n  source: trace:abc-123\n"
    )
    # A non-trace golden should be ignored
    (goldens / "g2.yaml").write_text(
        "id: g2\n"
        "input:\n  request: \"normal seed golden\"\n"
        "expected:\n  contains: [\"x\"]\n"
        "metadata:\n  source: seed\n"
    )
    samples = list(flagged_traces.iter_candidates(
        embedder=_MockEmbedder(),
        goldens_dir=goldens,
        pinned_dir=tmp_path / "nope",
    ))
    assert len(samples) == 1
    s = samples[0]
    assert s.prompt == "flagged prompt about refund"
    assert s.trace_id == "abc-123"
    assert s.tag == "refund"
    assert s.source == "flagged traces"
    assert s.ground_truth == "refund"


def test_flagged_traces_pinned_jsonl(tmp_path: Path):
    goldens = tmp_path / "goldens"  # not created
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    _write_trace(pinned / "2026-05-09.jsonl", [
        {"trace_id": "p1", "request": "pinned prompt one", "flag_reason": "manual review"},
        {"trace_id": "p2", "prompt": "pinned prompt two"},
    ])
    samples = list(flagged_traces.iter_candidates(
        embedder=_MockEmbedder(),
        goldens_dir=goldens,
        pinned_dir=pinned,
    ))
    prompts = sorted([s.prompt for s in samples])
    assert prompts == ["pinned prompt one", "pinned prompt two"]
    by_prompt = {s.prompt: s for s in samples}
    assert by_prompt["pinned prompt one"].tag == "manual review"
    assert by_prompt["pinned prompt two"].tag == "flagged"


def test_flagged_traces_merges_both_sources(tmp_path: Path):
    goldens = tmp_path / "goldens"
    goldens.mkdir()
    (goldens / "g1.yaml").write_text(
        "id: g1\n"
        "input:\n  request: \"from golden\"\n"
        "expected:\n  contains: [\"x\"]\n"
        "metadata:\n  source: trace:abc\n"
    )
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    _write_trace(pinned / "2026-05-09.jsonl", [{"trace_id": "p1", "request": "from pinned"}])
    samples = list(flagged_traces.iter_candidates(
        embedder=_MockEmbedder(),
        goldens_dir=goldens,
        pinned_dir=pinned,
    ))
    prompts = sorted(s.prompt for s in samples)
    assert prompts == ["from golden", "from pinned"]


# ---------------------------------------------------------------------------
# feedback_signals (stub)
# ---------------------------------------------------------------------------


def test_feedback_signals_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        # generator: must materialize to trigger
        list(feedback_signals.iter_candidates(embedder=_MockEmbedder()))


# ---------------------------------------------------------------------------
# Common dedup hash
# ---------------------------------------------------------------------------


def test_prompt_hash_stable_across_adapters():
    """Same prompt+tag → same ID regardless of which adapter produces it.

    This is what lets the proposer dedup across sources.
    """
    h1 = prompt_hash("hello world", "policy")
    h2 = prompt_hash("hello world", "policy")
    assert h1 == h2
    assert prompt_hash("hello world", "billing") != h1
    assert prompt_hash("hello world", None) != h1
