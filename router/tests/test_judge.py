"""Tests for router.augmentation.judge.

Uses a FakeComplete that returns canned strings — no live brain needed.
Live-brain end-to-end is exercised by the OPENTRACY_RUN_SMOKE smoke test
(documented in PLAN_P15.3.5.md T9).
"""

from __future__ import annotations

import logging

import pytest

from router.augmentation.judge import (
    JudgeVerdict,
    LLMJudge,
    PAIRWISE_PROMPT,
    POINTWISE_PROMPT,
    PointwiseScore,
    _parse_pairwise,
    _parse_pointwise,
)


class _FakeComplete:
    """Captures call args + returns a canned response per call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if not self._responses:
            return ""
        return self._responses.pop(0)


# --- prompts loaded ---


def test_prompt_templates_loaded():
    """Both templates are populated from judge_prompt.md at module import."""
    assert "{prompt}" in PAIRWISE_PROMPT
    assert "{model_a}" in PAIRWISE_PROMPT
    assert "{model_b}" in PAIRWISE_PROMPT
    assert "WINNER" in PAIRWISE_PROMPT
    assert "{prompt}" in POINTWISE_PROMPT
    assert "{model_id}" in POINTWISE_PROMPT
    assert "SCORE" in POINTWISE_PROMPT


# --- parsers ---


def test_parse_pairwise_basic():
    text = "WINNER: A\nCONFIDENCE: 4\nREASON: clearer answer"
    winner, conf, reason = _parse_pairwise(text)
    assert winner == "A"
    assert conf == 4
    assert reason == "clearer answer"


def test_parse_pairwise_handles_tie():
    text = "WINNER: TIE\nCONFIDENCE: 3\nREASON: equivalent"
    winner, conf, reason = _parse_pairwise(text)
    assert winner == "TIE"
    assert conf == 3


def test_parse_pairwise_garbage_returns_empty_winner():
    """Honest failure mode: no WINNER line → empty string (NOT TIE)."""
    text = "i don't know"
    winner, _, reason = _parse_pairwise(text)
    assert winner == ""
    assert reason == "i don't know"


def test_parse_pairwise_clamps_confidence():
    text = "WINNER: B\nCONFIDENCE: 9\nREASON: x"
    _, conf, _ = _parse_pairwise(text)
    assert conf == 5


def test_parse_pointwise_basic():
    text = "SCORE: 4\nREASON: solid response"
    score, reason = _parse_pointwise(text)
    assert score == 4
    assert reason == "solid response"


def test_parse_pointwise_default_on_garbage():
    score, reason = _parse_pointwise("nonsense")
    assert score == 3
    assert reason == "nonsense"


# --- LLMJudge.compare ---


def test_judge_compare_uses_complete_fn():
    fake = _FakeComplete(["WINNER: A\nCONFIDENCE: 4\nREASON: clearer"])
    judge = LLMJudge(complete_fn=fake)
    verdict = judge.compare("Q?", "haiku", "ans-a", "sonnet", "ans-b")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.winner == "A"
    assert verdict.confidence == 4
    assert verdict.winner_model == "haiku"
    assert verdict.loser_model == "sonnet"
    assert verdict.judge_model.startswith("harness.brain")
    # The fake was called with the formatted prompt.
    assert "haiku" in fake.calls[0]["prompt"]
    assert "ans-a" in fake.calls[0]["prompt"]


def test_judge_compare_handles_complete_exception():
    """When complete_fn raises, judge returns a parse-error sentinel verdict."""

    def boom(*a, **kw):
        raise RuntimeError("transport failed")

    judge = LLMJudge(complete_fn=boom)
    verdict = judge.compare("Q?", "a", "ra", "b", "rb")
    assert verdict.winner == ""  # parse-error sentinel (NOT a fake TIE)
    assert "transport failed" in verdict.reasoning


def test_judge_compare_truncates_long_responses():
    fake = _FakeComplete(["WINNER: A\nCONFIDENCE: 1\nREASON: x"])
    judge = LLMJudge(complete_fn=fake, max_response_chars=10)
    long_resp = "x" * 100
    judge.compare("Q?", "a", long_resp, "b", "ans-b")
    sent_prompt = fake.calls[0]["prompt"]
    # Only 10 chars of the long response should appear.
    assert "x" * 10 in sent_prompt
    assert "x" * 100 not in sent_prompt


# --- LLMJudge.rate ---


def test_judge_rate_basic():
    fake = _FakeComplete(["SCORE: 5\nREASON: excellent"])
    judge = LLMJudge(complete_fn=fake)
    s = judge.rate("Q?", "haiku", "perfect ans")
    assert isinstance(s, PointwiseScore)
    assert s.score == 5
    assert s.reasoning == "excellent"
    assert s.loss == 0.0  # 5/5 = perfect → 0 loss


def test_judge_rate_handles_exception():
    def boom(*a, **kw):
        raise RuntimeError("nope")

    judge = LLMJudge(complete_fn=boom)
    s = judge.rate("Q?", "haiku", "ans")
    assert s.score == 3  # default on error
    assert "nope" in s.reasoning


# --- batch + max_samples cap ---


def test_compare_batch_caps_at_max_samples(caplog):
    fake = _FakeComplete(["WINNER: A\nCONFIDENCE: 1\nREASON: x"] * 1000)
    judge = LLMJudge(complete_fn=fake, max_samples=5)
    prompts = [f"q{i}" for i in range(20)]
    a_resps = [f"a{i}" for i in range(20)]
    b_resps = [f"b{i}" for i in range(20)]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="router.augmentation.judge"):
        out = judge.compare_batch(prompts, "haiku", a_resps, "sonnet", b_resps)

    assert len(out) == 5
    assert any("truncating" in r.message for r in caplog.records)


def test_rate_batch_caps_at_max_samples(caplog):
    fake = _FakeComplete(["SCORE: 4\nREASON: x"] * 1000)
    judge = LLMJudge(complete_fn=fake, max_samples=3)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="router.augmentation.judge"):
        out = judge.rate_batch(["p"] * 10, "haiku", ["r"] * 10)

    assert len(out) == 3
    assert any("truncating" in r.message for r in caplog.records)


def test_compare_batch_no_truncation_when_under_cap():
    fake = _FakeComplete(["WINNER: A\nCONFIDENCE: 3\nREASON: x"] * 5)
    judge = LLMJudge(complete_fn=fake, max_samples=500)
    out = judge.compare_batch(
        prompts=["p1", "p2", "p3"],
        model_a="a",
        responses_a=["a1", "a2", "a3"],
        model_b="b",
        responses_b=["b1", "b2", "b3"],
    )
    assert len(out) == 3


# --- winner_model / loser_model attribution ---


def test_judge_verdict_winner_loser_attribution():
    fake = _FakeComplete([
        "WINNER: A\nCONFIDENCE: 5\nREASON: x",
        "WINNER: B\nCONFIDENCE: 5\nREASON: x",
        "WINNER: TIE\nCONFIDENCE: 3\nREASON: x",
    ])
    judge = LLMJudge(complete_fn=fake)
    v_a = judge.compare("p", "ma", "ra", "mb", "rb")
    v_b = judge.compare("p", "ma", "ra", "mb", "rb")
    v_tie = judge.compare("p", "ma", "ra", "mb", "rb")

    assert v_a.winner_model == "ma"
    assert v_a.loser_model == "mb"
    assert v_b.winner_model == "mb"
    assert v_b.loser_model == "ma"
    assert v_tie.winner_model is None
    assert v_tie.loser_model is None
