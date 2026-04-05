"""
LLM-as-Judge for evaluating model response quality.

Supports pairwise comparison (A vs B) and pointwise scoring (rate 1-5).
Used to generate training signal for prompts without ground truth —
especially production traces from ClickHouse.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging
import re

from ..models.llm_client import LLMClient

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT = """You are an expert judge comparing two AI responses to the same prompt.

**User Prompt:**
{prompt}

**Response A** (from {model_a}):
{response_a}

**Response B** (from {model_b}):
{response_b}

Compare both responses on: accuracy, completeness, clarity, and helpfulness.

Reply with EXACTLY one line in this format:
WINNER: A|B|TIE
CONFIDENCE: 1-5
REASON: <one sentence>"""

POINTWISE_PROMPT = """You are an expert evaluator rating an AI response.

**User Prompt:**
{prompt}

**Response** (from {model_id}):
{response}

Rate this response on a scale of 1-5:
1 = Incorrect, unhelpful, or harmful
2 = Partially correct but major issues
3 = Acceptable but could be better
4 = Good, mostly correct and helpful
5 = Excellent, accurate and comprehensive

Reply with EXACTLY one line in this format:
SCORE: 1-5
REASON: <one sentence>"""


@dataclass
class JudgeVerdict:
    """Result of an LLM judge evaluation."""

    prompt: str
    model_a: str
    model_b: str
    winner: str  # "A", "B", or "TIE"
    confidence: int  # 1-5
    reasoning: str
    judge_model: str

    @property
    def winner_model(self) -> Optional[str]:
        if self.winner == "A":
            return self.model_a
        elif self.winner == "B":
            return self.model_b
        return None

    @property
    def loser_model(self) -> Optional[str]:
        if self.winner == "A":
            return self.model_b
        elif self.winner == "B":
            return self.model_a
        return None


@dataclass
class PointwiseScore:
    """Result of a pointwise evaluation."""

    prompt: str
    model_id: str
    score: int  # 1-5
    reasoning: str
    judge_model: str

    @property
    def loss(self) -> float:
        """Convert to loss value (0.0 = perfect, 1.0 = worst)."""
        return 1.0 - (self.score - 1) / 4.0


class LLMJudge:
    """
    Uses an LLM to judge response quality.

    Two modes:
    - Pairwise: Compare two model responses, pick a winner.
    - Pointwise: Rate a single response 1-5.

    Usage:
        judge = LLMJudge(judge_client)

        # Pairwise
        verdict = judge.compare(prompt, "gpt-4o", resp_a, "llama-8b", resp_b)
        print(verdict.winner)  # "A" or "B"

        # Pointwise
        score = judge.rate(prompt, "gpt-4o-mini", response)
        print(score.score)  # 1-5
    """

    def __init__(
        self,
        judge_client: LLMClient,
        max_response_chars: int = 2000,
    ):
        """
        Args:
            judge_client: LLM client to use as judge (e.g., GPT-4o, Claude Sonnet).
            max_response_chars: Truncate responses to this length for judging.
        """
        self.judge_client = judge_client
        self.max_response_chars = max_response_chars

    def compare(
        self,
        prompt: str,
        model_a: str,
        response_a: str,
        model_b: str,
        response_b: str,
    ) -> JudgeVerdict:
        """
        Pairwise comparison: which response is better?

        Args:
            prompt: The original user prompt.
            model_a: Model ID for response A.
            response_a: Text of response A.
            model_b: Model ID for response B.
            response_b: Text of response B.

        Returns:
            JudgeVerdict with winner, confidence, and reasoning.
        """
        judge_prompt = PAIRWISE_PROMPT.format(
            prompt=prompt[:1000],
            model_a=model_a,
            response_a=response_a[:self.max_response_chars],
            model_b=model_b,
            response_b=response_b[:self.max_response_chars],
        )

        try:
            result = self.judge_client.generate(judge_prompt, max_tokens=100, temperature=0.0)
            winner, confidence, reasoning = _parse_pairwise(result.text)
        except Exception as e:
            logger.warning(f"Judge failed for {model_a} vs {model_b}: {e}")
            winner, confidence, reasoning = "TIE", 1, f"Judge error: {e}"

        return JudgeVerdict(
            prompt=prompt,
            model_a=model_a,
            model_b=model_b,
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            judge_model=self.judge_client.model_id,
        )

    def rate(
        self,
        prompt: str,
        model_id: str,
        response: str,
    ) -> PointwiseScore:
        """
        Pointwise scoring: rate a single response 1-5.

        Args:
            prompt: The original user prompt.
            model_id: Model that generated the response.
            response: The model's response text.

        Returns:
            PointwiseScore with score 1-5 and reasoning.
        """
        judge_prompt = POINTWISE_PROMPT.format(
            prompt=prompt[:1000],
            model_id=model_id,
            response=response[:self.max_response_chars],
        )

        try:
            result = self.judge_client.generate(judge_prompt, max_tokens=80, temperature=0.0)
            score, reasoning = _parse_pointwise(result.text)
        except Exception as e:
            logger.warning(f"Judge failed rating {model_id}: {e}")
            score, reasoning = 3, f"Judge error: {e}"

        return PointwiseScore(
            prompt=prompt,
            model_id=model_id,
            score=score,
            reasoning=reasoning,
            judge_model=self.judge_client.model_id,
        )

    def compare_batch(
        self,
        prompts: list[str],
        model_a: str,
        responses_a: list[str],
        model_b: str,
        responses_b: list[str],
    ) -> list[JudgeVerdict]:
        """Compare multiple prompt pairs."""
        return [
            self.compare(p, model_a, ra, model_b, rb)
            for p, ra, rb in zip(prompts, responses_a, responses_b)
        ]

    def rate_batch(
        self,
        prompts: list[str],
        model_id: str,
        responses: list[str],
    ) -> list[PointwiseScore]:
        """Rate multiple responses."""
        return [
            self.rate(p, model_id, r)
            for p, r in zip(prompts, responses)
        ]


def _parse_pairwise(text: str) -> tuple[str, int, str]:
    """Parse pairwise judge output."""
    winner = "TIE"
    confidence = 3
    reasoning = text.strip()

    winner_match = re.search(r"WINNER:\s*(A|B|TIE)", text, re.IGNORECASE)
    if winner_match:
        winner = winner_match.group(1).upper()

    conf_match = re.search(r"CONFIDENCE:\s*(\d)", text)
    if conf_match:
        confidence = max(1, min(5, int(conf_match.group(1))))

    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return winner, confidence, reasoning


def _parse_pointwise(text: str) -> tuple[int, str]:
    """Parse pointwise judge output."""
    score = 3
    reasoning = text.strip()

    score_match = re.search(r"SCORE:\s*(\d)", text)
    if score_match:
        score = max(1, min(5, int(score_match.group(1))))

    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return score, reasoning
