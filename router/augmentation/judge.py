"""LLM-as-judge for evaluating model response quality.

Two modes:
- Pairwise: compare two model responses, pick a winner.
- Pointwise: rate a single response 1-5.

P15.3 rewires the constructor to take a ``complete_fn`` callable instead
of an ``LLMClient`` — same brain transport the harness already uses
(``harness.brain.transport.complete``). One brain, no second cerebro.

Templates live in ``judge_prompt.md`` (loaded at module import) so we
can iterate on phrasing without touching the dispatch logic.

Caps batch operations at ``max_samples`` (default 500 — the locked
P15.3 augmentation budget) and emits a WARNING when truncation kicks in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from harness.brain.transport import complete


logger = logging.getLogger("router.augmentation.judge")


DEFAULT_MAX_SAMPLES = 500
DEFAULT_MAX_RESPONSE_CHARS = 2000
_PROMPT_FILE = Path(__file__).parent / "judge_prompt.md"


def _load_templates(path: Path = _PROMPT_FILE) -> tuple[str, str]:
    """Read judge_prompt.md and split on ``---`` into (pairwise, pointwise).

    Each section is the everything-after-the-h1 of that section. We trim
    the leading ``# Pairwise`` / ``# Pointwise`` heading so format-string
    substitution sees only the body.
    """
    raw = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in raw.split("\n---\n")]
    if len(parts) < 2:
        raise RuntimeError(
            f"{path} must contain two sections separated by a '---' line"
        )
    pairwise_block, pointwise_block = parts[0], parts[1]
    return _strip_heading(pairwise_block), _strip_heading(pointwise_block)


def _strip_heading(block: str) -> str:
    """Drop the leading '# Heading' line from a block, keep the body."""
    lines = block.split("\n")
    # Drop the first heading line (and any blank lines after it).
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines)


PAIRWISE_PROMPT, POINTWISE_PROMPT = _load_templates()


# ---------------------------------------------------------------------------
# Verdicts / scores
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    """Result of an LLM judge pairwise evaluation."""

    prompt: str
    model_a: str
    model_b: str
    winner: str  # "A", "B", "TIE", or "" (parse error)
    confidence: int  # 1-5
    reasoning: str
    judge_model: str

    @property
    def winner_model(self) -> Optional[str]:
        if self.winner == "A":
            return self.model_a
        if self.winner == "B":
            return self.model_b
        return None

    @property
    def loser_model(self) -> Optional[str]:
        if self.winner == "A":
            return self.model_b
        if self.winner == "B":
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
        """Convert to loss value (0.0 = perfect 5/5, 1.0 = worst 1/5)."""
        return 1.0 - (self.score - 1) / 4.0


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Uses the harness brain to judge response quality.

    Construction accepts a ``complete_fn`` callable (defaulting to
    ``harness.brain.transport.complete``). Tests pass a fake to avoid
    touching a live brain.

    Usage:
        judge = LLMJudge()                           # uses harness brain
        verdict = judge.compare(prompt, "haiku", a, "sonnet", b)
        score = judge.rate(prompt, "haiku", text)
    """

    def __init__(
        self,
        complete_fn: Callable[..., str] = complete,
        *,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
        judge_model: Optional[str] = None,
    ):
        self.complete_fn = complete_fn
        self.max_samples = max_samples
        self.max_response_chars = max_response_chars
        # judge_model is informational — recorded on the verdict for
        # provenance. The actual model is whatever transport.complete
        # decides (env var, then API, then CLI).
        self.judge_model = judge_model or "harness.brain.complete"

    def compare(
        self,
        prompt: str,
        model_a: str,
        response_a: str,
        model_b: str,
        response_b: str,
    ) -> JudgeVerdict:
        """Pairwise: which response is better?"""
        judge_prompt = PAIRWISE_PROMPT.format(
            prompt=prompt[:1000],
            model_a=model_a,
            response_a=response_a[: self.max_response_chars],
            model_b=model_b,
            response_b=response_b[: self.max_response_chars],
        )

        try:
            text = self.complete_fn(
                judge_prompt,
                max_tokens=200,
                temperature=0.0,
                model=None,  # let transport pick
            )
        except Exception as e:
            logger.warning("judge failed for %s vs %s: %s", model_a, model_b, e)
            return JudgeVerdict(
                prompt=prompt,
                model_a=model_a,
                model_b=model_b,
                winner="",  # explicit parse-error sentinel (no false TIE)
                confidence=1,
                reasoning=f"Judge error: {e}",
                judge_model=self.judge_model,
            )

        winner, confidence, reasoning = _parse_pairwise(text)
        return JudgeVerdict(
            prompt=prompt,
            model_a=model_a,
            model_b=model_b,
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            judge_model=self.judge_model,
        )

    def rate(
        self,
        prompt: str,
        model_id: str,
        response: str,
    ) -> PointwiseScore:
        """Pointwise: rate a single response 1-5."""
        judge_prompt = POINTWISE_PROMPT.format(
            prompt=prompt[:1000],
            model_id=model_id,
            response=response[: self.max_response_chars],
        )

        try:
            text = self.complete_fn(
                judge_prompt,
                max_tokens=200,
                temperature=0.0,
                model=None,
            )
        except Exception as e:
            logger.warning("judge failed rating %s: %s", model_id, e)
            return PointwiseScore(
                prompt=prompt,
                model_id=model_id,
                score=3,
                reasoning=f"Judge error: {e}",
                judge_model=self.judge_model,
            )

        score, reasoning = _parse_pointwise(text)
        return PointwiseScore(
            prompt=prompt,
            model_id=model_id,
            score=score,
            reasoning=reasoning,
            judge_model=self.judge_model,
        )

    # --- Batch ops with the locked 500-cap ---------------------------------

    def compare_batch(
        self,
        prompts: list[str],
        model_a: str,
        responses_a: list[str],
        model_b: str,
        responses_b: list[str],
    ) -> list[JudgeVerdict]:
        """Compare multiple prompt pairs. Caps at ``self.max_samples``."""
        prompts, responses_a, responses_b = self._truncate_three(
            prompts, responses_a, responses_b, label="compare_batch"
        )
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
        """Rate multiple responses. Caps at ``self.max_samples``."""
        prompts, responses = self._truncate_two(prompts, responses, label="rate_batch")
        return [self.rate(p, model_id, r) for p, r in zip(prompts, responses)]

    def _truncate_two(
        self, a: list, b: list, *, label: str
    ) -> tuple[list, list]:
        if len(a) > self.max_samples:
            logger.warning(
                "%s: truncating from %d to max_samples=%d",
                label, len(a), self.max_samples,
            )
            a = a[: self.max_samples]
            b = b[: self.max_samples]
        return a, b

    def _truncate_three(
        self, a: list, b: list, c: list, *, label: str
    ) -> tuple[list, list, list]:
        if len(a) > self.max_samples:
            logger.warning(
                "%s: truncating from %d to max_samples=%d",
                label, len(a), self.max_samples,
            )
            a = a[: self.max_samples]
            b = b[: self.max_samples]
            c = c[: self.max_samples]
        return a, b, c


# ---------------------------------------------------------------------------
# Parsers (regex-based; tolerant of small format drift)
# ---------------------------------------------------------------------------


def _parse_pairwise(text: str) -> tuple[str, int, str]:
    """Parse a pairwise judge response.

    Honest failure mode: if WINNER can't be extracted, return ``""`` (not
    ``"TIE"`` — a true tie is an explicit verdict). Callers can then
    decide whether to drop the row or treat it as low-signal noise.
    """
    text = text or ""
    winner = ""
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
    """Parse a pointwise judge response. Defaults to score=3 on parse fail."""
    text = text or ""
    score = 3
    reasoning = text.strip()

    score_match = re.search(r"SCORE:\s*(\d)", text)
    if score_match:
        score = max(1, min(5, int(score_match.group(1))))

    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return score, reasoning
