"""Preference data structures for router training.

A PreferencePair says: ``winner_model > loser_model on this prompt``.
PreferenceDataset is a bag of pairs with serialization + win-rate
computation.

Sources:
- Benchmark cache: model A correct, model B wrong → A wins.
- LLM judge verdicts: judge picked A over B (P15.3.5's LLMJudge).
- Pointwise scores: higher score wins (configurable diff threshold).

Ported from the reference impl. The judge / cache references are
intentionally untyped at module level to avoid circular imports
(``add_from_cache`` lazily checks duck-typed inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import json
import logging


logger = logging.getLogger("router.augmentation.preference")


@dataclass
class PreferencePair:
    """A single pairwise preference: winner_model > loser_model on this prompt."""

    prompt: str
    winner_model: str
    loser_model: str
    source: str = "benchmark"  # "benchmark" | "judge" | "trace" | "feedback"
    confidence: float = 1.0  # 0.0 - 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PreferenceDataset:
    """Collection of preference pairs for training."""

    def __init__(
        self,
        pairs: Optional[list[PreferencePair]] = None,
        name: str = "preferences",
    ):
        self.pairs = pairs or []
        self.name = name

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def add(self, pair: PreferencePair) -> None:
        self.pairs.append(pair)

    def add_from_cache(
        self,
        response_cache,
        model_ids: Optional[list[str]] = None,
    ) -> int:
        """Generate preference pairs from cached benchmark responses.

        For each prompt, compare all model pairs. If model A got it
        right (loss=0) and model B got it wrong (loss>0), A wins.
        Returns the count of pairs added.

        Note: ``response_cache`` is duck-typed — must expose
        ``model_ids``, ``prompt_hashes``, and
        ``get_all_models_by_hash(h)`` returning a dict[model_id ->
        entry-with-.loss]. P15.3.6's ResponseCache will provide these.
        """
        count = 0
        mids = model_ids or sorted(response_cache.model_ids)

        for prompt_hash in response_cache.prompt_hashes:
            models = response_cache.get_all_models_by_hash(prompt_hash)

            for i, mid_a in enumerate(mids):
                for mid_b in mids[i + 1:]:
                    entry_a = models.get(mid_a)
                    entry_b = models.get(mid_b)
                    if entry_a is None or entry_b is None:
                        continue

                    if entry_a.loss == 0.0 and entry_b.loss > 0.0:
                        self.add(PreferencePair(
                            prompt=f"__hash__{prompt_hash}",
                            winner_model=mid_a,
                            loser_model=mid_b,
                            source="benchmark",
                            confidence=1.0,
                        ))
                        count += 1
                    elif entry_b.loss == 0.0 and entry_a.loss > 0.0:
                        self.add(PreferencePair(
                            prompt=f"__hash__{prompt_hash}",
                            winner_model=mid_b,
                            loser_model=mid_a,
                            source="benchmark",
                            confidence=1.0,
                        ))
                        count += 1

        logger.info("generated %d preference pairs from cache", count)
        return count

    def add_from_verdicts(self, verdicts: list) -> int:
        """Add pairs from a list of JudgeVerdict.

        Verdicts with ``winner == "TIE"`` or ``winner == ""``
        (parse-error sentinel from P15.3.5's judge) are skipped.
        """
        count = 0
        for v in verdicts:
            wm = getattr(v, "winner_model", None)
            lm = getattr(v, "loser_model", None)
            if wm and lm:
                self.add(PreferencePair(
                    prompt=v.prompt,
                    winner_model=wm,
                    loser_model=lm,
                    source="judge",
                    confidence=v.confidence / 5.0,
                    metadata={"reasoning": v.reasoning, "judge": v.judge_model},
                ))
                count += 1
        logger.info("added %d preference pairs from judge verdicts", count)
        return count

    def add_from_pointwise_scores(self, scores: list, threshold: float = 0.5) -> int:
        """Convert pointwise scores into preference pairs.

        Groups by prompt and creates pairs where the higher-scored
        model wins. ``threshold`` is the minimum absolute score diff
        required to create a pair.
        """
        by_prompt: dict[str, list] = {}
        for s in scores:
            by_prompt.setdefault(s.prompt, []).append(s)

        count = 0
        for prompt, prompt_scores in by_prompt.items():
            for i, sa in enumerate(prompt_scores):
                for sb in prompt_scores[i + 1:]:
                    diff = abs(sa.score - sb.score)
                    if diff >= threshold:
                        winner = sa if sa.score > sb.score else sb
                        loser = sb if sa.score > sb.score else sa
                        self.add(PreferencePair(
                            prompt=prompt,
                            winner_model=winner.model_id,
                            loser_model=loser.model_id,
                            source="judge",
                            confidence=min(diff / 4.0, 1.0),
                        ))
                        count += 1
        logger.info("added %d preference pairs from pointwise scores", count)
        return count

    def model_win_rates(self) -> dict[str, float]:
        """Compute win rate for each model across all pairs."""
        wins: dict[str, int] = {}
        total: dict[str, int] = {}
        for p in self.pairs:
            wins[p.winner_model] = wins.get(p.winner_model, 0) + 1
            total[p.winner_model] = total.get(p.winner_model, 0) + 1
            total[p.loser_model] = total.get(p.loser_model, 0) + 1
        return {m: wins.get(m, 0) / total[m] for m in total if total[m] > 0}

    def filter_by_source(self, source: str) -> "PreferenceDataset":
        """Return a new dataset with only pairs from the given source."""
        return PreferenceDataset(
            [p for p in self.pairs if p.source == source],
            name=f"{self.name}_{source}",
        )

    def filter_by_confidence(self, min_confidence: float) -> "PreferenceDataset":
        """Return pairs with confidence >= threshold."""
        return PreferenceDataset(
            [p for p in self.pairs if p.confidence >= min_confidence],
            name=f"{self.name}_conf{min_confidence}",
        )

    def save(self, path: str | Path) -> None:
        """Save to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for p in self.pairs:
                f.write(json.dumps({
                    "prompt": p.prompt,
                    "winner_model": p.winner_model,
                    "loser_model": p.loser_model,
                    "source": p.source,
                    "confidence": p.confidence,
                    "metadata": p.metadata,
                }) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceDataset":
        """Load from JSONL."""
        pairs = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                pairs.append(PreferencePair(**d))
        return cls(pairs)
