"""Research-standard routing evaluation metrics.

Implements metrics from RouteLLM (arXiv:2406.18665), R2-Router, and
RouterXBench for rigorous comparison of routing strategies.

All metrics operate on arrays of per-sample results, not raw responses.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np  # noqa: F401  (kept; downstream callers may import via this module)


def compute_auroc(
    scores: list[float],
    labels: list[bool],
) -> float:
    """Area Under the ROC Curve for "should we use the strong model?"

    Score = router's predicted benefit of using the strong model. Label
    = True when the strong model was actually needed (weak got it wrong,
    strong got it right). Perfect router → 1.0; random → 0.5.
    """
    if not scores or not labels:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_tp = 0
    prev_fp = 0

    for i, (_, label) in enumerate(pairs):
        if label:
            tp += 1
        else:
            fp += 1

        if i + 1 == len(pairs) or pairs[i][0] != pairs[i + 1][0]:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp = tp
            prev_fp = fp

    return auc / (n_pos * n_neg)


def compute_apgr(
    quality_router: float,
    quality_weak: float,
    quality_strong: float,
) -> float:
    """Average Performance Gap Recovered.

    APGR = (Q_router - Q_weak) / (Q_strong - Q_weak)

    1.0 = router matches strong; 0.0 = router matches weak; >1.0 means
    selective routing beat the strong model on average.
    """
    gap = quality_strong - quality_weak
    if gap <= 0:
        return 1.0 if quality_router >= quality_strong else 0.0
    return (quality_router - quality_weak) / gap


def compute_cpt(
    pareto_points: list[tuple[float, float]],
    quality_target_pct: float = 0.95,
    quality_strong: float = 1.0,
    quality_weak: float = 0.0,
) -> Optional[float]:
    """Cost at Performance Target — min strong-model fraction to hit target.

    From RouteLLM: CPT(95%) = 14% means you only need 14% strong-model
    calls to reach 95% of the way from weak to strong on quality.
    """
    target_quality = quality_weak + quality_target_pct * (quality_strong - quality_weak)
    best_cost = None
    for quality, cost in pareto_points:
        if quality >= target_quality:
            if best_cost is None or cost < best_cost:
                best_cost = cost
    return best_cost


def compute_pgr_at_savings(
    pareto_points: list[tuple[float, float]],
    savings_target: float,
    quality_strong: float,
    quality_weak: float,
) -> Optional[float]:
    """Performance Gap Recovered at a given cost-savings level."""
    max_cost = 1.0 - savings_target
    best_quality = None
    for quality, cost in pareto_points:
        if cost <= max_cost:
            if best_quality is None or quality > best_quality:
                best_quality = quality
    if best_quality is None:
        return None
    return compute_apgr(best_quality, quality_weak, quality_strong)


def compute_win_rate(
    router_correct: list[bool],
    total: Optional[int] = None,
) -> float:
    """Fraction of prompts where the routed model answered correctly."""
    n = total if total is not None else len(router_correct)
    if n == 0:
        return 0.0
    return sum(router_correct) / n


@dataclass
class RoutingMetrics:
    """Aggregated routing evaluation metrics for a single evaluation run."""

    auroc: float
    apgr: float
    win_rate: float
    cpt_50: Optional[float]
    cpt_75: Optional[float]
    cpt_90: Optional[float]
    cpt_95: Optional[float]
    pgr_at_25_savings: Optional[float]
    pgr_at_50_savings: Optional[float]
    pgr_at_75_savings: Optional[float]
    quality_strong: float
    quality_weak: float
    strong_model: str
    weak_model: str
    num_samples: int

    def to_dict(self) -> dict:
        return {
            "auroc": self.auroc,
            "apgr": self.apgr,
            "win_rate": self.win_rate,
            "cpt_50": self.cpt_50,
            "cpt_75": self.cpt_75,
            "cpt_90": self.cpt_90,
            "cpt_95": self.cpt_95,
            "pgr_at_25_savings": self.pgr_at_25_savings,
            "pgr_at_50_savings": self.pgr_at_50_savings,
            "pgr_at_75_savings": self.pgr_at_75_savings,
            "quality_strong": self.quality_strong,
            "quality_weak": self.quality_weak,
            "strong_model": self.strong_model,
            "weak_model": self.weak_model,
            "num_samples": self.num_samples,
        }

    def summary(self) -> str:
        lines = [
            f"Routing Evaluation ({self.num_samples} samples)",
            f"  Strong model: {self.strong_model} (accuracy: {self.quality_strong:.2%})",
            f"  Weak model:   {self.weak_model} (accuracy: {self.quality_weak:.2%})",
            f"  AUROC:        {self.auroc:.4f}",
            f"  APGR:         {self.apgr:.2%}",
            f"  Win rate:     {self.win_rate:.2%}",
        ]
        if self.cpt_95 is not None:
            lines.append(f"  CPT(95%):     {self.cpt_95:.2%} strong calls needed")
        if self.cpt_90 is not None:
            lines.append(f"  CPT(90%):     {self.cpt_90:.2%} strong calls needed")
        if self.pgr_at_50_savings is not None:
            lines.append(f"  PGR@50% savings: {self.pgr_at_50_savings:.2%}")
        return "\n".join(lines)
