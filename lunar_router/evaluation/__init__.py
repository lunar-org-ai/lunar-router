"""
Evaluation framework for routing quality measurement.

Provides offline evaluation of routing decisions with research-standard
metrics (AUROC, CPT, APGR, PGR), baseline comparisons, and Pareto curve
generation for cost-quality tradeoff analysis.
"""

from .response_cache import ResponseCache, CachedResponse
from .metrics import (
    compute_auroc,
    compute_cpt,
    compute_apgr,
    compute_pgr_at_savings,
    compute_win_rate,
    RoutingMetrics,
)
from .baselines import (
    RandomBaseline,
    OracleBaseline,
    AlwaysStrongBaseline,
    AlwaysWeakBaseline,
)
from .evaluator import RouterEvaluator, EvaluationResult, ParetoPoint

__all__ = [
    # Response cache
    "ResponseCache",
    "CachedResponse",
    # Metrics
    "compute_auroc",
    "compute_cpt",
    "compute_apgr",
    "compute_pgr_at_savings",
    "compute_win_rate",
    "RoutingMetrics",
    # Baselines
    "RandomBaseline",
    "OracleBaseline",
    "AlwaysStrongBaseline",
    "AlwaysWeakBaseline",
    # Evaluator
    "RouterEvaluator",
    "EvaluationResult",
    "ParetoPoint",
]
