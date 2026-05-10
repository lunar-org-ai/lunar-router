"""RouterHealth snapshot — what the harness brain reads via MCP.

Surface for ``router_health_check()`` (P15.3.9). Cheap; the drift sample
is capped at 200 traces. All fields are best-effort: missing files /
unreadable artifacts yield ``None`` rather than exceptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


logger = logging.getLogger("router.feedback.health")


@dataclass
class RouterHealth:
    """Read-only snapshot of the router's current state."""

    cold_start: bool
    version: Optional[int]
    k: Optional[int]
    model_count: Optional[int]
    cost_weight: Optional[float]
    last_fit_at: Optional[str]
    last_fit_age_hours: Optional[float]
    trace_count_since_last_fit: int
    drift_score: Optional[float]
    drift_baseline: Optional[float]
    needs_reclustering: Optional[bool]
    current_avg_error: Optional[float]
    current_win_rate: Optional[float]
    cluster_distribution: Optional[dict[int, int]]
    fitted_from: Optional[dict] = None
    sample_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cold_start": self.cold_start,
            "version": self.version,
            "k": self.k,
            "model_count": self.model_count,
            "cost_weight": self.cost_weight,
            "last_fit_at": self.last_fit_at,
            "last_fit_age_hours": self.last_fit_age_hours,
            "trace_count_since_last_fit": self.trace_count_since_last_fit,
            "drift_score": self.drift_score,
            "drift_baseline": self.drift_baseline,
            "needs_reclustering": self.needs_reclustering,
            "current_avg_error": self.current_avg_error,
            "current_win_rate": self.current_win_rate,
            "cluster_distribution": self.cluster_distribution,
            "fitted_from": self.fitted_from,
            "sample_size": self.sample_size,
            "metadata": self.metadata,
        }


DEFAULT_DRIFT_SAMPLE_SIZE = 200


def compute_router_health(
    *,
    drift_sample_size: int = DEFAULT_DRIFT_SAMPLE_SIZE,
    embedder: Optional[Any] = None,
) -> RouterHealth:
    """Build a RouterHealth snapshot.

    Cheap operation. Reads:
      - ``versions/router_config_current`` payload (cold-start safe)
      - ``traces/raw/<date>.jsonl`` for trace count + drift sample
      - latest ``evals/reports/`` for current_avg_error / current_win_rate
        (best-effort; ``None`` when no router-suite report exists)
    """
    from router.config_io import (
        get_current_version,
        load_current_config_payload,
    )
    from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError

    # 1. Resolve current config (cold-start safe).
    try:
        payload = load_current_config_payload()
    except (RouterConfigNotFoundError, RouterConfigInvalidError):
        return _cold_start_health()

    version = int(payload.get("version", 0))
    last_fit_at = payload.get("created_at")
    last_fit_age_h = _hours_since(last_fit_at) if last_fit_at else None
    cost_weight = float(payload.get("cost_weight", 0.0))
    k = int(payload.get("k", 0))
    model_count = len(payload.get("model_psi") or {})
    fitted_from = payload.get("fitted_from")
    drift_baseline = payload.get("drift_baseline")
    if drift_baseline is not None:
        drift_baseline = float(drift_baseline)

    # 2. Count traces since the fit (cheap path: enumerate JSONL line counts).
    trace_count = _count_traces_since(last_fit_at)

    # 3. Drift report — only when an embedder is supplied (otherwise the
    # caller doesn't want to pay the embedding cost).
    drift_score: Optional[float] = None
    needs_reclustering: Optional[bool] = None
    cluster_distribution: Optional[dict[int, int]] = None
    sample_size = 0
    if embedder is not None and last_fit_at is not None and trace_count > 0:
        report = _compute_drift_report(
            since_iso=last_fit_at,
            embedder=embedder,
            payload=payload,
            sample_size=drift_sample_size,
            baseline=drift_baseline,
        )
        if report is not None:
            drift_score = report["drift_score"]
            needs_reclustering = report["needs_reclustering"]
            cluster_distribution = report["cluster_usage"]
            sample_size = report["num_embeddings"]

    # 4. Latest router-suite eval report (best-effort).
    avg_error, win_rate = _latest_router_eval_metrics(version=version)

    return RouterHealth(
        cold_start=False,
        version=version,
        k=k,
        model_count=model_count,
        cost_weight=cost_weight,
        last_fit_at=last_fit_at,
        last_fit_age_hours=last_fit_age_h,
        trace_count_since_last_fit=trace_count,
        drift_score=drift_score,
        drift_baseline=drift_baseline,
        needs_reclustering=needs_reclustering,
        current_avg_error=avg_error,
        current_win_rate=win_rate,
        cluster_distribution=cluster_distribution,
        fitted_from=fitted_from,
        sample_size=sample_size,
        metadata={"current_version": str(get_current_version() or 0)},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cold_start_health() -> RouterHealth:
    """Return health when no router_config exists yet."""
    return RouterHealth(
        cold_start=True,
        version=None,
        k=None,
        model_count=None,
        cost_weight=None,
        last_fit_at=None,
        last_fit_age_hours=None,
        trace_count_since_last_fit=_count_traces_since(None),
        drift_score=None,
        drift_baseline=None,
        needs_reclustering=None,
        current_avg_error=None,
        current_win_rate=None,
        cluster_distribution=None,
        fitted_from=None,
        sample_size=0,
        metadata={"current_version": "0"},
    )


def _hours_since(iso: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    delta = datetime.now(timezone.utc) - dt
    return delta.total_seconds() / 3600.0


def _count_traces_since(since_iso: Optional[str]) -> int:
    """Cheap line count over JSONL partitions newer than since_iso."""
    from router.feedback.store_adapter import _select_partition_files, _TRACES_RAW

    files = _select_partition_files(_TRACES_RAW, since_iso, None)
    count = 0
    for path in files:
        try:
            with path.open() as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            continue
    return count


def _compute_drift_report(
    *,
    since_iso: str,
    embedder: Any,
    payload: dict,
    sample_size: int,
    baseline: Optional[float],
) -> Optional[dict]:
    """Embed a sample of recent traces + run DriftDetector. Returns dict
    with the fields RouterHealth needs, or None on failure."""
    try:
        from router.config_io import _build_assigner, _vd
        from router.feedback.drift_detector import DriftDetector
        from router.feedback.store_adapter import iter_traces_since

        assigner = _build_assigner(payload, _vd(None))

        # Pull up to sample_size prompts after since_iso.
        prompts: list[str] = []
        for trace in iter_traces_since(since_iso=since_iso):
            if trace.input_text:
                prompts.append(trace.input_text)
            if len(prompts) >= sample_size:
                break
        if not prompts:
            return None

        embeddings = embedder.embed_batch(prompts)
        embeddings = np.asarray(embeddings)
        detector = DriftDetector(
            cluster_assigner=assigner,
            baseline_distance=baseline,
        )
        report = detector.check(embeddings)
        return {
            "drift_score": float(report.drift_ratio),
            "needs_reclustering": bool(report.needs_reclustering),
            "cluster_usage": dict(report.cluster_usage),
            "num_embeddings": int(report.num_embeddings),
        }
    except Exception as e:
        logger.warning("drift report failed (non-fatal): %s", e)
        return None


def _latest_router_eval_metrics(*, version: int) -> tuple[Optional[float], Optional[float]]:
    """Read the most recent router-kind eval report, if any.

    Returns ``(avg_error, win_rate)`` or ``(None, None)``.
    """
    reports_dir = Path("evals") / "reports"
    if not reports_dir.exists():
        return None, None

    latest_path: Optional[Path] = None
    latest_mtime = -1.0
    for path in reports_dir.glob("*.json"):
        try:
            mt = path.stat().st_mtime
        except OSError:
            continue
        if mt > latest_mtime:
            latest_mtime = mt
            latest_path = path

    if latest_path is None:
        return None, None

    try:
        import json

        body = json.loads(latest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None

    metrics = body.get("metrics") or {}
    win_rate = metrics.get("win_rate")
    quality = metrics.get("quality_router") or metrics.get("apgr")
    avg_error = (1.0 - quality) if isinstance(quality, (int, float)) else None
    return avg_error, win_rate
