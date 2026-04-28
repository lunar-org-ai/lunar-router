"""
Training Advisor — intelligent agent that monitors production and decides
when to trigger auto-training.

Instead of a dumb timer, this agent:
1. Runs heuristic checks on recent traces (fast, no LLM)
2. Calls the training_advisor LLM agent for nuanced analysis
3. Stores its decision in memory for audit trail
4. Triggers auto-training only when signals warrant it

Integrates with the existing harness (trace_scanner, memory_store, scheduler).
"""

from __future__ import annotations

import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .ledger import LedgerEntry, LedgerStore
from .memory_store import MemoryEntry, MemoryStore, get_memory_store
from .runner import AgentRunner

logger = logging.getLogger(__name__)

DECISION_CATEGORY = "training_decision"

# Heuristic thresholds
ERROR_RATE_INCREASE_THRESHOLD = 0.05  # 5% absolute increase
DRIFT_RATIO_THRESHOLD = 1.5
MIN_TRACES_FOR_TRAINING = 100
MIN_HOURS_BETWEEN_TRAINING = 48
HIGH_SEVERITY_ISSUE_THRESHOLD = 3  # this many high-severity issues = signal

# Maps heuristic signal names to the objective time-series they feed.
# `None` means the signal is a meta-indicator (drift) that's ledger-
# worthy but not part of any objective series.
SIGNAL_TO_OBJECTIVE: dict[str, Optional[str]] = {
    "error_rate_increase": "domain_coverage_ratio",
    "high_severity_issues": "domain_coverage_ratio",
    "drift_ratio": None,
    "trace_volume": None,
    "cooldown": None,
}

# Signals where `triggered=True` means "problem detected" and warrants
# a ledger observation. `cooldown` / `trace_volume` invert this polarity
# (triggered=True means "gate passed"), so they're excluded.
_OBSERVABLE_SIGNALS = frozenset({
    "error_rate_increase",
    "high_severity_issues",
    "drift_ratio",
})


@dataclass
class TrainingSignal:
    """A single signal contributing to the training decision."""

    name: str
    value: float
    threshold: float
    triggered: bool
    detail: str

    def __str__(self):
        status = "TRIGGERED" if self.triggered else "ok"
        return f"{self.name}: {self.value:.3f} (threshold={self.threshold:.3f}) [{status}] — {self.detail}"


@dataclass
class TrainingRecommendation:
    """Result of the training advisor analysis."""

    recommendation: str  # "train_now" | "wait" | "investigate"
    confidence: float  # 0.0 - 1.0
    reason: str
    signals: list[TrainingSignal]
    suggested_config: dict[str, Any] = field(default_factory=dict)
    source: str = "heuristic"  # "heuristic" or "agent"

    @property
    def should_train(self) -> bool:
        return self.recommendation == "train_now" and self.confidence >= 0.7

    def summary(self) -> str:
        lines = [
            f"Training Recommendation: {self.recommendation.upper()} (confidence={self.confidence:.0%})",
            f"  Source: {self.source}",
            f"  Reason: {self.reason}",
            "  Signals:",
        ]
        for s in self.signals:
            lines.append(f"    {'>>>' if s.triggered else '   '} {s}")
        return "\n".join(lines)


class TrainingAdvisor:
    """
    Monitors production health and decides when to trigger auto-training.

    Two-layer analysis (same pattern as TraceScanner):
    1. Heuristic layer: fast checks on error rates, drift, issue counts
    2. Agent layer: LLM-based nuanced analysis when heuristics are ambiguous

    Usage:
        advisor = TrainingAdvisor(engine_url="http://localhost:8080")

        # Quick heuristic check (fast, no LLM)
        rec = advisor.check_heuristics(traces, issues, drift_ratio)
        if rec.should_train:
            auto_trainer.train(...)

        # Full analysis with LLM agent (slower, smarter)
        rec = await advisor.analyze(days_lookback=7)
        if rec.should_train:
            auto_trainer.train(...)
    """

    def __init__(
        self,
        engine_url: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
        ledger: Optional[LedgerStore] = None,
    ):
        self.engine_url = engine_url
        self.memory_store = memory_store or get_memory_store()
        # `ledger=None` = no-op emission (keeps existing callsites unchanged).
        self.ledger = ledger
        self._runner = AgentRunner(
            engine_url=engine_url,
            memory_store=self.memory_store,
            record_memory=True,
        )

    # ── Layer 1: Heuristic checks (fast, no LLM) ────────────────────────

    def check_heuristics(
        self,
        error_rates: dict[str, float],
        baseline_error_rates: dict[str, float],
        high_severity_issues: int = 0,
        drift_ratio: float = 1.0,
        trace_count: int = 0,
        hours_since_last_training: Optional[float] = None,
        parent_id: Optional[str] = None,
    ) -> TrainingRecommendation:
        """
        Fast heuristic check — no LLM call needed.

        Args:
            error_rates: Current per-model error rates.
            baseline_error_rates: Per-model error rates from profiling.
            high_severity_issues: Count of high-severity trace issues.
            drift_ratio: Current drift ratio from DriftDetector.
            trace_count: Number of traces available for training.
            hours_since_last_training: Hours since last successful training.

        Returns:
            TrainingRecommendation with heuristic-based decision.
        """
        started_at = datetime.now(timezone.utc)
        signals: list[TrainingSignal] = []
        triggered_count = 0

        # Signal 1: Error rate increase
        max_increase = 0.0
        worst_model = ""
        for model_id in error_rates:
            baseline = baseline_error_rates.get(model_id, 0.0)
            current = error_rates[model_id]
            increase = current - baseline
            if increase > max_increase:
                max_increase = increase
                worst_model = model_id

        error_triggered = max_increase > ERROR_RATE_INCREASE_THRESHOLD
        signals.append(TrainingSignal(
            name="error_rate_increase",
            value=max_increase,
            threshold=ERROR_RATE_INCREASE_THRESHOLD,
            triggered=error_triggered,
            detail=f"worst: {worst_model} (+{max_increase:.1%})" if worst_model else "stable",
        ))
        if error_triggered:
            triggered_count += 1

        # Signal 2: High-severity issues
        issues_triggered = high_severity_issues >= HIGH_SEVERITY_ISSUE_THRESHOLD
        signals.append(TrainingSignal(
            name="high_severity_issues",
            value=float(high_severity_issues),
            threshold=float(HIGH_SEVERITY_ISSUE_THRESHOLD),
            triggered=issues_triggered,
            detail=f"{high_severity_issues} high-severity issues detected",
        ))
        if issues_triggered:
            triggered_count += 1

        # Signal 3: Distribution drift
        drift_triggered = drift_ratio > DRIFT_RATIO_THRESHOLD
        signals.append(TrainingSignal(
            name="drift_ratio",
            value=drift_ratio,
            threshold=DRIFT_RATIO_THRESHOLD,
            triggered=drift_triggered,
            detail=f"traffic {drift_ratio:.1f}x from trained centroids",
        ))
        if drift_triggered:
            triggered_count += 1

        # Signal 4: Sufficient data
        data_sufficient = trace_count >= MIN_TRACES_FOR_TRAINING
        signals.append(TrainingSignal(
            name="trace_volume",
            value=float(trace_count),
            threshold=float(MIN_TRACES_FOR_TRAINING),
            triggered=data_sufficient,
            detail=f"{trace_count} traces available",
        ))

        # Signal 5: Cooldown period
        cooldown_ok = True
        if hours_since_last_training is not None:
            cooldown_ok = hours_since_last_training >= MIN_HOURS_BETWEEN_TRAINING
        signals.append(TrainingSignal(
            name="cooldown",
            value=hours_since_last_training or 999.0,
            threshold=MIN_HOURS_BETWEEN_TRAINING,
            triggered=cooldown_ok,
            detail=f"{hours_since_last_training:.0f}h since last training" if hours_since_last_training else "no previous training",
        ))

        # Decision logic
        if not data_sufficient:
            rec = "wait"
            confidence = 0.9
            reason = f"Insufficient traces ({trace_count} < {MIN_TRACES_FOR_TRAINING})"
        elif not cooldown_ok:
            rec = "wait"
            confidence = 0.8
            reason = f"Cooldown active ({hours_since_last_training:.0f}h < {MIN_HOURS_BETWEEN_TRAINING}h)"
        elif triggered_count >= 2:
            rec = "train_now"
            confidence = min(0.6 + triggered_count * 0.1, 0.95)
            reasons = [s.detail for s in signals if s.triggered]
            reason = f"{triggered_count} signals triggered: " + "; ".join(reasons)
        elif triggered_count == 1:
            rec = "investigate"
            confidence = 0.6
            triggered_signal = next(s for s in signals if s.triggered)
            reason = f"Single signal: {triggered_signal.detail}"
        else:
            rec = "wait"
            confidence = 0.9
            reason = "All metrics within normal range"

        result = TrainingRecommendation(
            recommendation=rec,
            confidence=confidence,
            reason=reason,
            signals=signals,
            source="heuristic",
        )
        self._ledger_emit_decision(
            result,
            started_at=started_at,
            parameters={
                "error_rates": error_rates,
                "baseline_error_rates": baseline_error_rates,
                "high_severity_issues": high_severity_issues,
                "drift_ratio": drift_ratio,
                "trace_count": trace_count,
                "hours_since_last_training": hours_since_last_training,
            },
            parent_id=parent_id,
        )
        return result

    # ── Layer 2: LLM agent analysis (smart, slower) ─────────────────────

    async def analyze(
        self,
        error_rates: dict[str, float],
        baseline_error_rates: dict[str, float],
        high_severity_issues: int = 0,
        drift_ratio: float = 1.0,
        trace_count: int = 0,
        hours_since_last_training: Optional[float] = None,
        recent_issues_summary: str = "",
        training_history_summary: str = "",
    ) -> TrainingRecommendation:
        """
        Full LLM-powered analysis. Uses the training_advisor agent.

        Call this when heuristics return "investigate" or when you want
        a more nuanced decision.
        """
        # First run heuristics for context
        heuristic = self.check_heuristics(
            error_rates, baseline_error_rates,
            high_severity_issues, drift_ratio,
            trace_count, hours_since_last_training,
        )

        # Build context for the LLM agent
        error_lines = []
        for mid in sorted(error_rates):
            baseline = baseline_error_rates.get(mid, 0.0)
            current = error_rates[mid]
            delta = current - baseline
            error_lines.append(f"  {mid}: {baseline:.1%} → {current:.1%} ({delta:+.1%})")

        signal_lines = [f"  {s}" for s in heuristic.signals]

        user_input = f"""## Current Production State

**Error Rates (baseline → current):**
{chr(10).join(error_lines)}

**Heuristic Signals:**
{chr(10).join(signal_lines)}

**Heuristic Recommendation:** {heuristic.recommendation} (confidence={heuristic.confidence:.0%})
**Heuristic Reason:** {heuristic.reason}

**Drift Ratio:** {drift_ratio:.2f}x
**Available Traces:** {trace_count}
**Hours Since Last Training:** {hours_since_last_training or 'never'}

## Recent Quality Issues
{recent_issues_summary or 'None reported'}

## Training History
{training_history_summary or 'No previous training runs'}

Given all this data, should we trigger auto-training now?"""

        try:
            result = await self._runner.run("training_advisor", user_input)

            rec = TrainingRecommendation(
                recommendation=result.data.get("recommendation", "wait"),
                confidence=float(result.data.get("confidence", 0.5)),
                reason=result.data.get("reason", ""),
                signals=heuristic.signals,
                suggested_config=result.data.get("suggested_config", {}),
                source="agent",
            )

        except Exception as e:
            logger.warning(f"Agent analysis failed, falling back to heuristic: {e}")
            rec = heuristic

        # Store decision in memory
        self._store_decision(rec)
        return rec

    # ── Ledger ───────────────────────────────────────────────────────────

    def _ledger_emit_decision(
        self,
        rec: TrainingRecommendation,
        started_at: datetime,
        parameters: dict[str, Any],
        parent_id: Optional[str] = None,
    ) -> Optional[str]:
        """Write run + triggered-signal observations + decision to the ledger.

        Returns the run entry id, or None if no ledger is configured.
        Failure to append is logged at debug and does not propagate — the
        advisor's decision path must not be blocked by an observability bug.

        `parent_id` chains the run to an upstream ledger entry (typically
        a `signal` from the trigger engine). When None, the run is a root.
        """
        if self.ledger is None:
            return None

        run_id = str(uuid.uuid4())
        run_entry = LedgerEntry(
            id=run_id,
            type="run",
            agent="training_advisor",
            parameters_in=parameters,
            parent_id=parent_id,
            tags=["scheduler_tick", "training_advisor"],
        )
        try:
            self.ledger.append(run_entry)
        except Exception as e:
            logger.debug(f"Failed to append training-advisor run to ledger: {e}")
            return None

        for signal in rec.signals:
            if signal.name not in _OBSERVABLE_SIGNALS:
                continue
            if not signal.triggered:
                continue
            obs = LedgerEntry(
                type="observation",
                objective_id=SIGNAL_TO_OBJECTIVE.get(signal.name),
                agent="training_advisor",
                parent_id=run_id,
                data={
                    "signal_name": signal.name,
                    "value": signal.value,
                    "threshold": signal.threshold,
                    "detail": signal.detail,
                },
                tags=[signal.name, "triggered"],
            )
            try:
                self.ledger.append(obs)
            except Exception as e:
                logger.debug(f"Failed to append signal observation to ledger: {e}")

        # Map recommendation → outcome. `train_now` is the only positive
        # action-producing decision; everything else is either a deferred
        # investigation or an explicit skip (cooldown / insufficient data).
        outcome = "ok" if rec.recommendation == "train_now" else "skipped"
        duration_ms = int(
            (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        )
        decision = LedgerEntry(
            type="decision",
            agent="training_advisor",
            parent_id=run_id,
            data={
                "recommendation": rec.recommendation,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "source": rec.source,
            },
            tags=[rec.recommendation, rec.source],
            duration_ms=duration_ms,
            outcome=outcome,
        )
        try:
            self.ledger.append(decision)
        except Exception as e:
            logger.debug(f"Failed to append decision to ledger: {e}")

        return run_id

    # ── Memory ───────────────────────────────────────────────────────────

    def _store_decision(self, rec: TrainingRecommendation) -> None:
        """Persist the decision for audit trail."""
        entry = MemoryEntry(
            id=f"training_rec_{uuid.uuid4().hex[:8]}",
            agent="training_advisor",
            category=DECISION_CATEGORY,
            created_at=datetime.now(timezone.utc).isoformat(),
            body=(
                f"## Training Recommendation: {rec.recommendation.upper()}\n\n"
                f"**Confidence:** {rec.confidence:.0%}\n"
                f"**Source:** {rec.source}\n"
                f"**Reason:** {rec.reason}\n\n"
                f"### Signals\n" +
                "\n".join(f"- {s}" for s in rec.signals)
            ),
            model="",
            duration_ms=0,
            tokens_in=0,
            tokens_out=0,
            tags=[rec.recommendation, rec.source, f"conf_{rec.confidence:.1f}"],
            evaluation={
                "recommendation": rec.recommendation,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "signals": [
                    {"name": s.name, "value": s.value, "triggered": s.triggered}
                    for s in rec.signals
                ],
                "suggested_config": rec.suggested_config,
            },
        )
        self.memory_store.save(entry)
        logger.info(f"Stored training decision: {rec.recommendation} (confidence={rec.confidence:.0%})")

    def get_last_training_time(self) -> Optional[datetime]:
        """Get timestamp of last successful training from memory."""
        entries = self.memory_store.query(
            agent="training_advisor",
            category=DECISION_CATEGORY,
            tags=["train_now"],
        )
        if entries:
            latest = max(entries, key=lambda e: e.get("created_at", ""))
            try:
                return datetime.fromisoformat(latest["created_at"])
            except (KeyError, ValueError):
                pass
        return None

    def get_decision_history(self, limit: int = 10) -> list[dict]:
        """Get recent training decisions."""
        return self.memory_store.query(
            agent="training_advisor",
            category=DECISION_CATEGORY,
            limit=limit,
        )
