"""
Scheduled auto-training runner.

Runs the full feedback loop on a cadence:
1. Collect traces from ClickHouse
2. Run AutoTrainer (evaluate → augment → retrain → promote)
3. If promoted, reload Go engine weights
4. Log results

Uses asyncio for non-blocking periodic execution.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingScheduleConfig:
    """Configuration for scheduled auto-training."""

    enabled: bool = False
    check_interval_seconds: int = 3600  # check every hour (advisor decides if training needed)
    days_lookback: int = 7
    trace_limit: int = 10000
    production_alpha: float = 0.3
    weights_path: str = "./weights"
    engine_url: str = "http://localhost:8080"
    auto_reload_engine: bool = True
    use_advisor: bool = True  # use TrainingAdvisor to decide when to train


@dataclass
class TrainingRunLog:
    """Record of a scheduled training run."""

    run_id: str
    started_at: str
    completed_at: str
    traces_collected: int
    promoted: bool
    reason: str
    auroc_before: Optional[float] = None
    auroc_after: Optional[float] = None
    engine_reloaded: bool = False
    error: Optional[str] = None


class ScheduledTrainer:
    """
    Runs auto-training on a schedule.

    Usage:
        trainer = ScheduledTrainer(config, auto_trainer, embedder)

        # Run once manually
        log = await trainer.run_once()

        # Start periodic loop
        await trainer.start()  # runs forever on cadence

        # Stop
        trainer.stop()
    """

    def __init__(
        self,
        config: TrainingScheduleConfig,
        auto_trainer=None,  # AutoTrainer (lazy — built from weights if None)
        embedder=None,  # PromptEmbedder for drift detection
    ):
        self.config = config
        self.auto_trainer = auto_trainer
        self.embedder = embedder
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.history: list[TrainingRunLog] = []

        # Initialize training advisor (intelligent trigger)
        self._advisor = None
        if config.use_advisor:
            try:
                from ..harness.training_advisor import TrainingAdvisor
                self._advisor = TrainingAdvisor(engine_url=config.engine_url)
            except Exception as e:
                logger.warning(f"Could not initialize TrainingAdvisor: {e}")

    async def check_and_train(self) -> TrainingRunLog:
        """
        Ask the advisor if training is needed, then train if recommended.

        This is the smart alternative to run_once() — the advisor monitors
        error rates, drift, issue counts and only triggers training when
        multiple signals indicate degradation.
        """
        import uuid

        run_id = str(uuid.uuid4())[:8]
        started = datetime.now(timezone.utc).isoformat()

        try:
            from .collector import collect_traces, collect_quality_flags, collect_trace_embeddings

            traces = collect_traces(
                days=self.config.days_lookback,
                limit=self.config.trace_limit,
            )

            if not traces:
                log = TrainingRunLog(
                    run_id=run_id, started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    traces_collected=0, promoted=False,
                    reason="No traces — advisor skipped",
                )
                self.history.append(log)
                return log

            # Compute signals for the advisor
            from .trace_to_training import TraceToTraining
            num_clusters = 100
            if self.auto_trainer and self.auto_trainer.profiles:
                num_clusters = self.auto_trainer.profiles[0].num_clusters

            converter = TraceToTraining(num_clusters=num_clusters)
            converter.add_traces(traces)
            updates = converter.compute_psi_updates()

            # Current vs baseline error rates
            error_rates = {u.model_id: u.error_rate for u in updates}
            baseline_rates = {}
            if self.auto_trainer:
                baseline_rates = {
                    p.model_id: p.overall_error_rate
                    for p in self.auto_trainer.profiles
                }

            # Drift detection
            drift_ratio = 1.0
            if self.embedder:
                embeddings = collect_trace_embeddings(traces, self.embedder)
                if embeddings is not None and self.auto_trainer:
                    from .drift_detector import DriftDetector
                    detector = DriftDetector(self.auto_trainer.assigner)
                    report = detector.check(embeddings)
                    drift_ratio = report.drift_ratio

            # Issue count from memory
            quality_flags = collect_quality_flags(days=self.config.days_lookback)
            high_severity = len(quality_flags)

            # Hours since last training
            hours_since = None
            if self._advisor:
                last_time = self._advisor.get_last_training_time()
                if last_time:
                    delta = datetime.now(timezone.utc) - last_time
                    hours_since = delta.total_seconds() / 3600

            # Ask the advisor
            rec = self._advisor.check_heuristics(
                error_rates=error_rates,
                baseline_error_rates=baseline_rates,
                high_severity_issues=high_severity,
                drift_ratio=drift_ratio,
                trace_count=len(traces),
                hours_since_last_training=hours_since,
            )

            logger.info(f"[{run_id}] Advisor: {rec.recommendation} (confidence={rec.confidence:.0%})")

            if not rec.should_train:
                log = TrainingRunLog(
                    run_id=run_id, started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    traces_collected=len(traces), promoted=False,
                    reason=f"Advisor: {rec.recommendation} — {rec.reason}",
                )
                self.history.append(log)
                return log

            # Advisor says train — proceed
            logger.info(f"[{run_id}] Advisor triggered training: {rec.reason}")
            result = self.auto_trainer.train(
                production_traces=traces,
                quality_flags=quality_flags,
            )

            # Reload engine if promoted
            engine_reloaded = False
            if result.promoted and self.config.auto_reload_engine:
                from .engine_reload import reload_engine_weights
                reload = reload_engine_weights(
                    weights_path=self.config.weights_path,
                    engine_url=self.config.engine_url,
                )
                engine_reloaded = reload.success

            log = TrainingRunLog(
                run_id=run_id, started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                traces_collected=len(traces),
                promoted=result.promoted,
                reason=f"Advisor triggered | {result.reason}",
                auroc_before=result.baseline_metrics.auroc if result.baseline_metrics else None,
                auroc_after=result.new_metrics.auroc if result.new_metrics else None,
                engine_reloaded=engine_reloaded,
            )
            self.history.append(log)
            return log

        except Exception as e:
            log = TrainingRunLog(
                run_id=run_id, started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                traces_collected=0, promoted=False,
                reason="Error", error=str(e),
            )
            self.history.append(log)
            logger.error(f"[{run_id}] Check-and-train failed: {e}")
            return log

    async def run_once(self) -> TrainingRunLog:
        """
        Execute one training cycle:
        1. Collect traces from ClickHouse
        2. Collect quality flags
        3. Run auto-trainer
        4. Reload engine if promoted
        """
        import uuid

        run_id = str(uuid.uuid4())[:8]
        started = datetime.now(timezone.utc).isoformat()

        logger.info(f"[{run_id}] Starting scheduled training run...")

        try:
            # Step 1: Collect traces
            from .collector import collect_traces, collect_quality_flags, collect_trace_embeddings

            traces = collect_traces(
                days=self.config.days_lookback,
                limit=self.config.trace_limit,
            )

            if not traces:
                log = TrainingRunLog(
                    run_id=run_id,
                    started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    traces_collected=0,
                    promoted=False,
                    reason="No traces collected from ClickHouse",
                )
                self.history.append(log)
                logger.info(f"[{run_id}] No traces — skipping")
                return log

            # Step 2: Quality flags
            quality_flags = collect_quality_flags(days=self.config.days_lookback)

            # Step 3: Embeddings for drift detection
            embeddings = collect_trace_embeddings(traces, self.embedder)

            # Step 4: Run auto-trainer
            if self.auto_trainer is None:
                log = TrainingRunLog(
                    run_id=run_id,
                    started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    traces_collected=len(traces),
                    promoted=False,
                    reason="No auto_trainer configured",
                )
                self.history.append(log)
                return log

            result = self.auto_trainer.train(
                production_traces=traces,
                quality_flags=quality_flags,
            )

            # Step 5: Reload engine if promoted
            engine_reloaded = False
            if result.promoted and self.config.auto_reload_engine:
                from .engine_reload import reload_engine_weights

                reload = reload_engine_weights(
                    weights_path=self.config.weights_path,
                    engine_url=self.config.engine_url,
                )
                engine_reloaded = reload.success
                if reload.success:
                    logger.info(f"[{run_id}] Engine reloaded: gen {reload.old_generation} → {reload.new_generation}")
                else:
                    logger.warning(f"[{run_id}] Engine reload failed: {reload.message}")

            log = TrainingRunLog(
                run_id=run_id,
                started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                traces_collected=len(traces),
                promoted=result.promoted,
                reason=result.reason,
                auroc_before=result.baseline_metrics.auroc if result.baseline_metrics else None,
                auroc_after=result.new_metrics.auroc if result.new_metrics else None,
                engine_reloaded=engine_reloaded,
            )

            self.history.append(log)
            logger.info(
                f"[{run_id}] Complete: {'PROMOTED' if result.promoted else 'REJECTED'} "
                f"| traces={len(traces)} | engine_reloaded={engine_reloaded}"
            )
            return log

        except Exception as e:
            log = TrainingRunLog(
                run_id=run_id,
                started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                traces_collected=0,
                promoted=False,
                reason="Error",
                error=str(e),
            )
            self.history.append(log)
            logger.error(f"[{run_id}] Training run failed: {e}")
            return log

    async def start(self) -> None:
        """Start the periodic monitoring + training loop.

        If use_advisor=True: checks every interval, only trains when advisor says to.
        If use_advisor=False: trains every interval (legacy behavior).
        """
        if not self.config.enabled:
            logger.info("Scheduled training is disabled")
            return

        self._running = True
        mode = "advisor-driven" if self._advisor else "periodic"
        logger.info(
            f"Starting {mode} training: check every {self.config.check_interval_seconds}s, "
            f"lookback={self.config.days_lookback}d"
        )

        while self._running:
            if self._advisor:
                await self.check_and_train()
            else:
                await self.run_once()
            await asyncio.sleep(self.config.check_interval_seconds)

    def stop(self) -> None:
        """Stop the periodic loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Scheduled training stopped")

    def start_background(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start training loop as a background asyncio task."""
        if loop is None:
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.start())
