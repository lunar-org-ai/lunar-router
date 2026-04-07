"""
Production feedback loop for continuous router improvement.

Converts ClickHouse traces into training signal, detects distribution
drift, and incrementally updates router weights with zero-downtime.

Full pipeline:
    collect_traces() → TraceToTraining → AutoTrainer → reload_engine_weights()
    (ClickHouse)       (Psi update)     (quality gate)  (Go hot-reload)
"""

from .trace_to_training import TraceToTraining, TraceRecord, ProductionPsiUpdate
from .drift_detector import DriftDetector, DriftReport
from .incremental_updater import IncrementalUpdater, UpdateResult
from .collector import collect_traces, collect_quality_flags, collect_trace_embeddings
from .engine_reload import reload_engine_weights, check_engine_health, ReloadResult
from .scheduled_trainer import ScheduledTrainer, TrainingScheduleConfig, TrainingRunLog

__all__ = [
    # Core
    "TraceToTraining",
    "TraceRecord",
    "ProductionPsiUpdate",
    "DriftDetector",
    "DriftReport",
    "IncrementalUpdater",
    "UpdateResult",
    # ClickHouse collector
    "collect_traces",
    "collect_quality_flags",
    "collect_trace_embeddings",
    # Engine reload
    "reload_engine_weights",
    "check_engine_health",
    "ReloadResult",
    # Scheduled runner
    "ScheduledTrainer",
    "TrainingScheduleConfig",
    "TrainingRunLog",
]
