"""Scheduler — periodic trace scanning using asyncio.

No external dependencies (no APScheduler, no celery). Uses a simple
asyncio loop with configurable interval. Stores schedule config in memory.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .memory_store import MemoryEntry, MemoryStore, get_memory_store
from .trace_scanner import TraceScanner

logger = logging.getLogger(__name__)

SCHEDULE_CATEGORY = "scan_schedule"
SCHEDULE_CONFIG_ID = "scan_schedule_config"


@dataclass
class ScheduleConfig:
    """Configuration for periodic scans."""

    enabled: bool = False
    interval_seconds: int = 3600  # default: hourly
    days_lookback: int = 7
    trace_limit: int = 100
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    total_runs: int = 0
    total_issues_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "days_lookback": self.days_lookback,
            "trace_limit": self.trace_limit,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "total_runs": self.total_runs,
            "total_issues_found": self.total_issues_found,
        }


class ScanScheduler:
    """Manages periodic scan execution via asyncio."""

    def __init__(
        self,
        engine_url: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.engine_url = engine_url
        self.memory_store = memory_store or get_memory_store()
        self._config = ScheduleConfig()
        self._task: Optional[asyncio.Task] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load schedule config from memory if it exists."""
        entries = self.memory_store.query(
            agent="scheduler",
            category=SCHEDULE_CATEGORY,
            tags=["config"],
            limit=1,
        )
        if entries:
            ev = entries[0].evaluation
            self._config = ScheduleConfig(
                enabled=ev.get("enabled", False),
                interval_seconds=ev.get("interval_seconds", 3600),
                days_lookback=ev.get("days_lookback", 7),
                trace_limit=ev.get("trace_limit", 100),
                last_run_at=ev.get("last_run_at"),
                total_runs=ev.get("total_runs", 0),
                total_issues_found=ev.get("total_issues_found", 0),
            )

    def _save_config(self) -> None:
        """Persist schedule config to memory."""
        # Delete old config entry if it exists
        old = self.memory_store.get(SCHEDULE_CONFIG_ID)
        if old is not None:
            self.memory_store.delete(SCHEDULE_CONFIG_ID)

        entry = MemoryEntry(
            id=SCHEDULE_CONFIG_ID,
            agent="scheduler",
            category=SCHEDULE_CATEGORY,
            created_at=datetime.now(timezone.utc).isoformat(),
            body=(
                "## Scan Schedule Configuration\n\n"
                f"- **Enabled:** {self._config.enabled}\n"
                f"- **Interval:** {self._config.interval_seconds}s\n"
                f"- **Lookback:** {self._config.days_lookback} days\n"
                f"- **Trace limit:** {self._config.trace_limit}\n"
                f"- **Total runs:** {self._config.total_runs}\n"
                f"- **Total issues found:** {self._config.total_issues_found}\n"
            ),
            tags=["config"],
            evaluation=self._config.to_dict(),
        )
        self.memory_store.save(entry)

    @property
    def config(self) -> ScheduleConfig:
        return self._config

    def update_config(
        self,
        enabled: Optional[bool] = None,
        interval_seconds: Optional[int] = None,
        days_lookback: Optional[int] = None,
        trace_limit: Optional[int] = None,
    ) -> ScheduleConfig:
        if enabled is not None:
            self._config.enabled = enabled
        if interval_seconds is not None:
            self._config.interval_seconds = max(60, interval_seconds)
        if days_lookback is not None:
            self._config.days_lookback = days_lookback
        if trace_limit is not None:
            self._config.trace_limit = trace_limit

        self._save_config()

        # Start/stop based on enabled state
        if self._config.enabled and self._task is None:
            self.start()
        elif not self._config.enabled and self._task is not None:
            self.stop()

        return self._config

    def start(self) -> None:
        """Start the periodic scan loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Scan scheduler started (interval={self._config.interval_seconds}s)"
        )

    def stop(self) -> None:
        """Stop the periodic scan loop."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
            logger.info("Scan scheduler stopped")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run_loop(self) -> None:
        """Main loop: sleep then scan, repeat."""
        while True:
            try:
                await asyncio.sleep(self._config.interval_seconds)
                if not self._config.enabled:
                    break
                await self._execute_scan()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled scan failed: {e}")
                # Continue the loop — don't crash on transient errors

    async def _execute_scan(self) -> None:
        """Run a single scan and update metrics."""
        scan_id = str(uuid.uuid4())
        scanner = TraceScanner(
            engine_url=self.engine_url,
            memory_store=self.memory_store,
        )

        logger.info(f"Scheduled scan {scan_id} starting...")
        issues = await scanner.scan(
            scan_id=scan_id,
            days=self._config.days_lookback,
            limit=self._config.trace_limit,
        )

        self._config.last_run_at = datetime.now(timezone.utc).isoformat()
        self._config.total_runs += 1
        self._config.total_issues_found += len(issues)
        self._save_config()

        logger.info(
            f"Scheduled scan {scan_id} completed: "
            f"{len(issues)} issues found (total runs: {self._config.total_runs})"
        )


# Singleton
_scheduler: Optional[ScanScheduler] = None


def get_scheduler(engine_url: Optional[str] = None) -> ScanScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ScanScheduler(engine_url=engine_url)
    return _scheduler
