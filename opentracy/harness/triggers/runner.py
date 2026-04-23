"""TriggerEngineLoop — the async wrapper that makes the trigger engine
actually run in production.

Responsibilities:
  - Load objectives / policies / recipes from YAML on start.
  - Construct the engine with:
      * ObjectiveSensors (regression detection)
      * CadenceSensors (periodic heartbeat per objective)
      * Dispatch handlers: `__recipe__` wired to RecipeExecutor.
  - Drive `engine.tick()` on a fixed cadence, await pending recipe
    coroutines before sleeping.
  - Graceful shutdown via asyncio cancel.

This module is where all the YAML-declarative pieces come together.
The engine, sensors, recipes, actions, and agents remain isolated and
testable; the loop is the only place aware of "the full system."
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from opentracy.harness.ledger import LedgerStore, get_ledger_store
from opentracy.harness.objectives.loader import load_all as load_all_objectives
from opentracy.harness.sensors import (
    CadenceSensor,
    NewDatasetSensor,
    NewTracesThresholdSensor,
)
from opentracy.harness.tasks import RecipeExecutor, load_recipes
from opentracy.harness.tasks.executor import AgentRunnerProtocol

from .engine import TriggerEngine
from .policies import Policy, load_policies

logger = logging.getLogger(__name__)


# Map `update_cadence` strings on objectives to default cadence-sensor
# intervals. Contributors can override per-deployment via a custom
# cadence_sensors list, but the defaults cover the shipped YAMLs.
_CADENCE_INTERVAL_HOURS = {
    "hourly": 1.0,
    "daily": 24.0,
    "weekly": 168.0,
}


class TriggerEngineLoop:
    """Background asyncio loop that drives the trigger engine.

    Typical lifecycle:
        loop = TriggerEngineLoop(ledger=..., agent_runner=...)
        await loop.start()
        ...
        await loop.stop()

    The loop is safe to construct outside of an event loop; only
    `start()` touches asyncio primitives. Tests can drive the engine
    directly via `await loop.run_once()` without needing the background
    task.
    """

    def __init__(
        self,
        ledger: Optional[LedgerStore] = None,
        agent_runner: Optional[AgentRunnerProtocol] = None,
        interval_seconds: float = 60.0,
        cadence_interval_hours: Optional[dict[str, float]] = None,
        new_traces_threshold: int = 1000,
        new_traces_cooldown_hours: float = 6.0,
        new_traces_objective_id: str = "domain_coverage_ratio",
    ):
        self.ledger = ledger or get_ledger_store()
        self.agent_runner = agent_runner
        self.interval_seconds = interval_seconds

        self.objectives = load_all_objectives()
        self.policies = load_policies()
        self.recipes = load_recipes()
        self._recipes_by_id = {r.id: r for r in self.recipes}

        interval_map = {**_CADENCE_INTERVAL_HOURS, **(cadence_interval_hours or {})}
        cadence_sensors = [
            CadenceSensor(
                obj,
                self.ledger,
                interval_hours=interval_map.get(obj.update_cadence, 24.0),
            )
            for obj in self.objectives
        ]

        # One traces-threshold sensor tied to the objective whose id
        # matches `new_traces_objective_id`. Defaults to
        # domain_coverage_ratio because fresh trace volume = new
        # patterns = potential coverage drop. Skip silently if the
        # target objective isn't defined.
        traces_sensors = []
        target_obj = next(
            (o for o in self.objectives if o.id == new_traces_objective_id),
            None,
        )
        if target_obj is not None:
            traces_sensors = [
                NewTracesThresholdSensor(
                    objective=target_obj,
                    ledger=self.ledger,
                    threshold=new_traces_threshold,
                    cooldown_hours=new_traces_cooldown_hours,
                )
            ]

        # New-dataset sensor tied to the same objective — when
        # clustering produces a new run, suggestion policies fire.
        new_dataset_sensors = []
        if target_obj is not None:
            new_dataset_sensors = [
                NewDatasetSensor(objective=target_obj, ledger=self.ledger),
            ]

        self._executor = RecipeExecutor(self.ledger, runner=self.agent_runner)

        self.engine = TriggerEngine(
            objectives=self.objectives,
            policies=self.policies,
            ledger=self.ledger,
            dispatch_handlers={"__recipe__": self._recipe_dispatch},
            cadence_sensors=cadence_sensors,
            traces_threshold_sensors=traces_sensors,
            new_dataset_sensors=new_dataset_sensors,
        )

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="trigger_engine_loop")
        logger.info(
            f"TriggerEngineLoop started: {len(self.objectives)} objectives, "
            f"{len(self.policies)} policies, {len(self.recipes)} recipes, "
            f"interval={self.interval_seconds}s"
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("TriggerEngineLoop stopped")

    # ------------------------------------------------------------------
    # Exposed for tests and one-off drives
    # ------------------------------------------------------------------

    async def run_once(self) -> None:
        """Drive one tick synchronously, awaiting any pending recipe
        coroutines. Used by tests and can also be called from a CLI."""
        result = self.engine.tick()
        if result.pending:
            await asyncio.gather(*result.pending, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main loop. Tick, await pending recipes, sleep, repeat.

        Wrapped in a broad try/except so a bad tick doesn't kill the
        loop — logs the error and keeps going. Graceful shutdown via
        cancellation (see `stop`).
        """
        try:
            while not self._stop_event.is_set():
                try:
                    await self.run_once()
                except Exception as e:
                    logger.exception(f"TriggerEngineLoop tick failed: {e}")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    # Normal path — interval elapsed, run next tick.
                    pass
        except asyncio.CancelledError:
            # Graceful shutdown.
            return

    def _recipe_dispatch(self, signal, dispatch_id: str, policy: Policy):
        """Dispatch handler for recipe policies. Must return a coroutine
        so the engine collects it in TickResult.pending; the runner
        then awaits all pending coroutines before the next tick.
        """
        recipe_id = policy.dispatch.recipe
        recipe = self._recipes_by_id.get(recipe_id)
        if recipe is None:
            logger.warning(
                f"Policy {policy.id!r} references unknown recipe {recipe_id!r}"
            )
            return None
        # Return the coroutine — engine collects it into TickResult.pending.
        return self._executor.execute(recipe, root_parent_id=dispatch_id)
