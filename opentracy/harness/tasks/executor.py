"""RecipeExecutor — runs a recipe end-to-end, chaining ledger entries.

Contract:
  1. Steps run in declared order.
  2. Each step's ledger entry is a child of the previous step's entry
     (linear chain), not of the original signal — so `ledger.chain()`
     from the signal returns the full recipe in order.
  3. Budget is a hard cap summed over step.cost_usd; exceeding it halts
     the recipe (remaining steps recorded as skipped).
  4. Skip conditions are evaluated against the source step's output
     `data` dict — mismatch records a skipped row and moves on.
  5. Exceptions in an individual step are isolated: the failing step is
     recorded as outcome=failed and subsequent steps are recorded as
     skipped with a `cascade_halt` tag. The executor does not raise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from opentracy.harness.actions import get_action
from opentracy.harness.ledger import LedgerEntry, LedgerStore

from .schema import Recipe, RecipeStep

logger = logging.getLogger(__name__)


class AgentRunnerProtocol(Protocol):
    """Minimal subset of AgentRunner used by the executor.

    Real runner: `opentracy.harness.runner.AgentRunner.run(name, input)`.
    Tests inject a stub with the same shape.
    """

    async def run(self, agent_name: str, user_input: str) -> Any: ...


@dataclass
class ExecutionResult:
    """Outcome of one recipe execution.

    `step_outputs[step.id]` is whatever the step produced (agent .data
    dict or action .data dict). `step_entry_ids[step.id]` is the ledger
    id of the row the step wrote.
    """

    recipe_id: str
    step_outputs: dict[str, dict] = field(default_factory=dict)
    step_entry_ids: dict[str, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None


class RecipeExecutor:
    def __init__(
        self,
        ledger: LedgerStore,
        runner: Optional[AgentRunnerProtocol] = None,
    ):
        self.ledger = ledger
        self.runner = runner

    async def execute(self, recipe: Recipe, root_parent_id: str) -> ExecutionResult:
        result = ExecutionResult(recipe_id=recipe.id)
        previous_entry_id = root_parent_id
        cascade_halted = False

        for step in recipe.steps:
            # ---- budget gate ----------------------------------------
            if result.total_cost_usd >= recipe.budget.max_cost_usd:
                entry = self._record_skip(
                    step,
                    previous_entry_id,
                    reason="budget_exceeded",
                    extra_tags=["budget_exceeded"],
                )
                result.step_entry_ids[step.id] = entry.id
                result.halted = True
                result.halt_reason = "budget_exceeded"
                cascade_halted = True
                previous_entry_id = entry.id
                continue

            # ---- cascade after failure ------------------------------
            if cascade_halted:
                entry = self._record_skip(
                    step,
                    previous_entry_id,
                    reason="cascade_halt",
                    extra_tags=["cascade_halt"],
                )
                result.step_entry_ids[step.id] = entry.id
                previous_entry_id = entry.id
                continue

            # ---- condition gate -------------------------------------
            if step.condition is not None:
                source = result.step_outputs.get(step.condition.from_step, {})
                if source.get(step.condition.field) != step.condition.equals:
                    entry = self._record_skip(
                        step,
                        previous_entry_id,
                        reason="condition_unmet",
                        extra_tags=["condition_unmet"],
                    )
                    result.step_entry_ids[step.id] = entry.id
                    result.step_outputs[step.id] = {"skipped": True}
                    previous_entry_id = entry.id
                    continue

            # ---- resolve input --------------------------------------
            step_input = self._resolve_input(step, result.step_outputs)

            # ---- run step -------------------------------------------
            try:
                if step.type == "agent":
                    entry_id, output, cost = await self._run_agent(
                        step, step_input, previous_entry_id,
                    )
                else:
                    entry_id, output, cost = await self._run_action(
                        step, step_input, previous_entry_id,
                    )
                result.step_entry_ids[step.id] = entry_id
                result.step_outputs[step.id] = output
                result.total_cost_usd += cost
                previous_entry_id = entry_id
            except Exception as e:
                logger.warning(
                    f"recipe {recipe.id!r} step {step.id!r} raised "
                    f"{type(e).__name__}: {e}"
                )
                entry = LedgerEntry(
                    type="run" if step.type == "agent" else "action",
                    agent=step.agent or step.action,
                    parent_id=previous_entry_id,
                    data={"error": str(e), "error_type": type(e).__name__},
                    tags=[step.id, step.type, "step_failed"],
                    outcome="failed",
                )
                self.ledger.append(entry)
                result.step_entry_ids[step.id] = entry.id
                result.halted = True
                result.halt_reason = f"{step.id}:{type(e).__name__}"
                cascade_halted = True
                previous_entry_id = entry.id

        return result

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self, step: RecipeStep, step_input: Any, parent_id: str,
    ) -> tuple[str, dict, float]:
        if self.runner is None:
            raise RuntimeError(
                f"recipe step {step.id!r} is an agent call but no AgentRunner "
                "was supplied to RecipeExecutor"
            )
        started = datetime.now(timezone.utc)
        prompt = _format_agent_input(step_input)
        agent_result = await self.runner.run(step.agent, prompt)

        data = _extract_data(agent_result)
        cost = float(getattr(agent_result, "cost_usd", 0.0) or 0.0)
        duration_ms = getattr(agent_result, "duration_ms", None)
        if duration_ms is None:
            duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

        entry = LedgerEntry(
            type="run",
            agent=step.agent,
            parent_id=parent_id,
            parameters_in={"recipe_step": step.id},
            data=data,
            tags=[step.id, "agent", step.agent or ""],
            cost_usd=cost,
            duration_ms=int(duration_ms),
            outcome="ok",
        )
        self.ledger.append(entry)
        return entry.id, data, cost

    async def _run_action(
        self, step: RecipeStep, step_input: Any, parent_id: str,
    ) -> tuple[str, dict, float]:
        fn = get_action(step.action)
        if fn is None:
            raise RuntimeError(f"action {step.action!r} is not registered")
        # Actions take dict inputs; coerce if the prior step produced a non-dict.
        action_inputs = step_input if isinstance(step_input, dict) else {"text": str(step_input)}
        action_result = await fn(action_inputs, self.ledger, parent_id)
        entry_id = action_result.ledger_entry_id
        if entry_id is None:
            # Defensive: action didn't write its own entry. Write one now
            # so the chain isn't broken.
            entry = LedgerEntry(
                type="action",
                agent=step.action,
                parent_id=parent_id,
                data=action_result.data,
                tags=[step.id, "action", step.action or ""],
                outcome=action_result.outcome,
                cost_usd=action_result.cost_usd,
                duration_ms=action_result.duration_ms,
            )
            self.ledger.append(entry)
            entry_id = entry.id
        return entry_id, dict(action_result.data), float(action_result.cost_usd or 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_skip(
        self, step: RecipeStep, parent_id: str,
        reason: str, extra_tags: list[str],
    ) -> LedgerEntry:
        entry = LedgerEntry(
            type="run" if step.type == "agent" else "action",
            agent=step.agent or step.action,
            parent_id=parent_id,
            parameters_in={"recipe_step": step.id, "skip_reason": reason},
            tags=[step.id, step.type, "skipped", *extra_tags],
            outcome="skipped",
        )
        self.ledger.append(entry)
        return entry

    def _resolve_input(
        self, step: RecipeStep, outputs: dict[str, dict],
    ) -> Any:
        if step.input_from:
            return outputs.get(step.input_from, {})
        if step.input is not None:
            return step.input
        return {}


def _format_agent_input(step_input: Any) -> str:
    """Turn a step's resolved input into the string prompt the agent
    receives. Dicts get pretty-printed JSON so downstream agents can
    reliably parse; raw strings pass through."""
    if isinstance(step_input, str):
        return step_input
    if isinstance(step_input, dict):
        return (
            "## Context from previous step\n\n"
            f"```json\n{json.dumps(step_input, indent=2, default=str)}\n```"
        )
    return str(step_input)


def _extract_data(agent_result: Any) -> dict:
    """Pull the `.data` dict off an AgentRunner result, tolerating both
    the real AgentResult class and simple stubs that expose `.data`."""
    if hasattr(agent_result, "data"):
        data = agent_result.data
    elif isinstance(agent_result, dict):
        data = agent_result.get("data", agent_result)
    else:
        data = {}
    return dict(data) if isinstance(data, dict) else {}
