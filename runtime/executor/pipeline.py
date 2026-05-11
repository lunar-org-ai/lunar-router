"""Pipeline executor — runs a compiled Pipeline on a request and records a trace.

The executor wraps each stage with timing and i/o capture so that we always
emit a structured ExecutionRecord, even on failure. Fail-fast: if a stage
raises, the pipeline halts and the error is recorded. The record is the
canonical input to the trace writer (P1.7) and to the eval runners.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from runtime.compiler.builder import Pipeline
from runtime.protocols import Context, Message


@dataclass
class StageRecord:
    """Per-stage execution record. One per stage that ran (or attempted)."""

    stage: str
    technique: str
    variant: str
    duration_ms: float
    docs_in: int = 0
    docs_out: int = 0
    response_set: bool = False
    routing_model: Optional[str] = None
    routing_decision: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ExecutionRecord:
    """Full execution record — the trace event for one request.

    Persists `history` so the UI can render the full conversation thread
    instead of just the single turn — every prior turn the caller passed
    in via /run becomes part of the trace's transcript.

    `tokens_in` / `tokens_out` / `cost_usd` are populated by
    ``runtime.cost.estimate_cost`` against the selected `routing_model`.
    With stubs these are char-based estimates; with real LLM (P1.9)
    they become the SDK's actual usage numbers — same fields, no
    schema migration.
    """

    request: str
    response: Optional[str]
    duration_ms: float
    stages: list[StageRecord]
    success: bool
    error: Optional[str] = None
    agent_version: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineExecutor:
    """Runs a compiled Pipeline. One executor per Pipeline; thread-unsafe."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def run(
        self,
        request: str,
        history: Optional[list[Message]] = None,
        session_id: Optional[str] = None,
    ) -> tuple[Context, ExecutionRecord]:
        ctx = Context(request=request, history=list(history or []))
        # Snapshot the history we received so the trace can render the full
        # conversation thread later, not just this single turn.
        history_serialized: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in (history or [])
        ]
        records: list[StageRecord] = []
        success = True
        error: Optional[str] = None

        total_start = time.perf_counter()
        for stage_cfg, stage in zip(self.pipeline.config.pipeline, self.pipeline.stages):
            docs_in = len(ctx.documents)
            stage_start = time.perf_counter()
            err: Optional[str] = None
            try:
                ctx = stage.execute(ctx)
            except Exception as e:  # fail-fast
                err = f"{type(e).__name__}: {e}"
                success = False
                error = err
            duration_ms = round((time.perf_counter() - stage_start) * 1000, 3)
            records.append(
                StageRecord(
                    stage=stage_cfg.stage or stage_cfg.technique,
                    technique=stage_cfg.technique,
                    variant=stage_cfg.variant,
                    duration_ms=duration_ms,
                    docs_in=docs_in,
                    docs_out=len(ctx.documents),
                    response_set=ctx.response is not None,
                    routing_model=ctx.routing.model if ctx.routing else None,
                    routing_decision=(
                        ctx.routing.decision if ctx.routing is not None else None
                    ),
                    error=err,
                )
            )
            if err is not None:
                break

        total_ms = round((time.perf_counter() - total_start) * 1000, 3)

        # Cost estimation — uses the routing_model from the route stage when
        # set, falls back to default. Char-based until P1.9 swaps for real
        # usage numbers from the Anthropic SDK.
        from runtime.cost import estimate_cost
        chosen_model = next(
            (r.routing_model for r in records if r.routing_model is not None),
            None,
        )
        tokens_in, tokens_out, cost_usd = estimate_cost(
            request, ctx.response, model=chosen_model
        )

        record = ExecutionRecord(
            request=request,
            response=ctx.response,
            duration_ms=total_ms,
            stages=records,
            success=success,
            error=error,
            agent_version=self.pipeline.config.version,
            history=history_serialized,
            session_id=session_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        return ctx, record
