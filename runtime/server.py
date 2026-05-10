"""HTTP service exposing the runtime.

This is the in-process target for the TS backend. The server compiles
agent.yaml once at startup; when the harness promotes a new version, restart
the service (hot-reload comes in a later phase).

Endpoints:
  POST /run                            — execute one request through the pipeline.
  POST /introspect                     — ask the harness about itself.
  GET  /health                         — liveness + agent version.
  GET  /agent                          — current agent.yaml summary.
  GET  /versions                       — list all snapshotted versions + lessons.
  POST /versions/{version}/rollback    — restore live agent/ to a prior version.
  GET  /lessons                        — flat lesson feed (Evolution timeline).
  GET  /lessons/{id}                   — single lesson by id.
  POST /lessons/{id}/approve           — approve a queued review lesson + promote.
  POST /lessons/{id}/reject            — reject a queued review lesson.
  POST /lessons/{id}/requeue           — undo an approve/reject; back to queue.
  GET  /lessons/{id}/traces            — eval cases the candidate ran through.
  GET  /metrics/overview               — derived dashboard metrics.
  GET  /policy                         — current approval policy.
  PUT  /policy                         — update approval policy YAML.
  GET  /agent/config                   — full AgentSheet snapshot (prompt,
                                          models, integrations, key status).
  PUT  /agent/prompt                   — rewrite agent/prompts/system.md.
  PUT  /agent/route                    — rewrite route.yaml model knobs.
  GET  /traces                         — list raw traces (Technical / Traces).
  GET  /traces/{trace_id}              — single trace with full stages + history.
  GET  /sessions/{session_id}          — every trace sharing a session id (dialog).
  GET  /evals/suites                   — list eval suites (Technical / Eval).
  GET  /evals/suites/{name}            — suite detail (goldens + rubrics).
  GET  /evals/reports                  — recent runs (filterable by suite).
  GET  /evals/reports/{report_id}      — single eval report with cases.
  GET  /router/config                  — current router_config metadata (P15.3); cold-start safe.
  POST /router/decide                  — score a prompt against the router; no LLM call.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from runtime.compiler.builder import compile_agent
from runtime.compiler.loader import load_agent
from runtime.executor.pipeline import PipelineExecutor
from runtime.executor.tracing import bus as trace_bus
from runtime.executor.tracing import write_trace
from runtime.protocols import Message
from runtime.store import traces as traces_store

logger = logging.getLogger(__name__)


class HistoryMessage(BaseModel):
    role: str
    content: str


class RunRequest(BaseModel):
    request: str
    history: Optional[list[HistoryMessage]] = None
    session_id: Optional[str] = None


class StageOutcome(BaseModel):
    stage: str
    technique: str
    variant: str
    duration_ms: float
    docs_in: int
    docs_out: int
    routing_model: Optional[str] = None
    error: Optional[str] = None


class RunResponse(BaseModel):
    response: Optional[str]
    trace_id: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    agent_version: Optional[str] = None
    stages: list[StageOutcome]


class HealthResponse(BaseModel):
    status: str
    agent_version: Optional[str]


class StageInfo(BaseModel):
    stage: Optional[str]
    technique: str
    variant: str


class AgentInfo(BaseModel):
    version: str
    description: Optional[str]
    pipeline: list[StageInfo]
    cross_cutting: dict[str, Any]
    budget: dict[str, Any]


_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio as _asyncio

    cfg = load_agent("agent/agent.yaml")
    pipeline = compile_agent(cfg)
    executor = PipelineExecutor(pipeline)
    _state["cfg"] = cfg
    _state["executor"] = executor
    # TraceBus needs the running event loop to schedule fan-out from the
    # synchronous write_trace path back onto async subscriber queues.
    trace_bus.attach_loop(_asyncio.get_running_loop())
    logger.info("agent %s ready (%d stages)", cfg.version, len(pipeline.stages))
    yield
    _state.clear()


app = FastAPI(title="opentracy-runtime", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cfg = _state.get("cfg")
    return HealthResponse(
        status="ok" if cfg else "starting",
        agent_version=cfg.version if cfg else None,
    )


@app.get("/agent", response_model=AgentInfo)
async def agent() -> AgentInfo:
    cfg = _state.get("cfg")
    if not cfg:
        raise HTTPException(status_code=503, detail="agent not yet loaded")
    return AgentInfo(
        version=cfg.version,
        description=cfg.description,
        pipeline=[
            StageInfo(stage=s.stage, technique=s.technique, variant=s.variant)
            for s in cfg.pipeline
        ],
        cross_cutting=cfg.cross_cutting.model_dump(),
        budget=cfg.budget.model_dump(),
    )


class IntrospectRequest(BaseModel):
    request: str
    history: Optional[list[HistoryMessage]] = None


class IntrospectToolCall(BaseModel):
    tool: str
    input: dict[str, Any]
    output_preview: str


class IntrospectResponse(BaseModel):
    response: str
    tool_calls: list[IntrospectToolCall]
    success: bool
    error: Optional[str] = None
    model: Optional[str] = None
    iterations: int = 0


@app.post("/introspect", response_model=IntrospectResponse)
async def introspect_endpoint(payload: IntrospectRequest) -> IntrospectResponse:
    from harness.introspection.agent import introspect

    history = [{"role": m.role, "content": m.content} for m in (payload.history or [])]
    result = introspect(payload.request, history)
    return IntrospectResponse(
        response=result.response,
        tool_calls=[
            IntrospectToolCall(tool=tc.tool, input=tc.input, output_preview=tc.output_preview)
            for tc in result.tool_calls
        ],
        success=result.success,
        error=result.error,
        model=result.model,
        iterations=result.iterations,
    )


# ---------- versions ----------


class LessonSummary(BaseModel):
    id: str
    version: Optional[str] = None
    kind: str
    status: str
    title: str
    summary: str
    voice: Optional[str] = None
    delta: dict[str, Any] = {}
    mutations: list[str] = []
    parent_version: Optional[str] = None
    candidate_id: Optional[str] = None
    promoted_at: Optional[str] = None
    ledger_entry_id: Optional[str] = None
    proposal_source: Optional[str] = None
    n_traces: Optional[int] = None  # cases in candidate eval report, if persisted


def _lesson_trace_count(candidate_id: Optional[str]) -> Optional[int]:
    """Cheap on-disk lookup so the Evolution timeline can show "N traces" per
    lesson without paying a per-card round-trip."""
    if not candidate_id:
        return None
    import json
    from pathlib import Path

    report_path = (
        Path(__file__).resolve().parent.parent
        / "evals"
        / "reports"
        / f"cand_{candidate_id}.json"
    )
    if not report_path.exists():
        return None
    try:
        with report_path.open() as f:
            report = json.load(f)
        return len(report.get("cases", []))
    except Exception:
        return None


def _lesson_to_summary(lesson: Any) -> "LessonSummary":
    return LessonSummary(
        id=lesson.id,
        version=lesson.version,
        kind=lesson.kind,
        status=lesson.status,
        title=lesson.title,
        summary=lesson.summary,
        voice=lesson.voice,
        delta=lesson.delta,
        mutations=lesson.mutations,
        parent_version=lesson.parent_version,
        candidate_id=lesson.candidate_id or None,
        promoted_at=lesson.promoted_at,
        ledger_entry_id=lesson.ledger_entry_id,
        proposal_source=lesson.proposal_source,
        n_traces=_lesson_trace_count(lesson.candidate_id),
    )


class VersionInfo(BaseModel):
    id: str
    is_live: bool
    status: str          # "live" | "rolled_back" | "archived"
    snapshot_path: str
    promoted_at: Optional[str] = None
    rolled_back_at: Optional[str] = None
    lesson: Optional[LessonSummary] = None


@app.get("/versions", response_model=list[VersionInfo])
async def list_versions() -> list[VersionInfo]:
    from ledger.versioning import list_snapshots, read_version, snapshot_path
    from ledger.writer import read_entries, read_lessons

    live = read_version()
    snapshots = list_snapshots()

    entries = read_entries()
    promotes_by_target: dict[str, list[Any]] = {}
    rollbacks_by_target: dict[str, list[Any]] = {}
    for e in entries:
        if e.kind == "promote" and e.agent_version_after:
            promotes_by_target.setdefault(e.agent_version_after, []).append(e)
        if e.kind == "rollback" and e.agent_version_before:
            rollbacks_by_target.setdefault(e.agent_version_before, []).append(e)

    lessons_by_version: dict[str, Any] = {}
    for lesson in read_lessons():
        lessons_by_version.setdefault(lesson.version, lesson)

    out: list[VersionInfo] = []
    for v in snapshots:
        is_live = v == live
        rolled_back = bool(rollbacks_by_target.get(v))
        if is_live:
            status = "live"
        elif rolled_back:
            status = "rolled_back"
        else:
            status = "archived"

        promoted_at: Optional[str] = None
        if promotes_by_target.get(v):
            promoted_at = promotes_by_target[v][0].timestamp

        rolled_back_at: Optional[str] = None
        if rollbacks_by_target.get(v):
            rolled_back_at = rollbacks_by_target[v][-1].timestamp

        lesson = lessons_by_version.get(v)
        lesson_summary: Optional[LessonSummary] = None
        if lesson is not None:
            lesson_summary = _lesson_to_summary(lesson)

        out.append(
            VersionInfo(
                id=v,
                is_live=is_live,
                status=status,
                snapshot_path=str(snapshot_path(v)),
                promoted_at=promoted_at,
                rolled_back_at=rolled_back_at,
                lesson=lesson_summary,
            )
        )

    out.sort(key=lambda x: x.promoted_at or "", reverse=True)
    return out


# ---------- lessons ----------
#
# AHE paper three pillars (arxiv 2604.25850) structure these endpoints:
#   - component:  Lesson.mutations (what changed in the agent config)
#   - experience: Lesson.delta (eval rubric movement) + linked traces (TODO)
#   - decision:   Lesson.voice + proposal_source + (future) Prediction.rationale
#
# A flat lesson feed lets the Evolution timeline render newest-first.


@app.get("/lessons", response_model=list[LessonSummary])
async def list_lessons() -> list[LessonSummary]:
    from ledger.writer import read_lessons

    items = [_lesson_to_summary(l) for l in read_lessons()]
    items.sort(key=lambda x: x.promoted_at or "", reverse=True)
    return items


@app.get("/lessons/{lesson_id}", response_model=LessonSummary)
async def get_lesson(lesson_id: str) -> LessonSummary:
    from ledger.writer import read_lessons

    for l in read_lessons():
        if l.id == lesson_id:
            return _lesson_to_summary(l)
    raise HTTPException(status_code=404, detail=f"unknown lesson {lesson_id!r}")


class ReviewActionRequest(BaseModel):
    reviewer: Optional[str] = None
    reason: Optional[str] = None


@app.post("/lessons/{lesson_id}/approve", response_model=LessonSummary)
async def approve_lesson(
    lesson_id: str, payload: Optional[ReviewActionRequest] = None
) -> LessonSummary:
    from harness.executor.promote import promote_queued

    try:
        lesson = promote_queued(
            lesson_id, reviewer=(payload.reviewer if payload else None) or "ui"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _lesson_to_summary(lesson)


@app.post("/lessons/{lesson_id}/reject", response_model=LessonSummary)
async def reject_lesson(
    lesson_id: str, payload: Optional[ReviewActionRequest] = None
) -> LessonSummary:
    from harness.executor.promote import reject_queued

    try:
        lesson = reject_queued(lesson_id, reason=(payload.reason if payload else None))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _lesson_to_summary(lesson)


@app.post("/lessons/{lesson_id}/requeue", response_model=LessonSummary)
async def requeue_lesson(lesson_id: str) -> LessonSummary:
    from harness.executor.promote import requeue

    try:
        lesson = requeue(lesson_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _lesson_to_summary(lesson)


class LessonTraceCase(BaseModel):
    golden_id: str
    request: str
    response: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    trace_id: Optional[str] = None
    rubric_results: list[dict[str, Any]] = []


class LessonTracesResponse(BaseModel):
    lesson_id: str
    candidate_id: Optional[str] = None
    suite: Optional[str] = None
    agent_version: Optional[str] = None
    has_report: bool = False
    note: Optional[str] = None
    cases: list[LessonTraceCase] = []


@app.get("/lessons/{lesson_id}/traces", response_model=LessonTracesResponse)
async def get_lesson_traces(lesson_id: str) -> LessonTracesResponse:
    """Return the eval cases (request/response + rubric verdicts) the
    candidate ran through. Source: evals/reports/cand_<candidate_id>.json,
    persisted by experiments.runner since P12.1.

    Lessons promoted before P12.1 have no persisted report — we return
    has_report=false with a note so the UI can render an honest empty state.
    """
    import json
    from pathlib import Path

    from ledger.writer import read_lesson

    lesson = read_lesson(lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail=f"unknown lesson {lesson_id!r}")

    cand_id = lesson.candidate_id or ""
    project_root = Path(__file__).resolve().parent.parent
    report_path = project_root / "evals" / "reports" / f"cand_{cand_id}.json"

    if not cand_id or not report_path.exists():
        return LessonTracesResponse(
            lesson_id=lesson_id,
            candidate_id=cand_id or None,
            has_report=False,
            note=(
                "No candidate report on disk for this lesson — this lesson predates "
                "candidate-report persistence. Run a fresh sweep to capture lineage on "
                "future lessons."
            ),
            cases=[],
        )

    with report_path.open() as f:
        report = json.load(f)

    cases = [
        LessonTraceCase(
            golden_id=c.get("golden_id", ""),
            request=c.get("request", ""),
            response=c.get("response"),
            duration_ms=c.get("duration_ms"),
            success=bool(c.get("success", False)),
            error=c.get("error"),
            trace_id=c.get("trace_id"),
            rubric_results=c.get("rubric_results", []),
        )
        for c in report.get("cases", [])
    ]

    return LessonTracesResponse(
        lesson_id=lesson_id,
        candidate_id=cand_id,
        suite=report.get("suite"),
        agent_version=report.get("agent_version"),
        has_report=True,
        cases=cases,
    )


class RollbackRequest(BaseModel):
    reason: Optional[str] = None


class RollbackResponse(BaseModel):
    version: str
    previous_version: str
    rolled_back: bool


@app.post("/versions/{version}/rollback", response_model=RollbackResponse)
async def rollback_version(version: str, payload: Optional[RollbackRequest] = None) -> RollbackResponse:
    from ledger.versioning import list_snapshots, read_version
    from harness.rollback import rollback_to

    if version not in list_snapshots():
        raise HTTPException(
            status_code=404,
            detail=f"unknown version {version!r}; available: {list_snapshots()}",
        )

    previous = read_version()
    if previous == version:
        return RollbackResponse(version=version, previous_version=previous, rolled_back=False)

    reason = (payload.reason if payload else None) or "ui rollback"
    rollback_to(version, reason=reason)
    return RollbackResponse(version=version, previous_version=previous, rolled_back=True)


# ---------- metrics ----------
#
# Dashboard numbers derived from what we actually have on disk:
#   - traces/raw/<YYYY-MM-DD>.jsonl  → today_count, active_5min, resolution_rate, latency
#   - ledger/entries/*.jsonl         → promote/rollback ratio → trust_score + history
#   - ledger/lessons/*.json          → pending_review
#
# Fields we can't derive yet (avg_cost_usd, csat) are returned as null so the
# UI can render them as "—" until production traffic lands.


class MetricsOverview(BaseModel):
    today_count: int
    active_5min: int
    pending_review: int
    trust_score: int
    trust_score_delta_30d: int
    trust_history_30d: list[int]
    resolution_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    avg_cost_usd: Optional[float] = None
    csat: Optional[float] = None
    computed_at: str


@app.get("/metrics/overview", response_model=MetricsOverview)
async def metrics_overview() -> MetricsOverview:
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    from ledger.writer import read_entries, read_lessons

    now = datetime.now(timezone.utc)

    # Trace-derived metrics: today_count, active_5min, resolution_rate,
    # avg_latency_ms — one DuckDB query over the JSONL+Parquet union.
    trace_metrics = traces_store.metrics_traces_window(window_days=7)

    entries = read_entries()
    by_date_p: dict[str, int] = defaultdict(int)
    by_date_r: dict[str, int] = defaultdict(int)
    for e in entries:
        d = (e.timestamp or "")[:10]
        if not d:
            continue
        if e.kind == "promote":
            by_date_p[d] += 1
        elif e.kind == "rollback":
            by_date_r[d] += 1

    history: list[int] = []
    cum_p, cum_r = 0, 0
    for i in range(29, -1, -1):
        d = (now - timedelta(days=i)).date().isoformat()
        cum_p += by_date_p.get(d, 0)
        cum_r += by_date_r.get(d, 0)
        if cum_p > 0:
            history.append(max(0, min(100, int(round((1 - cum_r / cum_p) * 100)))))
        else:
            history.append(70)  # baseline before any promotions

    trust_score = history[-1]
    trust_score_delta_30d = trust_score - history[0]

    lessons = read_lessons()
    pending = sum(1 for l in lessons if l.status in ("pending", "awaiting_review"))

    return MetricsOverview(
        today_count=trace_metrics["today_count"],
        active_5min=trace_metrics["active_5min"],
        pending_review=pending,
        trust_score=trust_score,
        trust_score_delta_30d=trust_score_delta_30d,
        trust_history_30d=history,
        resolution_rate=trace_metrics["resolution_rate"],
        avg_latency_ms=trace_metrics["avg_latency_ms"],
        avg_cost_usd=None,
        csat=None,
        computed_at=now.isoformat(),
    )


# ---------- policy ----------


class AutoRollbackView(BaseModel):
    csat_drop: float = 0.3
    resolution_drop: float = 0.05
    window_hours: int = 24
    notify_channels: list[str] = ["email"]


class PolicyView(BaseModel):
    mode: str
    auto_min_lift: float
    overrides: dict[str, str] = {}
    auto_rollback: AutoRollbackView = AutoRollbackView()


VALID_MODES = ("auto", "review", "off")


def _policy_to_view(pol: Any) -> PolicyView:
    return PolicyView(
        mode=pol.mode,
        auto_min_lift=pol.auto_min_lift,
        overrides=dict(pol.overrides),
        auto_rollback=AutoRollbackView(
            csat_drop=pol.auto_rollback.csat_drop,
            resolution_drop=pol.auto_rollback.resolution_drop,
            window_hours=pol.auto_rollback.window_hours,
            notify_channels=list(pol.auto_rollback.notify_channels),
        ),
    )


@app.get("/policy", response_model=PolicyView)
async def get_policy() -> PolicyView:
    from harness.approver import Policy

    return _policy_to_view(Policy.from_yaml())


class PolicyUpdateRequest(BaseModel):
    mode: str
    auto_min_lift: float
    overrides: dict[str, str] = {}
    auto_rollback: Optional[AutoRollbackView] = None


@app.put("/policy", response_model=PolicyView)
async def update_policy(payload: PolicyUpdateRequest) -> PolicyView:
    """Persist policy changes to policies/auto_approve.yaml. Validates every
    mode field (global + each per-kind override) so the UI can't write
    something the approver doesn't understand."""
    from harness.approver.policy import AutoRollback, Policy

    if payload.mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {VALID_MODES}, got {payload.mode!r}",
        )
    if payload.auto_min_lift < 0 or payload.auto_min_lift > 1:
        raise HTTPException(
            status_code=400,
            detail=f"auto_min_lift must be in [0, 1], got {payload.auto_min_lift}",
        )
    for kind, mode in payload.overrides.items():
        if mode not in VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"override for {kind!r} must be one of {VALID_MODES}, got {mode!r}",
            )

    ar_in = payload.auto_rollback or AutoRollbackView()
    if ar_in.csat_drop < 0 or ar_in.csat_drop > 5:
        raise HTTPException(status_code=400, detail="csat_drop must be in [0, 5]")
    if ar_in.resolution_drop < 0 or ar_in.resolution_drop > 1:
        raise HTTPException(status_code=400, detail="resolution_drop must be in [0, 1]")
    if ar_in.window_hours < 1 or ar_in.window_hours > 24 * 30:
        raise HTTPException(
            status_code=400, detail="window_hours must be in [1, 720]"
        )

    pol = Policy(
        mode=payload.mode,
        auto_min_lift=float(payload.auto_min_lift),
        overrides=dict(payload.overrides),
        auto_rollback=AutoRollback(
            csat_drop=float(ar_in.csat_drop),
            resolution_drop=float(ar_in.resolution_drop),
            window_hours=int(ar_in.window_hours),
            notify_channels=list(ar_in.notify_channels),
        ),
    )
    pol.write_yaml()
    return _policy_to_view(pol)


# ---------- traces (Technical / Traces) ----------
#
# Source: traces/raw/<YYYY-MM-DD>.jsonl files written by runtime/executor/tracing.
# One JSON object per line. Listing pages by offset; trace detail scans newest
# day first so single-trace lookup is fast for recent traffic.


class TraceStageView(BaseModel):
    stage: Optional[str] = None
    technique: str
    variant: str
    duration_ms: float = 0.0
    docs_in: int = 0
    docs_out: int = 0
    response_set: Optional[bool] = None
    routing_model: Optional[str] = None
    error: Optional[str] = None


class HistoryTurn(BaseModel):
    role: str
    content: str


class TraceSummary(BaseModel):
    trace_id: str
    timestamp: str
    request: str
    response: Optional[str] = None
    duration_ms: float = 0.0
    success: bool = False
    error: Optional[str] = None
    agent_version: Optional[str] = None
    n_stages: int = 0
    routing_model: Optional[str] = None  # picked off the route stage
    session_id: Optional[str] = None
    n_turns: int = 1  # history length + this turn


class TracesPage(BaseModel):
    date: str
    available_dates: list[str]
    total_filtered: int
    items: list[TraceSummary]
    has_more: bool


class TraceDetail(TraceSummary):
    stages: list[TraceStageView] = []
    metadata: dict[str, Any] = {}
    history: list[HistoryTurn] = []  # the conversation up to (not including) this turn


@app.get("/traces", response_model=TracesPage)
async def list_traces(
    date: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    success: Optional[bool] = None,
    agent_version: Optional[str] = None,
    q: Optional[str] = None,
) -> TracesPage:
    available = traces_store.available_dates()
    if not available:
        return TracesPage(
            date=date or "",
            available_dates=[],
            total_filtered=0,
            items=[],
            has_more=False,
        )

    chosen_date = date or available[0]
    if chosen_date not in available:
        # client asked for a date with no data — return empty page but keep
        # the chosen_date so the UI keeps the picker honest.
        return TracesPage(
            date=chosen_date,
            available_dates=available,
            total_filtered=0,
            items=[],
            has_more=False,
        )

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    items, total = traces_store.query_traces(
        date=chosen_date,
        success=success,
        agent_version=agent_version,
        q=q,
        limit=limit,
        offset=offset,
    )

    return TracesPage(
        date=chosen_date,
        available_dates=available,
        total_filtered=total,
        items=[TraceSummary(**i) for i in items],
        has_more=offset + limit < total,
    )


@app.get("/traces/stream")
async def stream_traces(request: Request) -> StreamingResponse:
    """Server-sent events of trace summaries. Each event payload is the
    same shape as TraceSummary minus the heavy fields (no full request
    body, no stages array). Subscribers connect, get future writes only;
    backfill is done client-side via GET /traces?limit=...

    A 15s heartbeat keeps proxies from idling the connection."""
    queue = trace_bus.subscribe()

    async def gen() -> Any:
        import asyncio as _asyncio

        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await _asyncio.wait_for(queue.get(), timeout=15.0)
                except _asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"event: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            trace_bus.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering
        },
    )


@app.get("/traces/{trace_id}", response_model=TraceDetail)
async def get_trace(trace_id: str) -> TraceDetail:
    """Find a trace by id (DuckDB indexed lookup across JSONL+Parquet)."""
    detail = traces_store.get_trace(trace_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    stages = [TraceStageView(**s) for s in detail.get("stages", [])]
    history = [HistoryTurn(**h) for h in detail.get("history", [])]
    summary = {k: v for k, v in detail.items() if k not in ("stages", "history", "metadata")}
    return TraceDetail(
        **summary,
        stages=stages,
        metadata=detail.get("metadata") or {},
        history=history,
    )


# ---------- sessions (multi-turn conversations grouped by session_id) ----------


class SessionTurn(BaseModel):
    trace_id: str
    timestamp: str
    request: str
    response: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    agent_version: Optional[str] = None
    duration_ms: float = 0.0


class SessionDetail(BaseModel):
    session_id: str
    n_turns: int
    started_at: str
    last_at: str
    turns: list[SessionTurn] = []


@app.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    """Return every trace that shares a session_id, ordered chronologically.
    A session is the natural unit of conversation in the AHE paper's
    "experience pillar" — multiple /run calls that build on each other's
    history collapse into one thread for the UI to render as a dialog."""
    rows = traces_store.get_session_turns(session_id)
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"session {session_id!r} not found"
        )
    turns = [SessionTurn(**r) for r in rows]
    return SessionDetail(
        session_id=session_id,
        n_turns=len(turns),
        started_at=turns[0].timestamp if turns else "",
        last_at=turns[-1].timestamp if turns else "",
        turns=turns,
    )


# ---------- evals (Technical / Eval suites) ----------
#
# Suites live in evals/suites/<name>.yaml; each lists golden ids and rubric
# specs. Goldens live in evals/golden/<id>.yaml. Run reports live in
# evals/reports/<report_id>.json — both candidate runs (cand_<id>.json) and
# baseline runs (smoke_v0_<ts>.json). The UI sees them as one timeline so
# the operator can watch a suite's pass-rate move over agent versions.


class RubricSpec(BaseModel):
    name: str
    type: str
    params: dict[str, Any] = {}


class GoldenView(BaseModel):
    id: str
    request: str
    expected: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class SuiteSummary(BaseModel):
    name: str
    description: Optional[str] = None
    n_goldens: int
    n_rubrics: int
    aggregation: str = "mean"
    last_run_at: Optional[str] = None
    last_overall_score: Optional[float] = None
    last_pass_rate: Optional[float] = None
    last_agent_version: Optional[str] = None
    n_runs: int = 0
    # author: "human" if pinned by an operator, "agent" if auto-authored by a
    # future proposer. Read from suite YAML's metadata.author, defaults to
    # "human". Forward-looking — the harness has no suite proposer yet.
    author: str = "human"
    # Pass rate of the run before the most recent — used by the UI to label
    # a suite as "regressed" when last_pass_rate < baseline_pass_rate.
    baseline_pass_rate: Optional[float] = None


class SuiteDetail(SuiteSummary):
    goldens: list[GoldenView] = []
    rubrics: list[RubricSpec] = []


class ReportSummary(BaseModel):
    report_id: str
    suite: str
    agent_version: str
    started_at: str
    finished_at: str
    overall_score: float
    pass_rate: float
    n_passed: int
    n_total: int
    is_candidate: bool = False
    candidate_id: Optional[str] = None


class ReportCase(BaseModel):
    golden_id: str
    request: str
    response: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    trace_id: Optional[str] = None
    rubric_results: list[dict[str, Any]] = []


class ReportDetail(ReportSummary):
    cases: list[ReportCase] = []
    per_rubric: dict[str, Any] = {}


_EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    import yaml as _yaml

    if not path.exists():
        return {}
    with path.open() as f:
        return _yaml.safe_load(f) or {}


def _load_golden(golden_id: str) -> Optional[GoldenView]:
    path = _EVALS_DIR / "golden" / f"{golden_id}.yaml"
    if not path.exists():
        return None
    d = _safe_yaml_load(path)
    return GoldenView(
        id=d.get("id", golden_id),
        request=(d.get("input") or {}).get("request", ""),
        expected=d.get("expected") or {},
        metadata=d.get("metadata") or {},
    )


def _list_report_files() -> list[Path]:
    reports_dir = _EVALS_DIR / "reports"
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("*.json"))


def _load_report_meta(path: Path) -> Optional[ReportSummary]:
    """Read just enough of a report to populate the summary row. We still
    open the file (no per-line streaming) but we don't keep the cases."""
    try:
        with path.open() as f:
            d = json.load(f)
    except Exception:
        return None
    summary = d.get("summary") or {}
    cases = d.get("cases") or []
    is_candidate = path.name.startswith("cand_")
    candidate_id: Optional[str] = None
    if is_candidate:
        # filename pattern: cand_<candidate_id>.json (cand_id may itself
        # start with "cand_" — the wrapping prefix is just routing)
        stem = path.stem
        if stem.startswith("cand_"):
            candidate_id = stem[len("cand_") :]
    return ReportSummary(
        report_id=path.stem,
        suite=d.get("suite", ""),
        agent_version=d.get("agent_version", ""),
        started_at=d.get("started_at", ""),
        finished_at=d.get("finished_at", ""),
        overall_score=float(summary.get("overall_score", 0.0)),
        pass_rate=float(summary.get("pass_rate", 0.0)),
        n_passed=int(summary.get("n_passed", 0)),
        n_total=int(summary.get("n_total", len(cases))),
        is_candidate=is_candidate,
        candidate_id=candidate_id,
    )


def _list_suites() -> list[SuiteSummary]:
    suites_dir = _EVALS_DIR / "suites"
    if not suites_dir.exists():
        return []

    # Group runs by suite, sorted newest-first, so we can pick latest + baseline
    # in one pass.
    runs_by_suite: dict[str, list[ReportSummary]] = {}
    for p in _list_report_files():
        meta = _load_report_meta(p)
        if meta is None or not meta.suite:
            continue
        runs_by_suite.setdefault(meta.suite, []).append(meta)
    for runs in runs_by_suite.values():
        runs.sort(key=lambda r: r.finished_at or r.started_at, reverse=True)

    out: list[SuiteSummary] = []
    for sp in sorted(suites_dir.glob("*.yaml")):
        d = _safe_yaml_load(sp)
        name = d.get("suite") or sp.stem
        runs = runs_by_suite.get(name, [])
        latest = runs[0] if runs else None
        baseline = runs[1] if len(runs) >= 2 else None
        author = ((d.get("metadata") or {}).get("author") or "human").lower()
        if author not in ("human", "agent"):
            author = "human"
        out.append(
            SuiteSummary(
                name=name,
                description=d.get("description"),
                n_goldens=len(d.get("goldens") or []),
                n_rubrics=len(d.get("rubrics") or []),
                aggregation=d.get("aggregation", "mean"),
                last_run_at=latest.finished_at if latest else None,
                last_overall_score=latest.overall_score if latest else None,
                last_pass_rate=latest.pass_rate if latest else None,
                last_agent_version=latest.agent_version if latest else None,
                n_runs=len(runs),
                author=author,
                baseline_pass_rate=baseline.pass_rate if baseline else None,
            )
        )
    return out


@app.get("/evals/suites", response_model=list[SuiteSummary])
async def get_suites() -> list[SuiteSummary]:
    return _list_suites()


@app.get("/evals/suites/{name}", response_model=SuiteDetail)
async def get_suite(name: str) -> SuiteDetail:
    if not _REPORT_ID_RE.match(name):
        raise HTTPException(status_code=400, detail=f"invalid suite name {name!r}")
    suite_path = _EVALS_DIR / "suites" / f"{name}.yaml"
    if not suite_path.exists():
        raise HTTPException(status_code=404, detail=f"unknown suite {name!r}")
    d = _safe_yaml_load(suite_path)

    goldens: list[GoldenView] = []
    for gid in d.get("goldens") or []:
        g = _load_golden(str(gid))
        if g is not None:
            goldens.append(g)

    rubrics = [
        RubricSpec(
            name=r.get("name", ""),
            type=r.get("type", ""),
            params=r.get("params") or {},
        )
        for r in (d.get("rubrics") or [])
    ]

    # Reuse the summary builder for the latest run + counts so /suites/:name
    # can render the same headline numbers as the list page.
    summary = SuiteSummary(
        name=d.get("suite") or suite_path.stem,
        description=d.get("description"),
        n_goldens=len(goldens),
        n_rubrics=len(rubrics),
        aggregation=d.get("aggregation", "mean"),
    )
    for s in _list_suites():
        if s.name == summary.name:
            summary = s
            break

    return SuiteDetail(
        **summary.model_dump(),
        goldens=goldens,
        rubrics=rubrics,
    )


def _ts_safe(iso: str) -> str:
    """Mirror runner._write_report's filename rule."""
    return iso.replace(":", "").replace(".", "").replace("-", "")


@app.post("/evals/suites/{name}/run", response_model=ReportSummary)
async def run_suite_endpoint(name: str) -> ReportSummary:
    """Synchronously run a suite and return the new report's summary.

    Stub-LLM smoke suites complete in ~ms; we keep this synchronous so the UI
    can refresh and open the report drawer in one round-trip. If real-LLM
    suites land later this should move to a background job.
    """
    if not _REPORT_ID_RE.match(name):
        raise HTTPException(status_code=400, detail=f"invalid suite name {name!r}")
    suite_path = _EVALS_DIR / "suites" / f"{name}.yaml"
    if not suite_path.exists():
        raise HTTPException(status_code=404, detail=f"unknown suite {name!r}")

    from evals.runners.runner import run_suite

    try:
        report = run_suite(suite_path, agent_path=_AGENT_DIR / "agent.yaml")
    except Exception as e:  # surface load / compile / score errors verbatim
        raise HTTPException(status_code=500, detail=f"run failed: {e}") from e

    report_id = f"{report.suite}_{_ts_safe(report.started_at)}"
    path = _EVALS_DIR / "reports" / f"{report_id}.json"
    meta = _load_report_meta(path)
    if meta is None:
        raise HTTPException(status_code=500, detail="report written but not readable")
    return meta


@app.post("/evals/run_all")
async def run_all_suites_endpoint() -> dict[str, Any]:
    """Run every suite under evals/suites/ in series, return their report ids."""
    suites_dir = _EVALS_DIR / "suites"
    if not suites_dir.exists():
        return {"reports": [], "errors": []}

    from evals.runners.runner import run_suite

    reports: list[ReportSummary] = []
    errors: list[dict[str, str]] = []
    for sp in sorted(suites_dir.glob("*.yaml")):
        try:
            report = run_suite(sp, agent_path=_AGENT_DIR / "agent.yaml")
        except Exception as e:
            errors.append({"suite": sp.stem, "error": str(e)})
            continue
        report_id = f"{report.suite}_{_ts_safe(report.started_at)}"
        meta = _load_report_meta(_EVALS_DIR / "reports" / f"{report_id}.json")
        if meta is not None:
            reports.append(meta)
    return {
        "reports": [r.model_dump() for r in reports],
        "errors": errors,
    }


@app.get("/evals/reports", response_model=list[ReportSummary])
async def list_reports(
    suite: Optional[str] = None,
    limit: int = 50,
    candidate_only: bool = False,
) -> list[ReportSummary]:
    out: list[ReportSummary] = []
    for p in _list_report_files():
        meta = _load_report_meta(p)
        if meta is None:
            continue
        if suite and meta.suite != suite:
            continue
        if candidate_only and not meta.is_candidate:
            continue
        out.append(meta)
    out.sort(key=lambda r: r.finished_at or r.started_at, reverse=True)
    return out[: max(1, min(int(limit), 200))]


@app.get("/evals/reports/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str) -> ReportDetail:
    if not _REPORT_ID_RE.match(report_id):
        raise HTTPException(status_code=400, detail=f"invalid report_id {report_id!r}")
    path = _EVALS_DIR / "reports" / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown report {report_id!r}")
    with path.open() as f:
        d = json.load(f)

    cases = [
        ReportCase(
            golden_id=c.get("golden_id", ""),
            request=c.get("request", ""),
            response=c.get("response"),
            duration_ms=c.get("duration_ms"),
            success=bool(c.get("success", False)),
            error=c.get("error"),
            trace_id=c.get("trace_id"),
            rubric_results=c.get("rubric_results") or [],
        )
        for c in (d.get("cases") or [])
    ]

    meta = _load_report_meta(path)
    if meta is None:
        raise HTTPException(status_code=500, detail="failed to read report meta")

    summary = d.get("summary") or {}
    return ReportDetail(
        **meta.model_dump(),
        cases=cases,
        per_rubric=summary.get("per_rubric") or {},
    )


# ---------- failure mining (P16.1) ----------
#
# Promote a real production trace to a permanent regression test. The agent
# self-improves; the test suite should grow with it — every conversation that
# went sideways becomes a perpetual sentry against regressing into the same
# failure mode. This is the cheapest step toward AutoHarness-aligned evals.
#
# Idempotency: if a golden already exists with metadata.source = "trace:<id>",
# return it instead of writing a duplicate. Suite append is also idempotent.


class PromoteTraceRequest(BaseModel):
    expected_contains: list[str] = []
    category: Optional[str] = None
    difficulty: str = "medium"
    add_to_suite: Optional[str] = None  # e.g. "smoke_v0"


class PromoteTraceResponse(BaseModel):
    golden_id: str
    path: str
    request: str
    expected: dict[str, Any]
    metadata: dict[str, Any]
    suite_appended: Optional[str] = None
    already_existed: bool = False


_GOLDEN_ID_RE = re.compile(r"^golden_(\d+)$")


def _next_golden_id() -> str:
    """Find the largest existing golden number and return next, zero-padded."""
    golden_dir = _EVALS_DIR / "golden"
    max_n = 0
    if golden_dir.exists():
        for p in golden_dir.glob("golden_*.yaml"):
            m = _GOLDEN_ID_RE.match(p.stem)
            if m:
                max_n = max(max_n, int(m.group(1)))
    width = max(3, len(str(max_n + 1)))  # keep golden_001 style for small N
    return f"golden_{(max_n + 1):0{width}d}"


def _find_golden_by_source(source: str) -> Optional[Path]:
    """Idempotency: look for an existing golden whose metadata.source matches."""
    golden_dir = _EVALS_DIR / "golden"
    if not golden_dir.exists():
        return None
    for p in sorted(golden_dir.glob("golden_*.yaml")):
        d = _safe_yaml_load(p)
        if (d.get("metadata") or {}).get("source") == source:
            return p
    return None


def _append_golden_to_suite(suite_name: str, golden_id: str) -> bool:
    """Append golden_id to suite's goldens list. No-op if already there.
    Returns True if appended, False if already present. Raises if suite missing."""
    import yaml

    suite_path = _EVALS_DIR / "suites" / f"{suite_name}.yaml"
    if not suite_path.exists():
        raise FileNotFoundError(f"suite {suite_name!r} not found")
    with suite_path.open() as f:
        data = yaml.safe_load(f) or {}
    goldens = list(data.get("goldens") or [])
    if golden_id in goldens:
        return False
    goldens.append(golden_id)
    data["goldens"] = goldens
    with suite_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return True


@app.post(
    "/evals/goldens/promote-from-trace/{trace_id}",
    response_model=PromoteTraceResponse,
)
async def promote_trace_to_golden(
    trace_id: str,
    payload: Optional[PromoteTraceRequest] = None,
) -> PromoteTraceResponse:
    if not _REPORT_ID_RE.match(trace_id):
        raise HTTPException(status_code=400, detail=f"invalid trace_id {trace_id!r}")

    trace = traces_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"unknown trace {trace_id!r}")

    request_text = (trace.get("request") or "").strip()
    if not request_text:
        raise HTTPException(
            status_code=400, detail="trace has no request text — cannot promote"
        )

    body = payload or PromoteTraceRequest()
    source_marker = f"trace:{trace_id}"

    # Idempotency
    existing = _find_golden_by_source(source_marker)
    if existing is not None:
        d = _safe_yaml_load(existing)
        suite_appended: Optional[str] = None
        if body.add_to_suite:
            try:
                if _append_golden_to_suite(body.add_to_suite, d.get("id", existing.stem)):
                    suite_appended = body.add_to_suite
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
        return PromoteTraceResponse(
            golden_id=d.get("id", existing.stem),
            path=str(existing.relative_to(_EVALS_DIR.parent)),
            request=(d.get("input") or {}).get("request", ""),
            expected=d.get("expected") or {},
            metadata=d.get("metadata") or {},
            suite_appended=suite_appended,
            already_existed=True,
        )

    # New golden
    from datetime import datetime, timezone
    import yaml

    golden_id = _next_golden_id()
    verdict = (
        "fail" if not trace.get("success") or trace.get("error") else "pass"
    )

    expected: dict[str, Any] = {
        "contains": list(body.expected_contains),
        "category": body.category or "from-trace",
    }
    metadata: dict[str, Any] = {
        "source": source_marker,
        "promoted_at": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "trace_verdict": verdict,
        "agent_version": trace.get("agent_version"),
        "difficulty": body.difficulty,
    }
    doc: dict[str, Any] = {
        "id": golden_id,
        "input": {"request": request_text},
        "expected": expected,
        "metadata": metadata,
    }

    golden_path = _EVALS_DIR / "golden" / f"{golden_id}.yaml"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)

    suite_appended = None
    if body.add_to_suite:
        try:
            if _append_golden_to_suite(body.add_to_suite, golden_id):
                suite_appended = body.add_to_suite
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    return PromoteTraceResponse(
        golden_id=golden_id,
        path=str(golden_path.relative_to(_EVALS_DIR.parent)),
        request=request_text,
        expected=expected,
        metadata=metadata,
        suite_appended=suite_appended,
        already_existed=False,
    )


# ---------- agent config (AgentSheet) ----------
#
# Wires the four AgentSheet tabs (Brain / Hands / Channels / Keys) to real
# disk state. Two pieces actually mutate behavior on save: the system prompt
# and the route models. The rest are read-only status panels — claude_code
# binary detection, MCP server availability, webhook channel URL, env-var
# key presence. Anything we can't reach honestly is omitted from the
# response so the UI can fall back to "Coming soon" badges.


class AgentPromptView(BaseModel):
    path: str
    content: str


class AgentModelsView(BaseModel):
    small: Optional[str] = None
    big: Optional[str] = None
    confidence_threshold: Optional[float] = None


class IntegrationStatus(BaseModel):
    name: str
    available: bool
    detail: Optional[str] = None


class AgentKeyStatus(BaseModel):
    name: str
    env_var: str
    set: bool
    mask: Optional[str] = None


class AgentConfigView(BaseModel):
    version: str
    description: Optional[str] = None
    system_prompt: AgentPromptView
    models: AgentModelsView
    integrations: list[IntegrationStatus]
    keys: list[AgentKeyStatus]


_PROJECT_ROOT_AGENT = Path(__file__).resolve().parent.parent
_AGENT_DIR = _PROJECT_ROOT_AGENT / "agent"


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_prompt_path() -> Path:
    """Read generate.yaml's prompt knob, resolve relative to agent/."""
    gen = _read_yaml(_AGENT_DIR / "pipeline" / "generate.yaml")
    rel = (gen.get("knobs") or {}).get("prompt", "../prompts/system.md")
    return (_AGENT_DIR / "pipeline" / rel).resolve()


def _mask_key(value: str) -> str:
    if len(value) <= 10:
        return "•" * len(value)
    return f"{value[:6]}…{value[-4:]}"


@app.get("/agent/config", response_model=AgentConfigView)
async def get_agent_config() -> AgentConfigView:
    import os
    import shutil

    cfg = _state.get("cfg")
    if not cfg:
        raise HTTPException(status_code=503, detail="agent not yet loaded")

    prompt_path = _resolve_prompt_path()
    prompt_content = ""
    if prompt_path.exists():
        with prompt_path.open() as f:
            prompt_content = f.read()

    route = _read_yaml(_AGENT_DIR / "pipeline" / "route.yaml")
    knobs = route.get("knobs") or {}
    models = AgentModelsView(
        small=knobs.get("small"),
        big=knobs.get("big"),
        confidence_threshold=(
            float(knobs["confidence_threshold"])
            if "confidence_threshold" in knobs
            else None
        ),
    )

    claude_bin = shutil.which("claude")
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    if has_anthropic and claude_bin:
        cc_detail = "Anthropic API + claude CLI"
    elif has_anthropic:
        cc_detail = "Anthropic API only"
    elif claude_bin:
        cc_detail = "claude CLI only (no API key)"
    else:
        cc_detail = "no Anthropic key, no claude CLI"

    mcp_server_path = (
        _PROJECT_ROOT_AGENT / "harness" / "introspection" / "mcp_server.py"
    )

    integrations = [
        IntegrationStatus(
            name="Claude Code",
            available=bool(claude_bin or has_anthropic),
            detail=cc_detail,
        ),
        IntegrationStatus(
            name="Introspection MCP",
            available=mcp_server_path.exists(),
            detail=(
                "harness/introspection/mcp_server.py · stdio transport"
                if mcp_server_path.exists()
                else "mcp_server.py missing"
            ),
        ),
        IntegrationStatus(
            name="Webhook channel",
            available=True,
            detail="POST /v1/webhook on the backend",
        ),
    ]

    key_specs = [
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("OpenAI", "OPENAI_API_KEY"),
    ]
    keys: list[AgentKeyStatus] = []
    for name, env_var in key_specs:
        v = os.getenv(env_var)
        keys.append(
            AgentKeyStatus(
                name=name,
                env_var=env_var,
                set=bool(v),
                mask=_mask_key(v) if v else None,
            )
        )

    return AgentConfigView(
        version=cfg.version,
        description=cfg.description,
        system_prompt=AgentPromptView(
            path=str(prompt_path.relative_to(_PROJECT_ROOT_AGENT)),
            content=prompt_content,
        ),
        models=models,
        integrations=integrations,
        keys=keys,
    )


class PromptUpdateRequest(BaseModel):
    content: str


class ManualEditResult(BaseModel):
    """Returned from any manual-edit PUT — surfaces the resulting Lesson +
    new version so the UI can route the operator to Evolution or inform
    them what happened, AutoHarness-style: every edit is versioned."""

    new_version: str
    lesson_id: str
    parent_version: str


class PromptUpdateResponse(AgentPromptView, ManualEditResult):
    pass


@app.put("/agent/prompt", response_model=PromptUpdateResponse)
async def update_prompt(payload: PromptUpdateRequest) -> PromptUpdateResponse:
    """Apply a manual prompt edit through the same snapshot+ledger+lesson
    machinery a candidate-driven promotion uses. AutoHarness paper treats
    every change to the editable surface as a versioned event regardless
    of source — see harness.executor.promote.record_manual_change."""
    from harness.executor.promote import record_manual_change

    prompt_path = _resolve_prompt_path()
    if not str(prompt_path).startswith(str(_AGENT_DIR.resolve())):
        raise HTTPException(status_code=400, detail="resolved prompt path escapes agent/")

    rel_path = prompt_path.relative_to(_PROJECT_ROOT_AGENT)

    def _write_prompt() -> None:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_path.open("w") as f:
            f.write(payload.content)

    lesson = record_manual_change(
        _write_prompt,
        kind="prompt",
        summary="Updated system prompt",
        mutations_desc=[f"{rel_path} (manual)"],
        voice="I updated my system prompt directly.",
    )

    return PromptUpdateResponse(
        path=str(rel_path),
        content=payload.content,
        new_version=lesson.version or "",
        lesson_id=lesson.id,
        parent_version=lesson.parent_version,
    )


class RouteUpdateRequest(BaseModel):
    small: Optional[str] = None
    big: Optional[str] = None
    confidence_threshold: Optional[float] = None


class RouteUpdateResponse(AgentModelsView, ManualEditResult):
    pass


@app.put("/agent/route", response_model=RouteUpdateResponse)
async def update_route(payload: RouteUpdateRequest) -> RouteUpdateResponse:
    """Apply a manual route edit (model or threshold change) through the
    same versioning + ledger machinery as a candidate-driven promotion."""
    import yaml

    from harness.executor.promote import record_manual_change

    route_path = _AGENT_DIR / "pipeline" / "route.yaml"
    if not route_path.exists():
        raise HTTPException(status_code=404, detail="route.yaml missing")

    if payload.confidence_threshold is not None:
        if not 0.0 <= payload.confidence_threshold <= 1.0:
            raise HTTPException(
                status_code=400, detail="confidence_threshold must be in [0, 1]"
            )

    changes: list[str] = []

    def _write_route() -> None:
        with route_path.open() as f:
            doc = yaml.safe_load(f) or {}
        knobs = doc.setdefault("knobs", {})
        if payload.small is not None and knobs.get("small") != payload.small:
            knobs["small"] = payload.small
            changes.append(f"agent/pipeline/route.yaml:knobs.small={payload.small} (manual)")
        if payload.big is not None and knobs.get("big") != payload.big:
            knobs["big"] = payload.big
            changes.append(f"agent/pipeline/route.yaml:knobs.big={payload.big} (manual)")
        if (
            payload.confidence_threshold is not None
            and knobs.get("confidence_threshold") != payload.confidence_threshold
        ):
            knobs["confidence_threshold"] = float(payload.confidence_threshold)
            changes.append(
                f"agent/pipeline/route.yaml:knobs.confidence_threshold={payload.confidence_threshold} (manual)"
            )
        with route_path.open("w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)

    # Prepare summary text
    summary = "Updated routing knobs"
    if payload.small is not None and payload.big is None and payload.confidence_threshold is None:
        summary = f"Switched fast model to {payload.small}"
    elif payload.big is not None and payload.small is None and payload.confidence_threshold is None:
        summary = f"Switched escalation model to {payload.big}"

    lesson = record_manual_change(
        _write_route,
        kind="router",
        summary=summary,
        mutations_desc=changes,  # populated by _write_route
        voice="I changed how I route requests between models.",
    )

    # Re-read route to return current state
    with route_path.open() as f:
        doc = yaml.safe_load(f) or {}
    knobs = doc.get("knobs") or {}
    return RouteUpdateResponse(
        small=knobs.get("small"),
        big=knobs.get("big"),
        confidence_threshold=knobs.get("confidence_threshold"),
        new_version=lesson.version or "",
        lesson_id=lesson.id,
        parent_version=lesson.parent_version,
    )


@app.post("/run", response_model=RunResponse)
async def run(payload: RunRequest) -> RunResponse:
    import uuid

    executor: Optional[PipelineExecutor] = _state.get("executor")
    if not executor:
        raise HTTPException(status_code=503, detail="agent not yet loaded")

    history = [Message(role=m.role, content=m.content) for m in (payload.history or [])]
    # If the caller didn't supply a session_id, mint one. The trace persists
    # it so the UI can group multi-turn calls into a single conversation.
    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    _, rec = executor.run(payload.request, history=history, session_id=session_id)
    trace_id = write_trace(rec)

    return RunResponse(
        response=rec.response,
        trace_id=trace_id,
        duration_ms=rec.duration_ms,
        success=rec.success,
        error=rec.error,
        agent_version=rec.agent_version,
        stages=[
            StageOutcome(
                stage=s.stage,
                technique=s.technique,
                variant=s.variant,
                duration_ms=s.duration_ms,
                docs_in=s.docs_in,
                docs_out=s.docs_out,
                routing_model=s.routing_model,
                error=s.error,
            )
            for s in rec.stages
        ],
    )


# ---------------------------------------------------------------------------
# P15.3.2 — Router config + decide endpoints
# ---------------------------------------------------------------------------


class RouterConfigView(BaseModel):
    """Metadata snapshot of the current router_config. Cold-start safe."""

    version: Optional[int]
    k: int
    model_count: int
    cost_weight: float
    embedder_model: str
    embedding_dim: int
    last_fit_at: Optional[str]
    fitted_from: Optional[dict] = None
    cold_start: bool


class RouterDecideRequest(BaseModel):
    prompt: str
    allowed_models: Optional[list[str]] = None
    cost_weight_override: Optional[float] = None


class RouterDecideResponse(BaseModel):
    selected_model: str
    expected_error: float
    cost_adjusted_score: float
    all_scores: dict[str, float]
    cluster_id: int
    cluster_probabilities: list[float]
    reasoning: Optional[str] = None
    cold_start: bool = False


# Process-singleton embedder — lazy. P15.3.8 will replace this with a proper
# embedder_pool with warmup; for P15.3.2 a one-liner is enough.
_router_embedder: Any = None


def _get_router_embedder() -> Any:
    """Lazy PromptEmbedder. First call eats ~1-3s for the MiniLM load."""
    global _router_embedder
    if _router_embedder is not None:
        return _router_embedder
    from router.core.embeddings import PromptEmbedder, SentenceTransformerProvider

    _router_embedder = PromptEmbedder(SentenceTransformerProvider())
    return _router_embedder


@app.get("/router/config", response_model=RouterConfigView)
async def get_router_config() -> RouterConfigView:
    """Return the current router_config metadata.

    Cold-start (no fitted config yet): returns 200 with cold_start=True and
    empty/default fields. UI uses this to render the empty state.
    """
    from router.config_io import (
        cold_start_metadata,
        build_view_metadata,
        load_current_config_payload,
    )
    from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError

    try:
        payload = load_current_config_payload()
    except RouterConfigNotFoundError:
        return RouterConfigView(**cold_start_metadata())
    except RouterConfigInvalidError as e:
        raise HTTPException(status_code=500, detail=f"router_config_invalid: {e}")

    return RouterConfigView(**build_view_metadata(payload))


@app.post("/router/decide", response_model=RouterDecideResponse)
async def post_router_decide(req: RouterDecideRequest) -> RouterDecideResponse:
    """Score a prompt against the current router_config without executing.

    503 router_cold_start when no config exists. Caller should fall back
    to agent.models.default (or knobs.small) — see P15.3.8 for the engine
    wiring that does this automatically.
    """
    from router.config_io import load_current_config
    from router.errors import (
        RouterColdStartError,
        RouterConfigInvalidError,
        RouterConfigNotFoundError,
    )
    from router.uniroute import UniRouteRouter

    try:
        assigner, registry, lam = load_current_config()
    except RouterConfigNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="router_cold_start: no fitted config; wait for the harness to fit one",
        )
    except RouterConfigInvalidError as e:
        raise HTTPException(status_code=500, detail=f"router_config_invalid: {e}")

    embedder = _get_router_embedder()
    try:
        router = UniRouteRouter(embedder, assigner, registry, cost_weight=lam)
    except RouterColdStartError as e:
        raise HTTPException(status_code=503, detail=f"router_cold_start: {e}")

    try:
        decision = router.route(
            req.prompt,
            available_models=req.allowed_models,
            cost_weight_override=req.cost_weight_override,
        )
    except ValueError as e:
        # No models available after filtering — bad allowed_models list.
        raise HTTPException(status_code=400, detail=str(e))

    return RouterDecideResponse(
        selected_model=decision.selected_model,
        expected_error=decision.expected_error,
        cost_adjusted_score=decision.cost_adjusted_score,
        all_scores=decision.all_scores,
        cluster_id=decision.cluster_id,
        cluster_probabilities=decision.cluster_probabilities.tolist(),
        reasoning=decision.reasoning,
        cold_start=False,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("runtime.server:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
