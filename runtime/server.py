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
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from runtime.compiler.builder import compile_agent
from runtime.compiler.loader import load_agent
from runtime.dotenv import load_env
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
    routing_decision: Optional[dict] = None  # P15.3.8 — UniRoute decision dict
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

    loaded = load_env()
    if loaded:
        logger.info(".env loaded: %d key(s)", len(loaded))

    # P2.0 — ensure multi-agent registry exists. On first run this
    # migrates the legacy ``agent/`` dir to ``agents/_default/`` and
    # writes ``agents/registry.json`` pointing at it. P2.1 also moves
    # flat ledger/* and traces/* dirs into <root>/_default/<kind>/.
    # Idempotent.
    from runtime.agent_context import set_active as _set_active_agent
    from runtime.agents.registry import ensure_bootstrapped, get_registry
    ensure_bootstrapped()
    reg = get_registry()
    if reg.active:
        _set_active_agent(reg.active)
        logger.info("active agent: %s", reg.active)

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

    # P16.2 — real cost & CSAT signals.
    # avg_cost_usd: char-based estimate via runtime.cost (auto-swap to
    #   real Anthropic SDK usage when P1.9 lands; same fields).
    # csat: aggregate of POST /traces/{id}/feedback rows.
    from runtime.store import feedback as feedback_store
    csat_value = feedback_store.csat_for_window(window_days=7)

    return MetricsOverview(
        today_count=trace_metrics["today_count"],
        active_5min=trace_metrics["active_5min"],
        pending_review=pending,
        trust_score=trust_score,
        trust_score_delta_30d=trust_score_delta_30d,
        trust_history_30d=history,
        resolution_rate=trace_metrics["resolution_rate"],
        avg_latency_ms=trace_metrics["avg_latency_ms"],
        avg_cost_usd=trace_metrics.get("avg_cost_usd"),
        csat=csat_value,
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
    # P16.2 — cost telemetry. Optional because pre-P16.2 traces lack these.
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    # P16.3 — flag state. Resolved against traces/flagged/<date>.jsonl
    # side-table at response time. False when no flag row exists (or
    # the latest row is an unflag).
    flagged: bool = False


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

    # P16.3 — stamp flag state. One side-table read per list request;
    # cheap compared to refetching per-row.
    from runtime.store import flags as flags_store
    flagged_set = flags_store.flagged_trace_ids()
    rows: list[TraceSummary] = []
    for i in items:
        i = dict(i)
        i["flagged"] = i["trace_id"] in flagged_set
        rows.append(TraceSummary(**i))

    return TracesPage(
        date=chosen_date,
        available_dates=available,
        total_filtered=total,
        items=rows,
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
    # P16.3 — stamp flag state.
    from runtime.store import flags as flags_store
    summary["flagged"] = flags_store.is_flagged(trace_id)
    return TraceDetail(
        **summary,
        stages=stages,
        metadata=detail.get("metadata") or {},
        history=history,
    )


# ---------- trace flag (P16.3 — operator + auto-flag) ----------


class TraceFlagRequest(BaseModel):
    reason: Optional[str] = None


class TraceFlagEntry(BaseModel):
    trace_id: str
    reason: Optional[str] = None
    source: str  # "manual" | "csat_low" | "latency_outlier" | "error" | "unflag"
    at: str


class TraceFlagResponse(BaseModel):
    trace_id: str
    flagged: bool
    last_row: TraceFlagEntry


@app.post(
    "/traces/{trace_id}/flag",
    response_model=TraceFlagResponse,
    status_code=201,
)
async def post_trace_flag(
    trace_id: str,
    req: TraceFlagRequest,
) -> TraceFlagResponse:
    """Mark a trace as flagged (manual operator action).

    Side-table at `traces/flagged/<date>.jsonl`. Multiple flag rows per
    trace are kept; the latest row wins for "is flagged?" queries.
    404 when the trace doesn't exist.
    """
    from runtime.store import flags as flags_store

    trace = traces_store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")

    row = flags_store.write_flag(
        trace_id, reason=req.reason, source="manual"
    )
    return TraceFlagResponse(
        trace_id=trace_id,
        flagged=True,
        last_row=TraceFlagEntry(**row),
    )


@app.delete(
    "/traces/{trace_id}/flag",
    response_model=TraceFlagResponse,
    status_code=200,
)
async def delete_trace_flag(trace_id: str) -> TraceFlagResponse:
    """Clear the flag on a trace. Appends an `unflag` row (history preserved).

    Idempotent — unflagging an unflagged trace just appends another
    unflag row.
    """
    from runtime.store import flags as flags_store

    trace = traces_store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")

    row = flags_store.write_flag(trace_id, source="unflag")
    return TraceFlagResponse(
        trace_id=trace_id,
        flagged=False,
        last_row=TraceFlagEntry(**row),
    )


@app.get(
    "/traces/{trace_id}/flag",
    response_model=list[TraceFlagEntry],
)
async def list_trace_flag(trace_id: str) -> list[TraceFlagEntry]:
    """All flag rows for one trace (chronological).

    Operators see this in the Trace drawer's "Flag history" panel.
    The aggregate "is flagged?" is also surfaced as the boolean
    `flagged` field on every TraceSummary so the list-view filter pill
    works without per-row round-trips.
    """
    from runtime.store import flags as flags_store
    rows = flags_store.list_flag_rows_for_trace(trace_id)
    return [TraceFlagEntry(**r) for r in rows]


# ---------- trace feedback (P16.2 — CSAT signal) ----------


class TraceFeedbackRequest(BaseModel):
    score: int  # 1..5
    comment: Optional[str] = None


class TraceFeedbackEntry(BaseModel):
    trace_id: str
    score: int
    comment: Optional[str] = None
    at: str


class TraceFeedbackResponse(BaseModel):
    trace_id: str
    score: int
    comment: Optional[str] = None
    at: str
    n_total: int  # total feedback rows for this trace after writing


@app.post(
    "/traces/{trace_id}/feedback",
    response_model=TraceFeedbackResponse,
    status_code=201,
)
async def post_trace_feedback(
    trace_id: str,
    req: TraceFeedbackRequest,
) -> TraceFeedbackResponse:
    """Append a CSAT row for `trace_id`. 1-5 score; comment optional.

    Backed by `traces/feedback/<date>.jsonl` (P16.2). Aggregated by
    `runtime.store.feedback.csat_for_window` and exposed via
    `/metrics/overview.csat`. Side-table by design — trace JSONL files
    aren't rewritten.

    404 when the trace doesn't exist (so the UI can't silently spam
    feedback rows for ghost IDs).
    """
    from runtime.store import feedback as feedback_store

    trace = traces_store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")

    try:
        row = feedback_store.write_feedback(
            trace_id, score=req.score, comment=req.comment
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # P16.3 — auto-flag rule. Low CSAT (≤ 2) is a strong signal that the
    # trace deserves human review. We write an auto-flag row so the
    # "Flagged" filter pill catches it and the feedback_signals mining
    # adapter can pick it up later.
    if req.score <= _AUTO_FLAG_CSAT_THRESHOLD:
        from runtime.store import flags as flags_store
        # Idempotent: only auto-flag if the trace isn't already flagged.
        # Avoids writing a duplicate auto-flag row when the operator
        # corrects an earlier rating.
        if not flags_store.is_flagged(trace_id):
            flags_store.write_flag(
                trace_id,
                reason=f"CSAT score {req.score}/5 (auto-flag)",
                source="csat_low",
            )

    n_total = len(feedback_store.list_feedback_for_trace(trace_id))
    return TraceFeedbackResponse(**row, n_total=n_total)


# P16.3 — auto-flag rule threshold. Scores at or below this fire an
# automatic flag with source="csat_low".
_AUTO_FLAG_CSAT_THRESHOLD = 2


@app.get(
    "/traces/{trace_id}/feedback",
    response_model=list[TraceFeedbackEntry],
)
async def list_trace_feedback(trace_id: str) -> list[TraceFeedbackEntry]:
    """All feedback rows for one trace (most recent last)."""
    from runtime.store import feedback as feedback_store

    rows = feedback_store.list_feedback_for_trace(trace_id)
    return [TraceFeedbackEntry(**r) for r in rows]


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
                routing_decision=s.routing_decision,
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


# P15.3.10 — UI Router config feeds.


class RouterRuleHistoryEntry(BaseModel):
    when: str
    what: str
    lesson_id: Optional[str] = None


class RouterRuleView(BaseModel):
    """Synthesized rule row for the UI Router config screen.

    P15.3.10 ships a single row representing the UniRoute default. Manual
    rules are deferred — when the rules engine lands, this endpoint
    returns the real list and the UI lights up the rest of its chrome.
    """

    id: str
    name: str
    when: str
    then: str
    share: float
    cost: float
    auth: str   # "agent" | "human"
    enabled: bool
    isDefault: bool = False
    rationale: str
    history: list[RouterRuleHistoryEntry]
    samples: list[str]


class RouterCandidateView(BaseModel):
    """Pending router_config Lesson awaiting review."""

    lesson_id: str
    version: int
    title: str
    summary: str
    delta: dict[str, Any]
    created_at: Optional[str] = None
    review_link: str


class RouterHealthView(BaseModel):
    """Mirrors RouterHealth.to_dict() — same shape as the MCP tool."""

    cold_start: bool
    version: Optional[int] = None
    k: Optional[int] = None
    model_count: Optional[int] = None
    cost_weight: Optional[float] = None
    last_fit_at: Optional[str] = None
    last_fit_age_hours: Optional[float] = None
    trace_count_since_last_fit: int = 0
    drift_score: Optional[float] = None
    drift_baseline: Optional[float] = None
    needs_reclustering: Optional[bool] = None
    current_avg_error: Optional[float] = None
    current_win_rate: Optional[float] = None
    cluster_distribution: Optional[dict[str, int]] = None
    fitted_from: Optional[dict] = None
    sample_size: int = 0


@app.get("/router/rules", response_model=list[RouterRuleView])
async def list_router_rules() -> list[RouterRuleView]:
    """Synthesized rule list. v1 returns one row representing UniRoute."""
    from router.config_io import load_current_config_payload
    from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError
    from ledger.writer import read_lessons

    try:
        payload = load_current_config_payload()
        cold_start = False
    except (RouterConfigNotFoundError, RouterConfigInvalidError):
        payload = None
        cold_start = True

    history = _build_router_history()
    rule = _build_uniroute_default_rule(payload=payload, cold_start=cold_start, history=history)
    return [rule]


@app.get("/router/candidates", response_model=list[RouterCandidateView])
async def list_router_candidates() -> list[RouterCandidateView]:
    """Pending router_config Lessons awaiting human review."""
    from ledger.writer import read_lessons

    out: list[RouterCandidateView] = []
    for lesson in read_lessons():
        if lesson.kind != "router_config" or lesson.status != "awaiting_review":
            continue
        try:
            version = int(lesson.version)
        except (TypeError, ValueError):
            version = 0
        out.append(
            RouterCandidateView(
                lesson_id=lesson.id,
                version=version,
                title=lesson.title or "router_config candidate",
                summary=lesson.summary or "",
                delta=lesson.delta or {},
                created_at=lesson.promoted_at,
                review_link=f"/review/{lesson.id}",
            )
        )
    out.sort(key=lambda c: (c.created_at or ""), reverse=True)
    return out


@app.get("/router/health", response_model=RouterHealthView)
async def get_router_health() -> RouterHealthView:
    """Same payload as the MCP router_health_check tool, exposed over HTTP
    so the UI can render the same snapshot Claude Code reads."""
    from router.feedback.health import compute_router_health

    h = compute_router_health().to_dict()
    # Pydantic's dict[str, int] requires str keys; cluster_distribution comes
    # in with int keys.
    cluster_dist = h.get("cluster_distribution")
    if cluster_dist:
        h["cluster_distribution"] = {str(k): int(v) for k, v in cluster_dist.items()}
    h.pop("metadata", None)
    return RouterHealthView(**h)


def _build_router_history() -> list[RouterRuleHistoryEntry]:
    """Pull recent router_config Lessons as drawer history rows."""
    from ledger.writer import read_lessons

    items: list[RouterRuleHistoryEntry] = []
    for lesson in read_lessons():
        if lesson.kind != "router_config":
            continue
        when = lesson.promoted_at or "earlier"
        what = lesson.voice or lesson.title or "router_config change"
        items.append(
            RouterRuleHistoryEntry(when=when, what=what, lesson_id=lesson.id)
        )
    items.sort(key=lambda h: (h.when or ""), reverse=True)
    return items


def _build_uniroute_default_rule(
    *,
    payload: Optional[dict],
    cold_start: bool,
    history: list[RouterRuleHistoryEntry],
) -> RouterRuleView:
    if cold_start or payload is None:
        return RouterRuleView(
            id="uniroute_default",
            name="UniRoute (default)",
            when="default",
            then="uniroute (cold-start)",
            share=1.0,
            cost=0.0,
            auth="agent",
            enabled=True,
            isDefault=True,
            rationale=(
                "Cold-start: no router_config has been fitted yet. "
                "Claude Code will propose one once enough traffic accumulates."
            ),
            history=history,
            samples=[],
        )

    k = int(payload.get("k", 0))
    n_models = len(payload.get("model_psi") or {})
    silhouette = float((payload.get("metadata") or {}).get("silhouette", 0.0))
    return RouterRuleView(
        id="uniroute_default",
        name="UniRoute (default)",
        when="default",
        then=f"uniroute (K={k}, models={n_models})",
        share=1.0,
        cost=float(payload.get("cost_weight", 0.0)),
        auth="agent",
        enabled=True,
        isDefault=True,
        rationale=(
            f"Trained router. K={k} clusters across {n_models} models. "
            f"Silhouette={silhouette:.3f}. Picks per request based on "
            "prompt embedding cluster + per-model expected error."
        ),
        history=history,
        samples=[],
    )


# P15.3.8 — manual router_config edit through the AHE pipeline.


class RouterConfigUpdate(BaseModel):
    """Manual operator update body for PUT /router/config."""

    cost_weight: Optional[float] = None


class RouterConfigUpdateResponse(BaseModel):
    version: int
    lesson_id: str
    config: RouterConfigView


@app.put("/router/config", response_model=RouterConfigUpdateResponse)
async def put_router_config(req: RouterConfigUpdate) -> RouterConfigUpdateResponse:
    """Manual router_config edit (e.g., λ override).

    AHE-aligned: routes through ``record_manual_router_change`` so the edit
    surfaces as ``Lesson(kind="router_config", proposal_source="human")``,
    bumps the router_config version, and rolls back via the existing
    ``/versions/{v}/rollback`` machinery — same as any other promotion.

    Cold-start (no current config) → 409. Operators can't manually edit a
    config that doesn't exist; wait for the harness to fit one.
    """
    from harness.executor.promote import record_manual_router_change
    from router.config_io import build_view_metadata, load_current_config_payload
    from router.errors import RouterConfigInvalidError, RouterConfigNotFoundError

    try:
        current = load_current_config_payload()
    except RouterConfigNotFoundError:
        raise HTTPException(
            status_code=409,
            detail=(
                "cannot edit a config that doesn't exist yet — "
                "wait for the harness to fit one"
            ),
        )
    except RouterConfigInvalidError as e:
        raise HTTPException(status_code=500, detail=f"router_config_invalid: {e}")

    if req.cost_weight is None:
        raise HTTPException(status_code=400, detail="no fields to update")

    # Preserve centroids by carrying them on the new payload — apply_router_candidate
    # extracts and writes the sidecar .npz.
    import numpy as np

    from router.config_io import _centroids_path, _vd

    old_npz = _centroids_path(int(current["version"]), versions_dir=_vd(None))
    centroids_inline = None
    if old_npz.exists():
        centroids_inline = np.load(old_npz)["centroids"].tolist()

    new_payload = dict(current)
    new_payload["version"] = int(current["version"]) + 1
    new_payload["cost_weight"] = float(req.cost_weight)
    if centroids_inline is not None:
        new_payload["centroids"] = centroids_inline
    new_payload["created_at"] = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    new_payload["metadata"] = {
        **(current.get("metadata") or {}),
        "previous_cost_weight": float(current.get("cost_weight", 0.0)),
        "manual_edit_phase": "P15.3.8",
        "stage": "manual_edit",
    }

    summary = f"λ {current.get('cost_weight', 0.0)} → {req.cost_weight}"
    voice = f"I tweaked my routing weights manually — λ is now {req.cost_weight}."
    lesson = record_manual_router_change(
        new_payload, summary=summary, voice=voice
    )

    return RouterConfigUpdateResponse(
        version=int(new_payload["version"]),
        lesson_id=lesson.id,
        config=RouterConfigView(**build_view_metadata(new_payload)),
    )


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


# ---------------------------------------------------------------------------
# P15.4.2 — Datasets endpoints
# ---------------------------------------------------------------------------


class DatasetView(BaseModel):
    """Grid-row shape for the UI Datasets screen. Mirrors `interface Dataset`
    in ui/src/screens/Technical.tsx."""

    id: str
    name: str
    desc: str
    size: int
    source: str
    sourceType: str
    fresh: str
    use: list[str]
    owner: str
    growing: bool


class DatasetSampleView(BaseModel):
    id: str
    preview: str
    tag: Optional[str] = None


class DatasetHistoryEntry(BaseModel):
    when: str
    what: str


class DatasetDetail(DatasetView):
    samples: list[DatasetSampleView]
    history: list[DatasetHistoryEntry]


class DatasetHealth(BaseModel):
    name: str
    size: int
    cluster_distribution: dict[str, int]
    coverage_gap_score: Optional[float] = None
    last_curation_at: Optional[str] = None


class DatasetCreateRequest(BaseModel):
    name: str
    desc: str = ""
    source: str = "manual"
    sourceType: str = "manual"
    use: list[str] = ["Eval"]
    owner: str = "human"
    growing: bool = False


class DatasetUpdateRequest(BaseModel):
    desc: Optional[str] = None
    use: Optional[list[str]] = None
    growing: Optional[bool] = None


_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PREVIEW_MAX = 200


def _relative_time(iso: Optional[str]) -> str:
    """Render an ISO timestamp as a UI-friendly relative time ("3d", "6h", "just now").

    Returns "—" when iso is empty/None, or when iso is the 1970-01-01 epoch
    sentinel used by the migration script for byte-identical reruns. Best-effort
    parser; falls back to the raw string when parsing fails.
    """
    if not iso:
        return "—"
    try:
        ts = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return iso
    if dt.year < 2000:
        return "—"
    now = datetime.now(timezone.utc)
    delta = now - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    minutes = int(secs // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = int(secs // 3600)
    if hours < 24:
        return f"{hours}h"
    days = int(secs // 86400)
    if days < 30:
        return f"{days}d"
    months = int(days // 30)
    if months < 12:
        return f"{months}mo"
    years = int(days // 365)
    return f"{years}y"


def _registry_entry_fresh(entry: dict) -> str:
    return _relative_time(entry.get("updated_at"))


def _registry_to_view(name: str, entry: dict) -> DatasetView:
    """Project a `_registry.json` entry into the UI's DatasetView shape."""
    return DatasetView(
        id=name,
        name=name,
        desc=entry.get("desc", ""),
        size=int(entry.get("size", 0)),
        source=entry.get("source", entry.get("sourceType", "manual")),
        sourceType=entry.get("sourceType", "manual"),
        fresh=_registry_entry_fresh(entry),
        use=list(entry.get("use", ["Eval"])),
        owner=entry.get("owner", "human"),
        growing=bool(entry.get("growing", False)),
    )


def _dataset_to_detail(dataset, entry: dict) -> DatasetDetail:
    """Build the drawer payload (DatasetView + samples + history)."""
    view = _registry_to_view(dataset.metadata.name, entry)
    samples = [
        DatasetSampleView(
            id=s.id,
            preview=s.prompt if len(s.prompt) <= _PREVIEW_MAX else s.prompt[: _PREVIEW_MAX - 1] + "…",
            tag=s.tag,
        )
        for s in dataset.samples[:50]
    ]
    history = [
        DatasetHistoryEntry(
            when=_relative_time(h.get("when")),
            what=str(h.get("what", "")),
        )
        for h in dataset.history
    ]
    return DatasetDetail(
        **view.model_dump(),
        samples=samples,
        history=history,
    )


@app.get("/datasets", response_model=list[DatasetView])
async def list_datasets_endpoint(
    use: Optional[str] = None,
    owner: Optional[str] = None,
    sourceType: Optional[str] = None,
) -> list[DatasetView]:
    """List all (non-soft-deleted) datasets in the registry.

    Cold-start (no migration run yet): returns []. Filters are AND-combined
    when present.
    """
    from router.data.dataset_registry import _load_registry, _vd

    registry = _load_registry(datasets_dir=_vd(None))
    out: list[DatasetView] = []
    for ds_name, entry in (registry.get("datasets") or {}).items():
        if entry.get("deleted"):
            continue
        if use is not None and use not in (entry.get("use") or []):
            continue
        if owner is not None and entry.get("owner") != owner:
            continue
        if sourceType is not None and entry.get("sourceType") != sourceType:
            continue
        out.append(_registry_to_view(ds_name, entry))
    out.sort(key=lambda d: d.name)
    return out


@app.get("/datasets/{name}", response_model=DatasetDetail)
async def get_dataset_endpoint(name: str) -> DatasetDetail:
    """Full dataset payload + samples + history. 404 when unknown."""
    from router.data.dataset_io import load_current
    from router.data.dataset_registry import _load_registry, _vd
    from router.errors import DatasetInvalidError, DatasetNotFoundError

    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None or entry.get("deleted"):
        raise HTTPException(status_code=404, detail=f"dataset_not_found: {name}")

    try:
        dataset = load_current(name)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DatasetInvalidError as e:
        raise HTTPException(status_code=500, detail=f"dataset_invalid: {e}")

    return _dataset_to_detail(dataset, entry)


@app.get("/datasets/{name}/health", response_model=DatasetHealth)
async def get_dataset_health_endpoint(name: str) -> DatasetHealth:
    """Coverage report for the dataset. cluster_distribution is empty when
    router_config is cold-start (no centroids to assign against yet)."""
    from router.data.dataset_io import load_current
    from router.data.dataset_registry import _load_registry, _vd
    from router.errors import DatasetInvalidError, DatasetNotFoundError
    import numpy as np

    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None or entry.get("deleted"):
        raise HTTPException(status_code=404, detail=f"dataset_not_found: {name}")

    try:
        dataset = load_current(name)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DatasetInvalidError as e:
        raise HTTPException(status_code=500, detail=f"dataset_invalid: {e}")

    cluster_distribution: dict[str, int] = {}
    try:
        from router.config_io import load_current_config
        from router.errors import (
            RouterConfigInvalidError,
            RouterConfigNotFoundError,
        )

        try:
            assigner, _registry, _lam = load_current_config()
        except (RouterConfigNotFoundError, RouterConfigInvalidError):
            assigner = None

        if assigner is not None and dataset.samples:
            for s in dataset.samples:
                if not s.embedding:
                    continue
                vec = np.asarray(s.embedding, dtype=float)
                cid = int(assigner.assign(vec).cluster_id)
                key = str(cid)
                cluster_distribution[key] = cluster_distribution.get(key, 0) + 1
    except Exception:
        # Coverage is best-effort; never 5xx because of cluster math.
        cluster_distribution = {}

    return DatasetHealth(
        name=name,
        size=dataset.size(),
        cluster_distribution=cluster_distribution,
        coverage_gap_score=None,
        last_curation_at=entry.get("updated_at"),
    )


def _build_dataset_payload(
    req: DatasetCreateRequest,
    *,
    version: int = 1,
) -> dict:
    """Build a fresh dataset payload for a manual create."""
    now = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return {
        "version": version,
        "name": req.name,
        "desc": req.desc,
        "source": req.source,
        "sourceType": req.sourceType,
        "use": list(req.use),
        "owner": req.owner,
        "growing": bool(req.growing),
        "created_at": now,
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "samples": [],
        "history": [{"when": now, "what": f"You created this dataset ({req.sourceType})."}],
        "metadata": {"phase": "P15.4.2", "stage": "manual_create"},
    }


@app.post("/datasets", response_model=DatasetView, status_code=201)
async def create_dataset_endpoint(req: DatasetCreateRequest) -> DatasetView:
    """Manual dataset creation. Goes through ``record_manual_change(kind="dataset")``
    so the action surfaces as a Lesson + ledger entry, same as any other
    edit to the editable surface."""
    from harness.executor.promote import record_manual_dataset_change
    from router.data.dataset_io import save_dataset
    from router.data.dataset_registry import _load_registry, _vd

    if not _NAME_PATTERN.match(req.name):
        raise HTTPException(
            status_code=400,
            detail="name must match ^[a-z0-9][a-z0-9_-]{0,63}$",
        )
    valid_uses = {"Eval", "Distill"}
    if not req.use or any(u not in valid_uses for u in req.use):
        raise HTTPException(
            status_code=400,
            detail=f"use must be a non-empty subset of {sorted(valid_uses)}",
        )
    if req.owner not in {"agent", "human"}:
        raise HTTPException(status_code=400, detail="owner must be 'agent' or 'human'")
    if req.sourceType not in {"auto", "manual"}:
        raise HTTPException(status_code=400, detail="sourceType must be 'auto' or 'manual'")

    registry = _load_registry(datasets_dir=_vd(None))
    existing = (registry.get("datasets") or {}).get(req.name)
    if existing and not existing.get("deleted"):
        raise HTTPException(status_code=409, detail=f"dataset_already_exists: {req.name}")

    payload = _build_dataset_payload(req)

    def _writer() -> None:
        save_dataset(payload)

    use_label = " + ".join(req.use)
    record_manual_dataset_change(
        name=req.name,
        new_version=1,
        summary=f"Created dataset '{req.name}' ({use_label})",
        apply_edit=_writer,
        voice=f"I created the '{req.name}' dataset for {use_label.lower()} use.",
    )

    # Re-read the registry entry so `fresh` reflects the post-write timestamp.
    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(req.name) or {}
    return _registry_to_view(req.name, entry)


@app.put("/datasets/{name}", response_model=DatasetView)
async def update_dataset_endpoint(name: str, req: DatasetUpdateRequest) -> DatasetView:
    """Manual meta edit (desc / use / growing). Bumps the dataset version
    and emits a Lesson(kind="dataset", proposal_source="human")."""
    from harness.executor.promote import record_manual_dataset_change
    from router.data.dataset_io import load_current, save_dataset
    from router.data.dataset_registry import _load_registry, _vd
    from router.errors import DatasetInvalidError, DatasetNotFoundError

    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None or entry.get("deleted"):
        raise HTTPException(status_code=404, detail=f"dataset_not_found: {name}")

    if req.desc is None and req.use is None and req.growing is None:
        raise HTTPException(status_code=400, detail="no fields to update")

    try:
        dataset = load_current(name)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DatasetInvalidError as e:
        raise HTTPException(status_code=500, detail=f"dataset_invalid: {e}")

    changes: list[str] = []
    new_desc = dataset.metadata.desc if req.desc is None else req.desc
    new_use = dataset.metadata.use if req.use is None else list(req.use)
    new_growing = dataset.metadata.growing if req.growing is None else bool(req.growing)

    if req.desc is not None and req.desc != dataset.metadata.desc:
        changes.append(f"desc → {req.desc[:60]!r}")
    if req.use is not None and req.use != dataset.metadata.use:
        changes.append(f"use → {req.use}")
    if req.growing is not None and req.growing != dataset.metadata.growing:
        changes.append(f"growing → {req.growing}")

    new_version = dataset.version + 1
    now = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    payload = {
        "version": new_version,
        "name": name,
        "desc": new_desc,
        "source": dataset.metadata.source,
        "sourceType": dataset.metadata.sourceType,
        "use": new_use,
        "owner": dataset.metadata.owner,
        "growing": new_growing,
        "created_at": dataset.created_at or now,
        "embedder_model": dataset.metadata.embedder_model,
        "embedding_dim": dataset.metadata.embedding_dim,
        "samples": [
            {
                "id": s.id,
                "prompt": s.prompt,
                "ground_truth": s.ground_truth,
                "tag": s.tag,
                "trace_id": s.trace_id,
                "added_at": s.added_at,
                "source": s.source,
                "embedding": s.embedding,
            }
            for s in dataset.samples
        ],
        "history": list(dataset.history) + [
            {"when": now, "what": f"You edited the dataset ({', '.join(changes) or 'no-op'})."}
        ],
        "metadata": {
            **(dataset.extra or {}),
            "stage": "manual_edit",
            "previous_version": dataset.version,
        },
    }
    def _writer() -> None:
        save_dataset(payload)

    summary = f"Edited dataset '{name}': {'; '.join(changes) or 'no-op'}"
    record_manual_dataset_change(
        name=name,
        new_version=new_version,
        summary=summary,
        apply_edit=_writer,
        voice=f"I tweaked the '{name}' dataset's metadata.",
    )

    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(name) or {}
    return _registry_to_view(name, entry)


@app.get("/datasets/{name}/export")
async def export_dataset_endpoint(name: str):
    """Stream the full dataset as NDJSON. Each line is one sample with
    `prompt`, `ground_truth`, `tag`, `trace_id`, `added_at`, `source`,
    and `embedding`. The wire shape from /datasets/{name} hides
    embeddings + truncates prompt to a preview — this endpoint is the
    full payload, suitable for distillation / offline analysis."""
    from router.data.dataset_io import load_current
    from router.data.dataset_registry import _load_registry, _vd
    from router.errors import DatasetInvalidError, DatasetNotFoundError

    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None or entry.get("deleted"):
        raise HTTPException(status_code=404, detail=f"dataset_not_found: {name}")

    try:
        dataset = load_current(name)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DatasetInvalidError as e:
        raise HTTPException(status_code=500, detail=f"dataset_invalid: {e}")

    def _iter_lines():
        for s in dataset.samples:
            yield json.dumps({
                "id": s.id,
                "prompt": s.prompt,
                "ground_truth": s.ground_truth,
                "tag": s.tag,
                "trace_id": s.trace_id,
                "added_at": s.added_at,
                "source": s.source,
                "embedding": s.embedding,
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(
        _iter_lines(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.jsonl"',
        },
    )


@app.delete("/datasets/{name}", status_code=204)
async def delete_dataset_endpoint(name: str) -> None:
    """Soft-delete: mark as deleted in the registry; v<n>.json files stay
    on disk for rollback. Not a versioned mutation."""
    from router.data.dataset_registry import (
        _load_registry,
        _vd,
        delete_dataset,
    )

    registry = _load_registry(datasets_dir=_vd(None))
    entry = (registry.get("datasets") or {}).get(name)
    if entry is None or entry.get("deleted"):
        raise HTTPException(status_code=404, detail=f"dataset_not_found: {name}")

    delete_dataset(name)
    return None


# ---------------------------------------------------------------------------
# Day-0 onboarding (P1.11)
# ---------------------------------------------------------------------------


class OnboardingState(BaseModel):
    template: Optional[str] = None
    name: str = ""
    company: str = ""
    prompt: str = ""
    model: str = "claude-sonnet-4-6"
    tools: list[str] = []
    channels: list[str] = []
    completed: bool = False
    completed_at: Optional[str] = None
    skipped: bool = False


class OnboardingCompleteRequest(BaseModel):
    template: Optional[str] = None
    name: str = ""
    company: str = ""
    prompt: str = ""
    model: str = "claude-sonnet-4-6"
    tools: list[str] = []
    channels: list[str] = []


@app.get("/onboarding/state", response_model=OnboardingState)
def onboarding_state() -> OnboardingState:
    """Returns the current onboarding config. completed=False on first run."""
    from runtime.store.onboarding import load_state
    cfg = load_state()
    return OnboardingState(**cfg.to_dict())


@app.post("/onboarding/complete", response_model=OnboardingState)
def onboarding_complete(payload: OnboardingCompleteRequest) -> OnboardingState:
    """Materialize the day-0 config. P2.0+ creates a NEW agent in the
    registry (``agents/<id>/``) and activates it instead of overwriting
    the legacy global ``agent/`` dir. Backwards-compat: also bumps
    the legacy onboarding.json so the gate logic in the UI keeps
    working without further changes."""
    from runtime.agents.registry import activate as activate_agent
    from runtime.agents.registry import create_agent
    from runtime.store.onboarding import record_complete

    body = payload.model_dump()

    # Create the agent entry + activate it. The activate hook recompiles
    # the pipeline so /run starts using the new prompt/model immediately.
    try:
        meta = create_agent(body)
        activate_agent(meta.id, on_activate=lambda m: _reload_live_pipeline(m.id))
    except Exception as e:
        logger.warning(
            "agent registry create+activate failed (%s) — falling back to legacy "
            "single-agent record_complete()", e,
        )

    # Keep the legacy state file in sync so /onboarding/state returns
    # completed=True; this also writes the agent_created Lesson via the
    # existing hook so Evolution still surfaces the birth.
    cfg = record_complete(body)
    return OnboardingState(**cfg.to_dict())


@app.post("/onboarding/skip", response_model=OnboardingState)
def onboarding_skip() -> OnboardingState:
    """Operator skipped — mark complete without launching anything."""
    from runtime.store.onboarding import record_skip
    cfg = record_skip()
    return OnboardingState(**cfg.to_dict())


# Conversational onboarding turn (P1.12)


class OnboardingChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class OnboardingTurnRequest(BaseModel):
    messages: list[OnboardingChatMessage]
    model: Optional[str] = None  # override (default: claude-sonnet-4-6)


class OnboardingTurnConfig(BaseModel):
    name: str
    model: str
    prompt: str
    tools: list[str]
    channels: list[str]


class OnboardingJustAdded(BaseModel):
    tool: Optional[str] = None
    model: Optional[str] = None
    channel: Optional[str] = None


class OnboardingTurnResponse(BaseModel):
    reply: str
    config: OnboardingTurnConfig
    justAdded: Optional[OnboardingJustAdded] = None
    ready: bool = False


@app.post("/onboarding/turn", response_model=OnboardingTurnResponse)
def onboarding_turn(payload: OnboardingTurnRequest) -> OnboardingTurnResponse:
    """One conversational turn during day-0 setup. Claude reads the
    running history and returns the next reply + the agent config it
    has built so far. Offline-safe via the scripted fallback."""
    from runtime.store.onboarding_chat import run_turn

    out = run_turn(
        [m.model_dump() for m in payload.messages],
        model=payload.model,
    )
    return OnboardingTurnResponse(**out)


class OnboardingTransportInfo(BaseModel):
    transport: str           # "claude_code_cli" | "anthropic_api" | "none"
    cwd: str
    claude_version: Optional[str] = None


@app.get("/onboarding/transport", response_model=OnboardingTransportInfo)
def onboarding_transport() -> OnboardingTransportInfo:
    """Which brain is connected (Claude Code CLI vs Anthropic API vs none).
    UI surfaces this as a badge on the welcome screen so the operator
    sees what's powering the chat — and whether it has filesystem access."""
    from runtime.store.onboarding_chat import detect_transport
    return OnboardingTransportInfo(**detect_transport())


# ---------------------------------------------------------------------------
# Multi-agent (P2.0)
# ---------------------------------------------------------------------------


class AgentSummary(BaseModel):
    id: str
    name: str
    template: Optional[str] = None
    model: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    onboarding_completed_at: Optional[str] = None
    is_active: bool = False


class AgentListResponse(BaseModel):
    agents: list[AgentSummary]
    active: Optional[str] = None


class AgentCreateRequest(BaseModel):
    name: str
    prompt: str
    model: str = "claude-sonnet-4-6"
    template: Optional[str] = None
    company: str = ""
    tools: list[str] = []
    channels: list[str] = []
    activate: bool = False  # if True, switch to this agent immediately


def _summarize(meta, *, active_id: Optional[str]) -> AgentSummary:
    return AgentSummary(
        id=meta.id,
        name=meta.name,
        template=meta.template,
        model=meta.model,
        description=meta.description,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        onboarding_completed_at=meta.onboarding_completed_at,
        is_active=(meta.id == active_id),
    )


def _reload_live_pipeline(agent_id: Optional[str] = None) -> None:
    """Recompile the pipeline from the live ``agent/`` dir and swap into
    the running executor. Called after activate(). Also updates the
    process-global agent context so subsequent writes (traces, lessons,
    etc) partition under the new agent."""
    cfg = load_agent("agent/agent.yaml")
    pipeline = compile_agent(cfg)
    executor = PipelineExecutor(pipeline)
    _state["cfg"] = cfg
    _state["executor"] = executor
    if agent_id:
        from runtime.agent_context import set_active
        set_active(agent_id)
    logger.info("agent %s re-compiled after activate (%d stages)", cfg.version, len(pipeline.stages))


@app.get("/agents", response_model=AgentListResponse)
def list_agents_endpoint() -> AgentListResponse:
    from runtime.agents.registry import get_registry
    reg = get_registry()
    return AgentListResponse(
        agents=[_summarize(a, active_id=reg.active) for a in reg.agents],
        active=reg.active,
    )


@app.get("/agents/{agent_id}", response_model=AgentSummary)
def get_agent_endpoint(agent_id: str) -> AgentSummary:
    from runtime.agents.registry import get_agent, get_registry
    meta = get_agent(agent_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    return _summarize(meta, active_id=get_registry().active)


@app.post("/agents", response_model=AgentSummary, status_code=201)
def create_agent_endpoint(payload: AgentCreateRequest) -> AgentSummary:
    """Create a new agent from the onboarding payload. If ``activate``
    is true, swap the live ``agent/`` dir + recompile the pipeline."""
    from runtime.agents.registry import activate as activate_agent
    from runtime.agents.registry import create_agent, get_registry

    meta = create_agent(payload.model_dump())
    if payload.activate:
        activate_agent(meta.id, on_activate=lambda m: _reload_live_pipeline(m.id))
    return _summarize(meta, active_id=get_registry().active)


class AgentActivateResponse(BaseModel):
    active: str
    agent_version: str


@app.post("/agents/{agent_id}/activate", response_model=AgentActivateResponse)
def activate_agent_endpoint(agent_id: str) -> AgentActivateResponse:
    from runtime.agents.registry import activate as activate_agent
    try:
        activate_agent(agent_id, on_activate=lambda m: _reload_live_pipeline(m.id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AgentActivateResponse(
        active=agent_id,
        agent_version=_state["cfg"].version,
    )


@app.delete("/agents/{agent_id}", status_code=204)
def delete_agent_endpoint(agent_id: str) -> None:
    from runtime.agents.registry import delete_agent
    try:
        delete_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return None


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None


class ChannelStatus(BaseModel):
    connected: bool
    meta: dict


class AgentChannelsResponse(BaseModel):
    agent_id: str
    channels: dict[str, ChannelStatus]


class ApiChannelStatus(BaseModel):
    connected: bool
    token_mask: Optional[str] = None
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None


@app.get("/agents/{agent_id}/channels/api", response_model=ApiChannelStatus)
def get_api_channel_endpoint(agent_id: str) -> ApiChannelStatus:
    from runtime.agents.channels import load
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    cfg = load(agent_id, "api")
    if cfg is None:
        return ApiChannelStatus(connected=False)
    from runtime.agents.channels import _mask_token
    return ApiChannelStatus(
        connected=True,
        token_mask=_mask_token(cfg.get("token", "")),
        created_at=cfg.get("created_at"),
        last_used_at=cfg.get("last_used_at"),
    )


class ApiChannelConnectResponse(BaseModel):
    connected: bool
    token: str
    token_mask: str
    created_at: str
    public_url: str


def _mint_api_token() -> str:
    import secrets as _secrets
    return f"ot_{_secrets.token_urlsafe(32)}"


@app.post("/agents/{agent_id}/channels/api/connect", response_model=ApiChannelConnectResponse)
def connect_api_channel_endpoint(agent_id: str, request: Request) -> ApiChannelConnectResponse:
    """Mint a token. Returned ONCE (raw) so the operator can copy it.
    Subsequent reads only get a mask."""
    from runtime.agents.channels import _mask_token, load, save
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    existing = load(agent_id, "api")
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="already_connected: rotate to mint a new token, or disconnect first",
        )

    token = _mint_api_token()
    created_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    save(agent_id, "api", {
        "token": token,
        "created_at": created_at,
        "last_used_at": None,
    })
    base = str(request.base_url).rstrip("/")
    return ApiChannelConnectResponse(
        connected=True,
        token=token,
        token_mask=_mask_token(token),
        created_at=created_at,
        public_url=f"{base}/api/{agent_id}/chat",
    )


@app.post("/agents/{agent_id}/channels/api/rotate", response_model=ApiChannelConnectResponse)
def rotate_api_channel_endpoint(agent_id: str, request: Request) -> ApiChannelConnectResponse:
    """Replace the current token. Old one stops working immediately."""
    from runtime.agents.channels import _mask_token, load, save
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    existing = load(agent_id, "api") or {}

    token = _mint_api_token()
    created_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    save(agent_id, "api", {
        "token": token,
        "created_at": existing.get("created_at") or created_at,
        "last_used_at": existing.get("last_used_at"),
        "rotated_at": created_at,
    })
    base = str(request.base_url).rstrip("/")
    return ApiChannelConnectResponse(
        connected=True,
        token=token,
        token_mask=_mask_token(token),
        created_at=existing.get("created_at") or created_at,
        public_url=f"{base}/api/{agent_id}/chat",
    )


@app.delete("/agents/{agent_id}/channels/api", status_code=204)
def disconnect_api_channel_endpoint(agent_id: str) -> None:
    from runtime.agents.channels import remove
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    remove(agent_id, "api")
    return None


# ─── Web Widget channel (P3.5) ────────────────────────────────────────────
# Per-agent floating chat widget operators embed on their website. The
# widget_id is public (visible in the embed snippet); the signing secret
# is private (used to verify outbound webhooks from us to the operator's
# backend, if they wire one). The embed JS posts to /widget/<widget_id>/
# message; we validate the Origin against allowed_domains, then run the
# agent and return the response synchronously.


class WebWidgetSettings(BaseModel):
    position: str = "br"               # br | bl
    shape: str = "circle"              # circle | rounded | pill
    accent: str = "green"              # green | blue | plum | slate | brand
    greeting: str = ""
    welcome: str = ""
    fallback: str = ""
    show_greeting: bool = True
    require_email: bool = False
    pill_label: str = "Chat"


class WebChannelStatus(BaseModel):
    connected: bool
    widget_id: Optional[str] = None
    signing_secret_mask: Optional[str] = None
    allowed_domains: list[str] = []
    settings: WebWidgetSettings = WebWidgetSettings()
    installed_at: Optional[str] = None
    embed_url: Optional[str] = None
    message_url: Optional[str] = None


class WebChannelConnectResponse(BaseModel):
    connected: bool
    widget_id: str
    signing_secret: str                  # returned ONCE on connect/rotate
    signing_secret_mask: str
    installed_at: str
    embed_url: str
    message_url: str


class WebChannelUpdateRequest(BaseModel):
    allowed_domains: Optional[list[str]] = None
    settings: Optional[WebWidgetSettings] = None


def _mint_widget_id() -> str:
    import secrets as _secrets
    return "w_" + _secrets.token_hex(8)


def _mint_widget_secret() -> str:
    import secrets as _secrets
    return "whsec_" + _secrets.token_urlsafe(24)


def _normalize_domain(d: str) -> str:
    d = d.strip().lower()
    if d.startswith("http://"):
        d = d[7:]
    elif d.startswith("https://"):
        d = d[8:]
    # strip path / port
    return d.split("/", 1)[0].split(":", 1)[0]


def _web_public_url(request: Request, widget_id: str, kind: str) -> str:
    """Build the public URL operators paste into their site.

    Prefers ``PUBLIC_BASE_URL`` so production URLs route through the TS
    gateway (8002), not the internal Python runtime (8001). Falls back to
    the request's own base for local dev where they're the same host."""
    import os
    base = (os.environ.get("PUBLIC_BASE_URL") or str(request.base_url)).rstrip("/")
    if kind == "embed":
        return f"{base}/widget/{widget_id}/v1.js"
    return f"{base}/widget/{widget_id}/message"


def _web_status(agent_id: str, request: Request) -> WebChannelStatus:
    from runtime.agents.channels import _mask_token, load
    cfg = load(agent_id, "web")
    if cfg is None:
        return WebChannelStatus(connected=False)
    settings = cfg.get("settings") or {}
    return WebChannelStatus(
        connected=True,
        widget_id=cfg.get("widget_id"),
        signing_secret_mask=_mask_token(cfg.get("signing_secret", "")),
        allowed_domains=list(cfg.get("allowed_domains", [])),
        settings=WebWidgetSettings(**settings),
        installed_at=cfg.get("installed_at"),
        embed_url=_web_public_url(request, cfg.get("widget_id", ""), "embed"),
        message_url=_web_public_url(request, cfg.get("widget_id", ""), "message"),
    )


@app.get("/agents/{agent_id}/channels/web", response_model=WebChannelStatus)
def get_web_channel_endpoint(agent_id: str, request: Request) -> WebChannelStatus:
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    return _web_status(agent_id, request)


@app.post(
    "/agents/{agent_id}/channels/web/connect",
    response_model=WebChannelConnectResponse,
)
def connect_web_channel_endpoint(
    agent_id: str, request: Request,
) -> WebChannelConnectResponse:
    """Mint a widget_id + signing secret. Returned once so the operator
    can copy the snippet; subsequent reads only get a mask."""
    from runtime.agents.channels import _mask_token, load, save
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    existing = load(agent_id, "web")
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="already_connected: rotate secret or disconnect first",
        )
    widget_id = _mint_widget_id()
    secret = _mint_widget_secret()
    installed_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    save(agent_id, "web", {
        "widget_id": widget_id,
        "signing_secret": secret,
        "allowed_domains": [],
        "settings": WebWidgetSettings().model_dump(),
        "installed_at": installed_at,
    })
    return WebChannelConnectResponse(
        connected=True,
        widget_id=widget_id,
        signing_secret=secret,
        signing_secret_mask=_mask_token(secret),
        installed_at=installed_at,
        embed_url=_web_public_url(request, widget_id, "embed"),
        message_url=_web_public_url(request, widget_id, "message"),
    )


@app.post(
    "/agents/{agent_id}/channels/web/rotate-secret",
    response_model=WebChannelConnectResponse,
)
def rotate_web_channel_secret_endpoint(
    agent_id: str, request: Request,
) -> WebChannelConnectResponse:
    """Replace the signing secret. Old one stops working immediately."""
    from runtime.agents.channels import _mask_token, load, save
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    existing = load(agent_id, "web")
    if existing is None:
        raise HTTPException(status_code=404, detail="not_connected")
    secret = _mint_widget_secret()
    existing["signing_secret"] = secret
    existing["rotated_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    save(agent_id, "web", existing)
    return WebChannelConnectResponse(
        connected=True,
        widget_id=existing["widget_id"],
        signing_secret=secret,
        signing_secret_mask=_mask_token(secret),
        installed_at=existing.get("installed_at", existing["rotated_at"]),
        embed_url=_web_public_url(request, existing["widget_id"], "embed"),
        message_url=_web_public_url(request, existing["widget_id"], "message"),
    )


@app.patch("/agents/{agent_id}/channels/web", response_model=WebChannelStatus)
def update_web_channel_endpoint(
    agent_id: str,
    payload: WebChannelUpdateRequest,
    request: Request,
) -> WebChannelStatus:
    """Update allowed domains or appearance/behavior settings."""
    from runtime.agents.channels import load, save
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    cfg = load(agent_id, "web")
    if cfg is None:
        raise HTTPException(status_code=404, detail="not_connected")
    if payload.allowed_domains is not None:
        # Dedupe + normalize before persisting so we never store
        # protocol/port/path mixed with bare hostnames.
        seen: list[str] = []
        for d in payload.allowed_domains:
            n = _normalize_domain(d)
            if n and n not in seen:
                seen.append(n)
        cfg["allowed_domains"] = seen
    if payload.settings is not None:
        cfg["settings"] = payload.settings.model_dump()
    save(agent_id, "web", cfg)
    return _web_status(agent_id, request)


@app.delete("/agents/{agent_id}/channels/web", status_code=204)
def disconnect_web_channel_endpoint(agent_id: str) -> None:
    from runtime.agents.channels import remove
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    remove(agent_id, "web")
    return None


# ─── Public widget endpoints — Origin-gated, no bearer token ───────────────


class WidgetMessageRequest(BaseModel):
    message: str
    session: Optional[str] = None
    history: Optional[list[HistoryMessage]] = None


class WidgetMessageResponse(BaseModel):
    response: Optional[str]
    session: str
    trace_id: str


def _origin_matches(origin: str, allowed: list[str]) -> bool:
    """Allow exact match, plus a single leading wildcard for subdomains
    (``*.acme.com`` matches ``help.acme.com`` but not ``acme.com``).
    Localhost is allowed when ``allowed`` is empty so the operator can
    test the embed locally before pinning origins."""
    if not allowed:
        return origin in ("", "null") or origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1")
    host = _normalize_domain(origin)
    for d in allowed:
        if d.startswith("*."):
            if host.endswith(d[1:]):
                return True
        elif host == d:
            return True
    return False


@app.options("/widget/{widget_id}/message")
def widget_message_options(widget_id: str, request: Request) -> Response:
    """CORS preflight — the embed script runs cross-origin on the
    customer's site, so we must echo the requested origin."""
    origin = request.headers.get("origin", "")
    headers = {
        "Access-Control-Allow-Origin": origin or "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "content-type",
        "Access-Control-Max-Age": "600",
    }
    return Response(status_code=204, headers=headers)


@app.post("/widget/{widget_id}/message", response_model=WidgetMessageResponse)
def widget_message_endpoint(
    widget_id: str, payload: WidgetMessageRequest, request: Request,
) -> WidgetMessageResponse:
    """Public inbound: the embed JS POSTs every visitor message here. We
    look up the owning agent by widget_id, gate on Origin (allowed_domains),
    run the pipeline, return the response synchronously. No auth — the
    widget_id itself is the routing token; origin pinning is the gate."""
    from runtime.agents.channels import find_agent_by_channel, load
    from runtime.agents.registry import activate as activate_agent
    from runtime.agents.registry import get_registry
    from runtime.executor.tracing import write_trace
    from runtime.protocols import Message

    agent_id = find_agent_by_channel("web", {"widget_id": widget_id})
    if agent_id is None:
        raise HTTPException(status_code=404, detail="unknown_widget")
    cfg = load(agent_id, "web") or {}
    origin = request.headers.get("origin", "")
    if not _origin_matches(origin, list(cfg.get("allowed_domains", []))):
        raise HTTPException(status_code=403, detail="origin_not_allowed")

    reg = get_registry()
    if reg.active != agent_id:
        try:
            activate_agent(agent_id, on_activate=lambda m: _reload_live_pipeline(m.id))
        except Exception as e:
            logger.warning("widget message: failed to activate %s: %s", agent_id, e)

    history = [Message(role=m.role, content=m.content) for m in (payload.history or [])]
    executor = _state["executor"]
    _, exec_record = executor.run(payload.message, history=history)
    trace_id = write_trace(exec_record)
    session = payload.session or f"web_{uuid.uuid4().hex[:12]}"
    resp = WidgetMessageResponse(
        response=exec_record.response,
        session=session,
        trace_id=trace_id,
    )
    # Echo the CORS header so the browser delivers the body.
    return JSONResponse(
        content=resp.model_dump(),
        headers={
            "Access-Control-Allow-Origin": origin or "*",
            "Vary": "Origin",
        },
    )


@app.get("/widget/{widget_id}/v1.js")
def widget_embed_js_endpoint(widget_id: str, request: Request) -> Response:
    """Self-hostable embed script. The operator drops the snippet (which
    only knows widget_id + endpoint host) and this serves a tiny launcher
    + chat panel UI that talks to /widget/<id>/message.

    Kept intentionally small + dependency-free so it ships fast and is
    easy to audit. The full design lives in the dashboard; this is the
    bootstrap that customers see on the operator's site."""
    from runtime.agents.channels import load, find_agent_by_channel
    agent_id = find_agent_by_channel("web", {"widget_id": widget_id})
    if agent_id is None:
        return Response(status_code=404, content="// unknown widget", media_type="application/javascript")
    cfg = load(agent_id, "web") or {}
    settings = cfg.get("settings") or {}
    base = str(request.base_url).rstrip("/")
    accent_map = {
        "green": "#22a06b",
        "blue": "#2563eb",
        "plum": "#9333ea",
        "slate": "#334155",
        "brand": "#ea580c",
    }
    accent = accent_map.get(settings.get("accent", "green"), "#22a06b")
    position = settings.get("position", "br")
    welcome = (settings.get("welcome") or "Hi! How can I help?").replace("`", "\\`")
    greeting = (settings.get("greeting") or "").replace("`", "\\`")
    js = _WIDGET_JS_TEMPLATE % {
        "widget_id": widget_id,
        "endpoint": f"{base}/widget/{widget_id}/message",
        "accent": accent,
        "position": position,
        "welcome": welcome,
        "greeting": greeting,
        "pill_label": settings.get("pill_label", "Chat"),
    }
    return Response(
        content=js,
        media_type="application/javascript",
        headers={
            "Cache-Control": "public, max-age=60",
            "Access-Control-Allow-Origin": "*",
        },
    )


_WIDGET_JS_TEMPLATE = r"""// OpenTracy Web Widget v1 — dependency-free
(function(){
  if (window.__OT_WIDGET_LOADED__) return;
  window.__OT_WIDGET_LOADED__ = true;
  var WID = "%(widget_id)s";
  var URL_ = "%(endpoint)s";
  var ACCENT = "%(accent)s";
  var POS = "%(position)s";
  var WELCOME = "%(welcome)s";
  var GREETING = "%(greeting)s";
  var PILL = "%(pill_label)s";
  var session = null;
  var history = [];
  var open = false;

  function el(tag, props){
    var e = document.createElement(tag);
    for (var k in props) {
      if (k === "style") for (var s in props[k]) e.style[s] = props[k][s];
      else if (k === "text") e.textContent = props[k];
      else e[k] = props[k];
    }
    return e;
  }

  var root = el("div", { style: {
    position: "fixed",
    zIndex: 2147483000,
    bottom: "20px",
    right: POS === "br" ? "20px" : "auto",
    left: POS === "bl" ? "20px" : "auto",
    fontFamily: "system-ui, sans-serif",
  }});

  var launcher = el("button", {
    style: {
      width: "56px", height: "56px", borderRadius: "50%%",
      background: ACCENT, color: "white", border: "none",
      boxShadow: "0 6px 24px rgba(0,0,0,0.18)", cursor: "pointer",
      fontSize: "24px",
    },
    text: "\u{1F4AC}",
    onclick: function(){ togglePanel(); },
  });

  var panel = el("div", { style: {
    position: "absolute",
    bottom: "70px",
    right: POS === "br" ? "0" : "auto",
    left: POS === "bl" ? "0" : "auto",
    width: "340px", maxWidth: "calc(100vw - 40px)",
    height: "480px", maxHeight: "calc(100vh - 100px)",
    background: "white", color: "#111",
    borderRadius: "14px", boxShadow: "0 12px 40px rgba(0,0,0,0.22)",
    display: "none", flexDirection: "column", overflow: "hidden",
  }});

  var head = el("div", { style: {
    padding: "14px 16px", background: ACCENT, color: "white",
    fontWeight: "600", fontSize: "14px",
  }, text: "Chat with us"});
  var thread = el("div", { style: {
    flex: "1", overflowY: "auto", padding: "14px",
    display: "flex", flexDirection: "column", gap: "8px",
    fontSize: "13.5px",
  }});
  var inputRow = el("div", { style: {
    borderTop: "1px solid #eee", padding: "10px", display: "flex", gap: "8px",
  }});
  var inputField = el("input", {
    type: "text", placeholder: "Type a message…",
    style: { flex: "1", padding: "8px 10px", border: "1px solid #ddd",
             borderRadius: "8px", fontSize: "13.5px", outline: "none" },
  });
  var sendBtn = el("button", {
    style: { padding: "8px 12px", background: ACCENT, color: "white",
             border: "none", borderRadius: "8px", cursor: "pointer",
             fontSize: "13.5px" },
    text: "Send",
  });
  inputField.addEventListener("keydown", function(e){
    if (e.key === "Enter") send();
  });
  sendBtn.onclick = send;

  inputRow.appendChild(inputField);
  inputRow.appendChild(sendBtn);
  panel.appendChild(head);
  panel.appendChild(thread);
  panel.appendChild(inputRow);
  root.appendChild(panel);
  root.appendChild(launcher);

  function addBubble(role, text){
    var b = el("div", { style: {
      alignSelf: role === "user" ? "flex-end" : "flex-start",
      background: role === "user" ? ACCENT : "#f1f3f5",
      color: role === "user" ? "white" : "#111",
      padding: "8px 12px", borderRadius: "14px", maxWidth: "80%%",
      whiteSpace: "pre-wrap", lineHeight: "1.45",
    }, text: text});
    thread.appendChild(b);
    thread.scrollTop = thread.scrollHeight;
  }

  function togglePanel(){
    open = !open;
    panel.style.display = open ? "flex" : "none";
    if (open && thread.childNodes.length === 0 && WELCOME) {
      addBubble("agent", WELCOME);
    }
    if (open) setTimeout(function(){ inputField.focus(); }, 80);
  }

  function send(){
    var text = inputField.value.trim();
    if (!text) return;
    addBubble("user", text);
    inputField.value = "";
    history.push({ role: "user", content: text });
    var typing = el("div", { style: {
      alignSelf: "flex-start", color: "#888", fontSize: "12px",
      padding: "4px 12px",
    }, text: "…"});
    thread.appendChild(typing);
    fetch(URL_, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: text, session: session, history: history }),
    }).then(function(r){ return r.json(); }).then(function(data){
      thread.removeChild(typing);
      session = data.session || session;
      var reply = data.response || "(no response)";
      addBubble("agent", reply);
      history.push({ role: "assistant", content: reply });
    }).catch(function(){
      thread.removeChild(typing);
      addBubble("agent", "Sorry, something went wrong. Try again?");
    });
  }

  document.body.appendChild(root);
})();
"""


# ─── Public API: external callers chat with an agent via bearer token ─────


class ApiChatRequest(BaseModel):
    request: str
    history: Optional[list[HistoryMessage]] = None


class ApiChatResponse(BaseModel):
    response: Optional[str]
    trace_id: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


@app.post("/api/{agent_id}/chat", response_model=ApiChatResponse)
def api_chat_endpoint(
    agent_id: str, payload: ApiChatRequest, request: Request,
) -> ApiChatResponse:
    """Public chat endpoint — authenticate via Bearer token, route to
    the agent, return the response synchronously. The token is the one
    minted by /agents/<id>/channels/api/connect."""
    from runtime.agents.channels import load, save
    from runtime.agents.registry import activate as activate_agent
    from runtime.agents.registry import get_agent, get_registry
    from runtime.executor.tracing import write_trace
    from runtime.protocols import Message

    auth = request.headers.get("authorization", "")
    m = _BEARER_RE.match(auth)
    if not m:
        raise HTTPException(status_code=401, detail="missing_bearer_token")
    token = m.group(1).strip()

    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    cfg = load(agent_id, "api")
    if cfg is None or cfg.get("token") != token:
        raise HTTPException(status_code=401, detail="invalid_token")

    # Bump last_used_at so the UI shows freshness
    now_iso = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    cfg["last_used_at"] = now_iso
    save(agent_id, "api", cfg)

    # Activate the agent if not already, so /run uses the right pipeline
    reg = get_registry()
    if reg.active != agent_id:
        try:
            activate_agent(agent_id, on_activate=lambda m: _reload_live_pipeline(m.id))
        except Exception as e:
            logger.warning("api chat: failed to activate %s: %s", agent_id, e)

    history = [Message(role=m.role, content=m.content) for m in (payload.history or [])]
    executor = _state["executor"]
    _, exec_record = executor.run(payload.request, history=history)
    trace_id = write_trace(exec_record)

    return ApiChatResponse(
        response=exec_record.response,
        trace_id=trace_id,
        duration_ms=exec_record.duration_ms,
        success=exec_record.success,
        error=exec_record.error,
    )


_BEARER_RE = re.compile(r"^Bearer\s+(.+)$")


@app.post("/api/{agent_id}/internal-run", response_model=ApiChatResponse)
def internal_run_endpoint(
    agent_id: str, payload: ApiChatRequest,
) -> ApiChatResponse:
    """Internal runtime endpoint used by trusted backends (Slack events
    webhook, future channel handlers). NOT exposed publicly — only the
    same-host TS gateway should hit this. Skips bearer auth because the
    Slack-side signature already verified the request came from Slack;
    if the TS gateway is compromised the rest of the system is too.

    Activates the agent if needed, runs the pipeline, returns the result.
    """
    from runtime.agents.registry import activate as activate_agent
    from runtime.agents.registry import get_agent, get_registry
    from runtime.executor.tracing import write_trace
    from runtime.protocols import Message

    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")

    reg = get_registry()
    if reg.active != agent_id:
        try:
            activate_agent(agent_id, on_activate=lambda m: _reload_live_pipeline(m.id))
        except Exception as e:
            logger.warning("internal_run: failed to activate %s: %s", agent_id, e)

    history = [Message(role=m.role, content=m.content) for m in (payload.history or [])]
    executor = _state["executor"]
    _, exec_record = executor.run(payload.request, history=history)
    trace_id = write_trace(exec_record)
    return ApiChatResponse(
        response=exec_record.response,
        trace_id=trace_id,
        duration_ms=exec_record.duration_ms,
        success=exec_record.success,
        error=exec_record.error,
    )


class MCPServerView(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: Optional[str] = None
    enabled: bool = True
    description: str = ""


class MCPServersResponse(BaseModel):
    agent_id: str
    servers: list[MCPServerView]


class MCPToolView(BaseModel):
    server_name: str
    tool_name: str
    qualified_name: str
    description: str
    input_schema: dict


class MCPToolsResponse(BaseModel):
    agent_id: str
    tools: list[MCPToolView]
    discovery_errors: list[str] = []


@app.get("/agents/{agent_id}/mcp", response_model=MCPServersResponse)
def list_mcp_servers_endpoint(agent_id: str) -> MCPServersResponse:
    from runtime.agents.mcp import load
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    servers = load(agent_id)
    return MCPServersResponse(
        agent_id=agent_id,
        servers=[MCPServerView(**s.to_dict()) for s in servers],
    )


class MCPServerCreateRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: Optional[str] = None
    enabled: bool = True
    description: str = ""


@app.post("/agents/{agent_id}/mcp", response_model=MCPServersResponse, status_code=201)
def add_mcp_server_endpoint(
    agent_id: str, payload: MCPServerCreateRequest,
) -> MCPServersResponse:
    from runtime.agents.mcp import MCPServer, add_server
    from runtime.agents.registry import get_agent
    from runtime.mcp.client import invalidate_cache
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="missing_name")
    try:
        servers = add_server(agent_id, MCPServer(**payload.model_dump()))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    invalidate_cache(agent_id)
    return MCPServersResponse(
        agent_id=agent_id,
        servers=[MCPServerView(**s.to_dict()) for s in servers],
    )


class MCPServerUpdateRequest(BaseModel):
    transport: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    url: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None


@app.patch("/agents/{agent_id}/mcp/{server_name}", response_model=MCPServersResponse)
def update_mcp_server_endpoint(
    agent_id: str, server_name: str, payload: MCPServerUpdateRequest,
) -> MCPServersResponse:
    from runtime.agents.mcp import update_server
    from runtime.agents.registry import get_agent
    from runtime.mcp.client import invalidate_cache
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    try:
        servers = update_server(
            agent_id, server_name, **payload.model_dump(exclude_unset=True),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"server_not_found: {server_name}")
    invalidate_cache(agent_id)
    return MCPServersResponse(
        agent_id=agent_id,
        servers=[MCPServerView(**s.to_dict()) for s in servers],
    )


@app.delete("/agents/{agent_id}/mcp/{server_name}", status_code=204)
def remove_mcp_server_endpoint(agent_id: str, server_name: str) -> None:
    from runtime.agents.mcp import remove_server
    from runtime.agents.registry import get_agent
    from runtime.mcp.client import invalidate_cache
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    remove_server(agent_id, server_name)
    invalidate_cache(agent_id)
    return None


@app.get("/agents/{agent_id}/mcp/tools", response_model=MCPToolsResponse)
def discover_mcp_tools_endpoint(agent_id: str) -> MCPToolsResponse:
    """Live discovery: spawns each enabled server, calls listTools,
    returns the combined catalog. Cache invalidated on every mutation."""
    from runtime.agents.registry import get_agent
    from runtime.mcp.client import list_tools_for_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    discovered = list_tools_for_agent(agent_id, force_refresh=True)
    return MCPToolsResponse(
        agent_id=agent_id,
        tools=[
            MCPToolView(
                server_name=t.server_name,
                tool_name=t.tool_name,
                qualified_name=t.qualified_name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in discovered
        ],
    )


@app.get("/agents/{agent_id}/channels", response_model=AgentChannelsResponse)
def list_agent_channels_endpoint(agent_id: str) -> AgentChannelsResponse:
    """Per-channel connection status for the agent (P3.3). Used by the
    AgentSheet's Channels tab. Channel-specific connect/disconnect lives
    on the dedicated channel routers (api/slack/whatsapp)."""
    from runtime.agents.channels import status as channel_status
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    raw = channel_status(agent_id)
    return AgentChannelsResponse(
        agent_id=agent_id,
        channels={k: ChannelStatus(**v) for k, v in raw.items()},
    )


class ImprovementConfigView(BaseModel):
    enabled: bool = True
    transport: str = "auto"
    model: str = "claude-sonnet-4-6"
    cadence_minutes: int = 30
    notes: str = ""


class ImprovementUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    transport: Optional[str] = None
    model: Optional[str] = None
    cadence_minutes: Optional[int] = None
    notes: Optional[str] = None


@app.get("/agents/{agent_id}/improvement", response_model=ImprovementConfigView)
def get_improvement_endpoint(agent_id: str) -> ImprovementConfigView:
    """Per-agent self-improvement brain config (P3.2)."""
    from runtime.agents.improvement import load
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    cfg = load(agent_id)
    return ImprovementConfigView(**cfg.to_dict())


@app.put("/agents/{agent_id}/improvement", response_model=ImprovementConfigView)
def put_improvement_endpoint(
    agent_id: str, payload: ImprovementUpdateRequest,
) -> ImprovementConfigView:
    """Update the per-agent improvement config. Partial body merges
    over existing values."""
    from runtime.agents.improvement import ImprovementConfig, load, save
    from runtime.agents.registry import get_agent
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")

    cfg = load(agent_id)
    body = payload.model_dump(exclude_unset=True)
    if "enabled" in body:
        cfg.enabled = bool(body["enabled"])
    if "transport" in body:
        cfg.transport = str(body["transport"] or "auto")
    if "model" in body and body["model"]:
        cfg.model = str(body["model"])
    if "cadence_minutes" in body and body["cadence_minutes"] is not None:
        cfg.cadence_minutes = max(0, int(body["cadence_minutes"]))
    if "notes" in body:
        cfg.notes = str(body["notes"] or "")

    # Re-validate via from_dict so the transport normalizes
    cfg = ImprovementConfig.from_dict(cfg.to_dict())
    save(agent_id, cfg)
    return ImprovementConfigView(**cfg.to_dict())


class AgentSecretsStatus(BaseModel):
    """Per-provider key status — never carries the raw key.

    ``source`` is one of ``per-agent`` | ``global`` | ``unset``.
    """
    set: bool
    source: str
    mask: Optional[str] = None
    var: str


class AgentSecretsResponse(BaseModel):
    agent_id: str
    providers: dict[str, AgentSecretsStatus]


class AgentSecretsUpdateRequest(BaseModel):
    """Pass an empty string as the value to remove a key. Omitted
    providers stay unchanged. Only the raw key value is accepted; the
    server picks the canonical env var name per provider."""
    anthropic: Optional[str] = None
    openai: Optional[str] = None


@app.get("/agents/{agent_id}/secrets", response_model=AgentSecretsResponse)
def get_agent_secrets_endpoint(agent_id: str) -> AgentSecretsResponse:
    from runtime.agents.registry import get_agent
    from runtime.agents.secrets import status
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    providers = {k: AgentSecretsStatus(**v) for k, v in status(agent_id).items()}
    return AgentSecretsResponse(agent_id=agent_id, providers=providers)


@app.put("/agents/{agent_id}/secrets", response_model=AgentSecretsResponse)
def put_agent_secrets_endpoint(
    agent_id: str, payload: AgentSecretsUpdateRequest,
) -> AgentSecretsResponse:
    """Rotate per-agent keys. Empty string removes the key (falls back
    to global). Omitted providers stay untouched.
    """
    from runtime.agents.registry import get_agent
    from runtime.agents.secrets import PROVIDERS, save_secrets, status
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")

    to_write: dict[str, str] = {}
    body = payload.model_dump(exclude_unset=True)
    for provider, value in body.items():
        if provider not in PROVIDERS:
            raise HTTPException(status_code=400, detail=f"unknown_provider: {provider}")
        if value is None:
            continue
        canonical = PROVIDERS[provider][0]
        to_write[canonical] = value
    if to_write:
        save_secrets(agent_id, to_write)

    providers = {k: AgentSecretsStatus(**v) for k, v in status(agent_id).items()}
    return AgentSecretsResponse(agent_id=agent_id, providers=providers)


@app.patch("/agents/{agent_id}", response_model=AgentSummary)
def update_agent_endpoint(agent_id: str, payload: AgentUpdateRequest) -> AgentSummary:
    """Mutate an agent's metadata. When ``model`` is provided, propagates
    to the agent's ``pipeline/route.yaml`` so the next /run uses it. If
    the agent is currently active, the change is also reflected in the
    live ``agent/`` dir on the next activate."""
    from runtime.agents.registry import get_registry, update_agent
    try:
        meta = update_agent(
            agent_id,
            name=payload.name,
            description=payload.description,
            model=payload.model,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent_not_found: {agent_id}")
    return _summarize(meta, active_id=get_registry().active)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("runtime.server:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
