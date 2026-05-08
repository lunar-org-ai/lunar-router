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
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from runtime.compiler.builder import compile_agent
from runtime.compiler.loader import load_agent
from runtime.executor.pipeline import PipelineExecutor
from runtime.executor.tracing import write_trace
from runtime.protocols import Message

logger = logging.getLogger(__name__)


class HistoryMessage(BaseModel):
    role: str
    content: str


class RunRequest(BaseModel):
    request: str
    history: Optional[list[HistoryMessage]] = None


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
    cfg = load_agent("agent/agent.yaml")
    pipeline = compile_agent(cfg)
    executor = PipelineExecutor(pipeline)
    _state["cfg"] = cfg
    _state["executor"] = executor
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
    import json
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from ledger.writer import read_entries, read_lessons

    now = datetime.now(timezone.utc)
    today_str = now.date().isoformat()
    five_min_ago = now - timedelta(minutes=5)

    traces_dir = Path(__file__).resolve().parent.parent / "traces" / "raw"

    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows

    today_traces = _load_jsonl(traces_dir / f"{today_str}.jsonl")
    today_count = len(today_traces)

    active_5min = 0
    for t in today_traces:
        ts_raw = t.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if ts > five_min_ago:
                active_5min += 1
        except Exception:
            pass

    recent_traces: list[dict[str, Any]] = []
    if traces_dir.exists():
        for fp in sorted(traces_dir.glob("*.jsonl"), reverse=True)[:7]:
            recent_traces.extend(_load_jsonl(fp))

    resolution_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    if recent_traces:
        def _ok(t: dict[str, Any]) -> bool:
            if not t.get("response"):
                return False
            for s in t.get("stages") or []:
                if s.get("error"):
                    return False
            return True

        ok_count = sum(1 for t in recent_traces if _ok(t))
        resolution_rate = ok_count / len(recent_traces)
        avg_latency_ms = sum(float(t.get("duration_ms", 0) or 0) for t in recent_traces) / len(
            recent_traces
        )

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
        today_count=today_count,
        active_5min=active_5min,
        pending_review=pending,
        trust_score=trust_score,
        trust_score_delta_30d=trust_score_delta_30d,
        trust_history_30d=history,
        resolution_rate=resolution_rate,
        avg_latency_ms=avg_latency_ms,
        avg_cost_usd=None,
        csat=None,
        computed_at=now.isoformat(),
    )


@app.post("/run", response_model=RunResponse)
async def run(payload: RunRequest) -> RunResponse:
    executor: Optional[PipelineExecutor] = _state.get("executor")
    if not executor:
        raise HTTPException(status_code=503, detail="agent not yet loaded")

    history = [Message(role=m.role, content=m.content) for m in (payload.history or [])]
    _, rec = executor.run(payload.request, history=history)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("runtime.server:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
