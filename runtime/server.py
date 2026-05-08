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
    promoted_at: Optional[str] = None
    ledger_entry_id: Optional[str] = None
    proposal_source: Optional[str] = None


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
        promoted_at=lesson.promoted_at,
        ledger_entry_id=lesson.ledger_entry_id,
        proposal_source=lesson.proposal_source,
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
