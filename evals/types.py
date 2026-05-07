"""Eval data types.

Pydantic models for things parsed from YAML (Golden, Suite). Dataclasses for
pure-runtime values (RubricResult, EvalCase, Report). The split is deliberate:
Pydantic where we need validation, dataclass where we want speed and clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------- YAML-parsed (Pydantic) ----------


class HistoryMessage(BaseModel):
    role: str
    content: str


class GoldenInput(BaseModel):
    request: str
    history: list[HistoryMessage] = Field(default_factory=list)


class GoldenExpected(BaseModel):
    """All fields optional — different rubrics consume different shapes."""

    contains: list[str] = Field(default_factory=list)
    exact: Optional[str] = None
    category: Optional[str] = None
    must_succeed: bool = True


class Golden(BaseModel):
    id: str
    input: GoldenInput
    expected: GoldenExpected = Field(default_factory=GoldenExpected)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RubricSpec(BaseModel):
    """A rubric usage in a Suite — name + per-call params."""

    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class Suite(BaseModel):
    suite: str
    description: Optional[str] = None
    goldens: list[str]            # ids; loaded from evals/golden/<id>.yaml
    rubrics: list[RubricSpec]
    aggregation: str = "mean"     # extensible later (median, p95…)


# ---------- Runtime values (dataclass) ----------


@dataclass
class RubricResult:
    """One rubric's verdict on one golden."""

    rubric: str
    type: str
    score: float          # in [0.0, 1.0] for v0
    passed: bool
    detail: Optional[str] = None


@dataclass
class EvalCase:
    """One golden run end-to-end + every rubric scored against it."""

    golden_id: str
    request: str
    response: Optional[str]
    duration_ms: float
    success: bool
    error: Optional[str]
    trace_id: Optional[str]
    rubric_results: list[RubricResult] = field(default_factory=list)


@dataclass
class Report:
    """Full suite run — what gets written to evals/reports/."""

    suite: str
    agent_version: Optional[str]
    started_at: str        # ISO 8601
    finished_at: str
    cases: list[EvalCase]
    summary: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
