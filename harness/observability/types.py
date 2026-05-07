"""Distilled corpus types — Session and Epoch.

These are the structured artifacts the introspection LLM and UI consume,
*never* raw traces directly. Designed to be:

  - Cheap (small JSON, ~1-3KB per session, ~5-15KB per epoch).
  - LLM-friendly (every entity has a free-form `summary` for one-line use).
  - Causal (links to ledger entries, trace ids, candidate ids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------- Session: one experiment / one trajectory ----------


@dataclass
class SessionAggregate:
    """Numerical aggregate over the traces of a session."""

    n_traces: int
    n_passed: int             # success=true
    n_failed: int             # success=false
    avg_latency_ms: float
    p95_latency_ms: float


@dataclass
class DistilledSession:
    """One coherent run — typically a candidate × suite execution.

    A session ties together:
      - a Proposal (if from the loop) or a live serving window
      - a candidate (if branched)
      - the traces produced
      - the ledger entries written
      - the verdicts/decisions taken
    """

    session_id: str
    kind: str                 # "experiment" | "live_serving" | "introspection"
    started_at: str           # ISO 8601
    ended_at: str

    # What happened
    candidate_id: Optional[str] = None
    parent_version: Optional[str] = None
    suite: Optional[str] = None
    proposal_source: Optional[str] = None    # "heuristic_sweep" | "claude_code" | …
    mutations: list[str] = field(default_factory=list)

    # Predictions (P7.4-7.5)
    prediction: Optional[dict[str, Any]] = None      # {rubric, expected_delta, rationale}
    actual: Optional[dict[str, Any]] = None          # {rubric, actual_delta}
    prediction_verified: Optional[bool] = None       # null if no prediction

    # Aggregate signals
    aggregate: Optional[SessionAggregate] = None
    overall_score: Optional[float] = None
    pass_rate: Optional[float] = None
    delta_overall: Optional[float] = None

    # Decision outcome
    final_decision: Optional[str] = None      # "auto_approve" | "queue_human" | "reject"
    promoted_version: Optional[str] = None
    blocking_critic: Optional[str] = None     # which critic blocked, if any

    # Causal links
    ledger_entries: list[str] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    lesson_id: Optional[str] = None

    # Free-form summary (the LLM-consumable rendering)
    summary: str = ""


# ---------- Epoch: time-bounded aggregation ----------


@dataclass
class EpochCounts:
    n_proposals: int
    n_branched: int
    n_promoted: int
    n_rolled_back: int
    n_rejected_by_critic: int
    n_rejected_by_policy: int


@dataclass
class TopEvent:
    """A high-signal event referenced from an epoch."""

    kind: str                 # "promotion" | "rollback" | "regression" | "rejection"
    when: str
    summary: str
    candidate_id: Optional[str] = None
    version: Optional[str] = None
    lesson_id: Optional[str] = None
    ledger_entry_id: Optional[str] = None


@dataclass
class DistilledEpoch:
    """Time-bounded summary of harness activity.

    epoch_id formats:
      - "day:2026-05-07"
      - "version:v0.0.2"
      - "incident:<id>"  (future)
    """

    epoch_id: str
    kind: str                 # "day" | "version" | "incident"
    started_at: str
    ended_at: str

    counts: EpochCounts
    top_events: list[TopEvent] = field(default_factory=list)

    # Aggregate quality
    avg_overall_score: Optional[float] = None
    avg_delta_when_promoted: Optional[float] = None

    # Pointers (children / referenced sessions)
    sessions: list[str] = field(default_factory=list)         # session_ids included
    referenced_versions: list[str] = field(default_factory=list)

    # Free-form summary
    summary: str = ""
