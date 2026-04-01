"""Trace scanner — detects issues in LLM traces using heuristics + agents.

Two-layer detection:
1. Heuristic checks (fast, no LLM call): latency_spike, cost_anomaly,
   format_violation, incomplete_response
2. LLM agent checks (semantic): hallucination, refusal, safety, quality_regression

Results are stored as memory entries for historical analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .memory_store import MemoryEntry, MemoryStore, get_memory_store
from .runner import AgentRunner

logger = logging.getLogger(__name__)

ISSUE_CATEGORY = "trace_issue"
SCAN_CATEGORY = "trace_scan"
FEEDBACK_CATEGORY = "user_feedback"
AUTO_EVAL_CATEGORY = "auto_evaluation"

# Heuristic thresholds
LATENCY_SPIKE_ZSCORE = 2.0  # standard deviations above mean
COST_ANOMALY_ZSCORE = 2.5
MIN_OUTPUT_TOKENS = 5  # below this → incomplete_response candidate
REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i'm not able", "as an ai", "i apologize, but",
    "i'm sorry, but i can't", "i must decline",
]


@dataclass
class TraceIssue:
    """A detected issue in a trace."""

    id: str
    trace_id: str
    type: str  # hallucination | refusal | safety | quality_regression | latency_spike | cost_anomaly | format_violation | incomplete_response
    severity: str  # high | medium | low
    title: str
    description: str
    ai_confidence: float
    model_id: str
    trace_input: str
    trace_output: str
    detected_at: str
    resolved: bool = False
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "type": self.type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "ai_confidence": self.ai_confidence,
            "model_id": self.model_id,
            "trace_input": self.trace_input,
            "trace_output": self.trace_output,
            "detected_at": self.detected_at,
            "resolved": self.resolved,
            "suggested_action": self.suggested_action,
        }


@dataclass
class ScanState:
    """Tracks a running or completed scan."""

    scan_id: str
    status: str = "running"  # running | completed | failed
    traces_scanned: int = 0
    issues_found: int = 0
    started_at: str = ""
    completed_at: Optional[str] = None


# In-memory scan tracking (lightweight — scans are short-lived)
_active_scans: dict[str, ScanState] = {}


def _compute_stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, stdev) for a list of values. Returns (0, 1) for < 2 values."""
    if len(values) < 2:
        return (values[0] if values else 0.0, 1.0)
    return statistics.mean(values), statistics.stdev(values)


def _zscore(value: float, mean: float, stdev: float) -> float:
    if stdev == 0:
        return 0.0
    return (value - mean) / stdev


# ---------------------------------------------------------------------------
# Heuristic detectors (fast, no LLM)
# ---------------------------------------------------------------------------


def detect_latency_spike(
    trace: dict, latency_mean: float, latency_stdev: float,
) -> Optional[TraceIssue]:
    """Flag traces with latency > N standard deviations above mean."""
    latency = float(trace.get("latency_ms", 0))
    if latency <= 0:
        return None

    z = _zscore(latency, latency_mean, latency_stdev)
    if z < LATENCY_SPIKE_ZSCORE:
        return None

    severity = "high" if z > 3.5 else "medium" if z > 2.5 else "low"
    confidence = min(0.99, 0.6 + z * 0.1)

    return TraceIssue(
        id=str(uuid.uuid4()),
        trace_id=trace.get("request_id", ""),
        type="latency_spike",
        severity=severity,
        title=f"Latency spike: {latency:.0f}ms",
        description=f"Response took {latency:.0f}ms ({z:.1f} std devs above mean of {latency_mean:.0f}ms).",
        ai_confidence=round(confidence, 2),
        model_id=trace.get("selected_model", ""),
        trace_input=str(trace.get("input_text", ""))[:500],
        trace_output=str(trace.get("output_text", ""))[:500],
        detected_at=datetime.now(timezone.utc).isoformat(),
        suggested_action="Investigate model or provider latency. Consider timeout tuning or fallback routing.",
    )


def detect_cost_anomaly(
    trace: dict, cost_mean: float, cost_stdev: float,
) -> Optional[TraceIssue]:
    """Flag traces with cost > N standard deviations above mean."""
    cost = float(trace.get("total_cost_usd", 0))
    if cost <= 0:
        return None

    z = _zscore(cost, cost_mean, cost_stdev)
    if z < COST_ANOMALY_ZSCORE:
        return None

    severity = "high" if z > 4 else "medium"
    confidence = min(0.99, 0.6 + z * 0.08)

    return TraceIssue(
        id=str(uuid.uuid4()),
        trace_id=trace.get("request_id", ""),
        type="cost_anomaly",
        severity=severity,
        title=f"Cost anomaly: ${cost:.4f}",
        description=f"Request cost ${cost:.4f} ({z:.1f} std devs above mean of ${cost_mean:.4f}).",
        ai_confidence=round(confidence, 2),
        model_id=trace.get("selected_model", ""),
        trace_input=str(trace.get("input_text", ""))[:500],
        trace_output=str(trace.get("output_text", ""))[:500],
        detected_at=datetime.now(timezone.utc).isoformat(),
        suggested_action="Review token usage. Consider a smaller model or input truncation.",
    )


def detect_incomplete_response(trace: dict) -> Optional[TraceIssue]:
    """Flag traces with very short or empty output."""
    tokens_out = int(trace.get("tokens_out", 0))
    output = str(trace.get("output_text", "")).strip()

    if tokens_out >= MIN_OUTPUT_TOKENS and len(output) > 20:
        return None

    # Skip error responses — those are expected to be short
    if int(trace.get("is_error", 0)):
        return None

    severity = "high" if not output else "medium"
    confidence = 0.95 if not output else 0.75

    return TraceIssue(
        id=str(uuid.uuid4()),
        trace_id=trace.get("request_id", ""),
        type="incomplete_response",
        severity=severity,
        title="Incomplete or empty response",
        description=f"Output has {tokens_out} tokens and {len(output)} chars. May indicate a truncated or failed generation.",
        ai_confidence=confidence,
        model_id=trace.get("selected_model", ""),
        trace_input=str(trace.get("input_text", ""))[:500],
        trace_output=output[:500],
        detected_at=datetime.now(timezone.utc).isoformat(),
        suggested_action="Check max_tokens setting. Verify the model completed generation.",
    )


def detect_format_violation(trace: dict) -> Optional[TraceIssue]:
    """Flag traces where JSON output was expected but not produced."""
    input_text = str(trace.get("input_text", "")).lower()
    output_text = str(trace.get("output_text", "")).strip()

    # Only check if the input suggests structured output was expected
    json_indicators = ["json", "```json", '"type":', "respond with json", "return json", "valid json"]
    expects_json = any(ind in input_text for ind in json_indicators)
    if not expects_json:
        return None

    # Try to parse as JSON
    cleaned = output_text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        json.loads(cleaned.strip())
        return None  # Valid JSON — no issue
    except (json.JSONDecodeError, ValueError):
        pass

    return TraceIssue(
        id=str(uuid.uuid4()),
        trace_id=trace.get("request_id", ""),
        type="format_violation",
        severity="medium",
        title="Expected JSON output not produced",
        description="The input requested JSON-formatted output, but the response is not valid JSON.",
        ai_confidence=0.80,
        model_id=trace.get("selected_model", ""),
        trace_input=str(trace.get("input_text", ""))[:500],
        trace_output=output_text[:500],
        detected_at=datetime.now(timezone.utc).isoformat(),
        suggested_action="Add output format instructions or use structured output mode.",
    )


# ---------------------------------------------------------------------------
# Scanner orchestrator
# ---------------------------------------------------------------------------


class TraceScanner:
    """Orchestrates heuristic + LLM-based trace scanning.

    Scan flow:
    1. Fetch recent traces from ClickHouse
    2. Compute population stats (latency, cost) for z-score thresholds
    3. Run heuristic detectors on all traces (fast)
    4. Run LLM trace_scanner agent on traces with content (batched)
    5. Store all issues as memory entries
    """

    def __init__(
        self,
        engine_url: Optional[str] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.engine_url = engine_url
        self.memory_store = memory_store or get_memory_store()
        self._runner: Optional[AgentRunner] = None

    def _get_runner(self) -> AgentRunner:
        if self._runner is None:
            self._runner = AgentRunner(
                engine_url=self.engine_url,
                memory_store=self.memory_store,
                record_memory=True,
            )
        return self._runner

    async def scan(
        self,
        scan_id: str,
        days: int = 7,
        limit: int = 100,
    ) -> list[TraceIssue]:
        """Run a full scan and return detected issues."""
        state = ScanState(
            scan_id=scan_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        _active_scans[scan_id] = state

        try:
            traces = await self._fetch_traces(days=days, limit=limit)
            state.traces_scanned = len(traces)

            if not traces:
                state.status = "completed"
                state.completed_at = datetime.now(timezone.utc).isoformat()
                return []

            issues: list[TraceIssue] = []

            # Layer 1: heuristic checks
            heuristic_issues = self._run_heuristics(traces)
            issues.extend(heuristic_issues)

            # Layer 2: LLM agent checks on traces with content
            content_traces = [
                t for t in traces
                if str(t.get("input_text", "")).strip()
                and str(t.get("output_text", "")).strip()
            ]
            if content_traces:
                agent_issues = await self._run_agent_checks(content_traces)
                issues.extend(agent_issues)

            # Store issues in memory
            self._store_issues(issues, scan_id)

            # Auto-generate eval cases from high/medium issues
            eval_count = 0
            if issues:
                eval_entries = await self._generate_eval_cases(issues, scan_id)
                eval_count = len(eval_entries)
                logger.info(
                    f"Auto-generated {eval_count} eval cases from "
                    f"{len(issues)} issues"
                )

            state.issues_found = len(issues)
            state.status = "completed"
            state.completed_at = datetime.now(timezone.utc).isoformat()

            # Store scan summary in memory
            self._store_scan_summary(state, issues, eval_count=eval_count)

            return issues

        except Exception as e:
            logger.error(f"Scan {scan_id} failed: {e}")
            state.status = "failed"
            state.completed_at = datetime.now(timezone.utc).isoformat()
            raise

    async def _fetch_traces(self, days: int = 7, limit: int = 100) -> list[dict]:
        """Fetch recent traces from ClickHouse via the toolkit."""
        from .tools import ToolKit

        toolkit = ToolKit(engine_url=self.engine_url)
        result = await toolkit.query_traces(days=days, limit=limit)
        return result.get("traces", [])

    def _run_heuristics(self, traces: list[dict]) -> list[TraceIssue]:
        """Run all heuristic detectors against traces."""
        issues: list[TraceIssue] = []

        # Compute population stats for z-score detectors
        latencies = [float(t.get("latency_ms", 0)) for t in traces if float(t.get("latency_ms", 0)) > 0]
        costs = [float(t.get("total_cost_usd", 0)) for t in traces if float(t.get("total_cost_usd", 0)) > 0]

        lat_mean, lat_stdev = _compute_stats(latencies)
        cost_mean, cost_stdev = _compute_stats(costs)

        for trace in traces:
            for detector in [
                lambda t: detect_latency_spike(t, lat_mean, lat_stdev),
                lambda t: detect_cost_anomaly(t, cost_mean, cost_stdev),
                detect_incomplete_response,
                detect_format_violation,
            ]:
                issue = detector(trace)
                if issue:
                    issues.append(issue)

        return issues

    def _build_feedback_context(self) -> str:
        """Build a concise feedback context string from past false positives."""
        feedback_entries = self.memory_store.query(
            agent="trace_scanner",
            category=FEEDBACK_CATEGORY,
            limit=50,
        )
        if not feedback_entries:
            return ""

        lines = ["## Past False Positives (adjust your analysis accordingly)\n"]
        for fb in feedback_entries[:20]:
            ev = fb.evaluation
            reason = ev.get("reason", "")
            reason_str = f" Reason: {reason}" if reason else ""
            lines.append(
                f"- Issue type '{ev.get('issue_type', '?')}' on model "
                f"'{ev.get('model_id', '?')}' was dismissed as false positive. "
                f"Input preview: {ev.get('trace_input_preview', '')[:100]}"
                f"{reason_str}"
            )
        return "\n".join(lines) + "\n\n"

    async def _run_agent_checks(self, traces: list[dict]) -> list[TraceIssue]:
        """Run the trace_scanner LLM agent on traces with content."""
        runner = self._get_runner()
        issues: list[TraceIssue] = []

        # Load past feedback to inject as context
        feedback_context = self._build_feedback_context()

        for trace in traces:
            input_text = str(trace.get("input_text", ""))[:1000]
            output_text = str(trace.get("output_text", ""))[:1000]
            model_id = trace.get("selected_model", "")
            trace_id = trace.get("request_id", "")

            user_input = (
                f"{feedback_context}"
                f"Model: {model_id}\n\n"
                f"## Input\n{input_text}\n\n"
                f"## Output\n{output_text}"
            )

            try:
                result = await runner.run("trace_scanner", user_input)
                agent_issues = result.data.get("issues", [])

                for ai in agent_issues:
                    if not isinstance(ai, dict):
                        continue
                    issue_type = ai.get("type", "")
                    if issue_type not in (
                        "hallucination", "refusal", "safety", "quality_regression",
                    ):
                        continue

                    issues.append(TraceIssue(
                        id=str(uuid.uuid4()),
                        trace_id=trace_id,
                        type=issue_type,
                        severity=ai.get("severity", "medium"),
                        title=ai.get("title", f"{issue_type} detected"),
                        description=ai.get("description", ""),
                        ai_confidence=float(ai.get("confidence", 0.5)),
                        model_id=model_id,
                        trace_input=input_text[:500],
                        trace_output=output_text[:500],
                        detected_at=datetime.now(timezone.utc).isoformat(),
                        suggested_action=ai.get("suggested_action", ""),
                    ))
            except Exception as e:
                logger.warning(f"Agent scan failed for trace {trace_id}: {e}")

        return issues

    def _store_issues(self, issues: list[TraceIssue], scan_id: str) -> None:
        """Store each issue as a memory entry."""
        for issue in issues:
            entry = MemoryEntry(
                id=issue.id,
                agent="trace_scanner",
                category=ISSUE_CATEGORY,
                created_at=issue.detected_at,
                body=self._issue_to_body(issue),
                model=issue.model_id,
                tags=[issue.type, issue.severity, f"scan:{scan_id}"],
                evaluation={
                    "type": issue.type,
                    "severity": issue.severity,
                    "confidence": issue.ai_confidence,
                    "resolved": issue.resolved,
                    "trace_id": issue.trace_id,
                    "title": issue.title,
                    "description": issue.description,
                    "trace_input": issue.trace_input,
                    "trace_output": issue.trace_output,
                    "suggested_action": issue.suggested_action,
                },
            )
            try:
                self.memory_store.save(entry)
            except Exception as e:
                logger.debug(f"Failed to save issue to memory: {e}")

    async def _generate_eval_cases(
        self, issues: list[TraceIssue], scan_id: str,
    ) -> list[MemoryEntry]:
        """Auto-generate eval cases from high/medium severity issues."""
        runner = self._get_runner()
        entries: list[MemoryEntry] = []

        qualifying = [i for i in issues if i.severity in ("high", "medium")]
        for issue in qualifying:
            user_input = (
                f"## Detected Issue\n\n"
                f"- **Type:** {issue.type}\n"
                f"- **Severity:** {issue.severity}\n"
                f"- **Title:** {issue.title}\n"
                f"- **Description:** {issue.description}\n"
                f"- **Model:** {issue.model_id}\n"
                f"- **Suggested action:** {issue.suggested_action}\n\n"
                f"## Original Trace Input\n```\n{issue.trace_input[:500]}\n```\n\n"
                f"## Original Trace Output\n```\n{issue.trace_output[:500]}\n```"
            )
            try:
                result = await runner.run("eval_generator", user_input)
                eval_case = result.data.get("eval_case", {})
                rationale = result.data.get("rationale", "")

                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    agent="eval_generator",
                    category=AUTO_EVAL_CATEGORY,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    body=(
                        f"## Auto-Generated Eval Case\n\n"
                        f"**Source issue:** {issue.id} ({issue.type})\n"
                        f"**Check type:** {eval_case.get('check_type', 'unknown')}\n\n"
                        f"### Input\n{eval_case.get('input', '')}\n\n"
                        f"### Expected Behavior\n{eval_case.get('expected_behavior', '')}\n\n"
                        f"### Rationale\n{rationale}"
                    ),
                    model=issue.model_id,
                    tags=[
                        issue.type,
                        issue.severity,
                        f"source:{issue.id}",
                        f"scan:{scan_id}",
                    ],
                    evaluation={
                        "eval_case": eval_case,
                        "rationale": rationale,
                        "source_issue_id": issue.id,
                        "source_issue_type": issue.type,
                    },
                )
                self.memory_store.save(entry)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to generate eval case for issue {issue.id}: {e}")

        return entries

    def _store_scan_summary(self, state: ScanState, issues: list[TraceIssue], eval_count: int = 0) -> None:
        """Store scan summary as a memory entry for historical tracking."""
        type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        for issue in issues:
            type_counts[issue.type] = type_counts.get(issue.type, 0) + 1
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        body_lines = [
            "## Scan Summary\n",
            f"- **Traces scanned:** {state.traces_scanned}",
            f"- **Issues found:** {state.issues_found}",
            f"- **Eval cases generated:** {eval_count}",
            f"- **Duration:** {state.started_at} to {state.completed_at}",
            "",
            "## Issues by Type\n",
        ]
        for t, c in sorted(type_counts.items()):
            body_lines.append(f"- **{t}:** {c}")
        body_lines.append("\n## Issues by Severity\n")
        for s, c in sorted(severity_counts.items()):
            body_lines.append(f"- **{s}:** {c}")

        entry = MemoryEntry(
            id=state.scan_id,
            agent="trace_scanner",
            category=SCAN_CATEGORY,
            created_at=state.started_at,
            body="\n".join(body_lines),
            tags=["scan_summary"],
            evaluation={
                "traces_scanned": state.traces_scanned,
                "issues_found": state.issues_found,
                "eval_cases_generated": eval_count,
                "type_counts": type_counts,
                "severity_counts": severity_counts,
            },
        )
        try:
            self.memory_store.save(entry)
        except Exception as e:
            logger.debug(f"Failed to save scan summary: {e}")

    @staticmethod
    def _issue_to_body(issue: TraceIssue) -> str:
        lines = [
            f"## {issue.title}\n",
            f"**Type:** {issue.type} | **Severity:** {issue.severity} | **Confidence:** {issue.ai_confidence:.0%}\n",
            f"**Model:** {issue.model_id}\n",
        ]
        if issue.description:
            lines.append(f"\n{issue.description}\n")
        if issue.trace_input:
            lines.append(f"\n### Input\n```\n{issue.trace_input[:300]}\n```\n")
        if issue.trace_output:
            lines.append(f"\n### Output\n```\n{issue.trace_output[:300]}\n```\n")
        if issue.suggested_action:
            lines.append(f"\n**Suggested action:** {issue.suggested_action}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Issue management (read / resolve via memory)
# ---------------------------------------------------------------------------


def list_issues(
    memory_store: Optional[MemoryStore] = None,
    severity: Optional[str] = None,
    issue_type: Optional[str] = None,
    resolved: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Query stored issues from memory with optional filters."""
    store = memory_store or get_memory_store()

    # Get all trace_issue entries
    tags_filter = []
    if severity:
        tags_filter.append(severity)
    if issue_type:
        tags_filter.append(issue_type)

    entries = store.query(
        agent="trace_scanner",
        category=ISSUE_CATEGORY,
        tags=tags_filter or None,
        limit=500,
    )

    issues = []
    for entry in entries:
        ev = entry.evaluation
        if resolved is not None:
            if ev.get("resolved", False) != resolved:
                continue

        issues.append({
            "id": entry.id,
            "trace_id": ev.get("trace_id", ""),
            "type": ev.get("type", ""),
            "severity": ev.get("severity", ""),
            "title": ev.get("title", ""),
            "description": ev.get("description", ""),
            "ai_confidence": ev.get("confidence", 0),
            "model_id": entry.model,
            "trace_input": ev.get("trace_input", ""),
            "trace_output": ev.get("trace_output", ""),
            "detected_at": entry.created_at,
            "resolved": ev.get("resolved", False),
            "suggested_action": ev.get("suggested_action", ""),
        })

    return issues


def resolve_issue(
    issue_id: str,
    memory_store: Optional[MemoryStore] = None,
) -> bool:
    """Mark an issue as resolved in memory."""
    store = memory_store or get_memory_store()
    entry = store.get(issue_id)
    if entry is None or entry.category != ISSUE_CATEGORY:
        return False

    entry.evaluation["resolved"] = True
    entry.tags = [t for t in entry.tags if t != "unresolved"]
    entry.tags.append("resolved")

    # Re-save with updated evaluation
    store.save(entry)
    return True


def dismiss_issue(
    issue_id: str,
    reason: str = "",
    memory_store: Optional[MemoryStore] = None,
) -> tuple[bool, str]:
    """Dismiss an issue as a false positive and store user feedback.

    Returns (success, feedback_entry_id).
    """
    store = memory_store or get_memory_store()
    entry = store.get(issue_id)
    if entry is None or entry.category != ISSUE_CATEGORY:
        return False, ""

    # Update the original issue entry
    entry.evaluation["dismissed"] = True
    entry.evaluation["resolved"] = True
    entry.tags = [t for t in entry.tags if t not in ("unresolved",)]
    entry.tags.extend(["dismissed", "false_positive"])
    store.save(entry)

    # Create a separate feedback entry for future learning
    issue_type = entry.evaluation.get("type", "unknown")
    model_id = entry.model or "unknown"
    trace_input = entry.evaluation.get("trace_input", "")

    feedback_id = str(uuid.uuid4())
    reason_text = f"\n\n**User reason:** {reason}" if reason else ""
    feedback_entry = MemoryEntry(
        id=feedback_id,
        agent="trace_scanner",
        category=FEEDBACK_CATEGORY,
        created_at=datetime.now(timezone.utc).isoformat(),
        body=(
            f"## False Positive Dismissal\n\n"
            f"- **Source issue:** {issue_id}\n"
            f"- **Issue type:** {issue_type}\n"
            f"- **Model:** {model_id}\n"
            f"- **Original title:** {entry.evaluation.get('title', '')}\n"
            f"- **Input preview:** {trace_input[:200]}"
            f"{reason_text}"
        ),
        model=model_id,
        tags=["false_positive", issue_type, model_id],
        evaluation={
            "source_issue_id": issue_id,
            "issue_type": issue_type,
            "model_id": model_id,
            "feedback_type": "false_positive",
            "trace_input_preview": trace_input[:200],
            "reason": reason,
        },
    )
    store.save(feedback_entry)

    return True, feedback_id


def get_scan_state(scan_id: str) -> Optional[ScanState]:
    """Get the state of an active or recent scan."""
    return _active_scans.get(scan_id)
