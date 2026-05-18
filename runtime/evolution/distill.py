"""Distill phase — turn a rollout into an evidence corpus.

Two layers (v1):

  - :func:`summarize_rollout` — flat pass/fail corpus, one entry per
    (task, run). Always built; cheap.
  - :func:`cluster_failures` — Agent Debugger Lite. Inline Anthropic
    call (no sandbox) that groups failures by root cause + ranks
    severity. Runs only when there ARE failures + an API key. Result
    is attached to :class:`Evidence.clusters` so the Evolve Agent
    sees a layered view per the AHE paper's Experience Observability
    invariant (§3.2).

The clustering is best-effort: if the LLM call fails or returns
unparseable JSON, we log + return the raw summary with no clusters.
Falling back to v0 behavior is always safe — the Evolve Agent's
prompt handles empty clusters gracefully.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from runtime.evolution.types import Evidence, EvidenceCluster, RolloutResult


logger = logging.getLogger("runtime.evolution.distill")


_SUMMARY_HEAD_CHARS = 240
_CLUSTER_MODEL_DEFAULT = "claude-sonnet-4-6"
_CLUSTER_MAX_TOKENS = 1500
_CLUSTER_MAX_FAILURES = 20  # don't ship a 200-failure prompt; trim
_JSON_BLOCK_RE = re.compile(r"\[[\s\S]*\]")


def summarize_rollout(rollout: RolloutResult) -> Evidence:
    """Pack a free-text summary of the rollout for the Evolve Agent.

    Per-(task, run) lines with PASS/FAIL + short response snippet. The
    aggregate count at the top distinguishes "all runs passed" from
    "flaky on some retries" so the Evolve Agent can read variance at
    a glance.
    """
    aggs = rollout.task_aggregates
    flaky = rollout.flaky_tasks
    head = (
        f"Rollout: {rollout.passed}/{rollout.total_tasks} tasks passed "
        f"(k={rollout.k} replays each)"
    )
    if flaky:
        head += f", flaky: {len(flaky)}"
    lines = [head + "."]

    for o in rollout.outcomes:
        verdict = "PASS" if (o.success and not o.error) else "FAIL"
        snippet = (o.error or o.response or "").strip().replace("\n", " ")
        if len(snippet) > _SUMMARY_HEAD_CHARS:
            snippet = snippet[:_SUMMARY_HEAD_CHARS] + "…"
        lines.append(f"[run {o.run_index}] {verdict} — task: {o.task!r}")
        if snippet:
            lines.append(f"    response: {snippet}")

    # Per-task aggregate roll-up at the bottom so the Evolve Agent
    # has a quick reference for "how many runs of task X passed".
    lines.append("")
    lines.append("Per-task aggregates:")
    for task, a in aggs.items():
        verdict = "MAJORITY-PASS" if a["majority_pass"] else "MAJORITY-FAIL"
        if a["flaky"]:
            verdict += " (flaky)"
        lines.append(
            f"  - {task!r}: {a['passed_runs']}/{a['total_runs']} runs → {verdict}"
        )

    return Evidence(rollout=rollout, summary="\n".join(lines))


def cluster_failures(
    evidence: Evidence,
    *,
    anthropic_key: Optional[str],
    model: str = _CLUSTER_MODEL_DEFAULT,
) -> Evidence:
    """Attach root-cause clusters to ``evidence``.

    Returns the same Evidence with ``clusters`` populated. No-op (and
    returns ``evidence`` unmodified) if any of:
      - no failures in the rollout (nothing to cluster)
      - no Anthropic key passed
      - the LLM call or JSON parse fails

    The contract with the model is structured JSON: ``[{root_cause,
    tasks, severity, notes}, ...]``. We do strict JSON extraction
    rather than tool-use; this keeps the dep surface minimal and
    avoids round-tripping through the tool-use loop for a one-shot
    classifier.
    """
    aggs = evidence.rollout.task_aggregates
    failing = sorted(
        task for task, a in aggs.items() if not a["majority_pass"]
    )
    if not failing:
        return evidence
    if not anthropic_key:
        logger.info("distill: skipping clustering — no Anthropic key")
        return evidence

    try:
        from anthropic import Anthropic
    except Exception as exc:  # pragma: no cover — anthropic is required
        logger.warning("distill: anthropic SDK missing (%s)", exc)
        return evidence

    failure_dump = _failure_dump(evidence.rollout, limit=_CLUSTER_MAX_FAILURES)
    prompt = (
        "You are the Agent Debugger from the AHE paper (§3.2). Cluster "
        "the following agent failures into root-cause groups. For each "
        "cluster:\n"
        "  - root_cause: 5–8 word name of the underlying issue\n"
        "  - tasks: list of failing task strings that belong here\n"
        "  - severity: integer 1..5 (5 = breaks agent's core contract, 1 = polish)\n"
        "  - notes: ≤140 chars on why this is the diagnosis\n\n"
        "Return ONLY a JSON array of these objects — no preamble, no "
        "code fences, no commentary. Example:\n"
        '[{"root_cause":"missing-persona","tasks":["t1","t2"],"severity":4,"notes":"..."}]\n\n'
        f"--- failures ---\n{failure_dump}"
    )

    try:
        client = Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model=model,
            max_tokens=_CLUSTER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("distill: clustering call failed (%s)", exc)
        return evidence

    text = ""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "") or ""
    clusters = _parse_clusters(text)
    if not clusters:
        logger.info("distill: clustering returned 0 valid clusters")
        return evidence

    evidence.clusters = clusters
    return evidence


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _failure_dump(rollout: RolloutResult, *, limit: int) -> str:
    lines = []
    seen = 0
    aggs = rollout.task_aggregates
    for o in rollout.outcomes:
        if seen >= limit:
            break
        if o.success and not o.error:
            continue
        task_agg = aggs.get(o.task, {})
        marker = "ALL-FAIL" if task_agg.get("passed_runs", 0) == 0 else "FLAKY-FAIL"
        snippet = (o.error or o.response or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(
            f"[{marker}] task: {o.task!r}\n  response/error: {snippet}"
        )
        seen += 1
    if seen == 0:
        return "(no failures)"
    return "\n\n".join(lines)


def _parse_clusters(text: str) -> list[EvidenceCluster]:
    """Best-effort JSON extraction from the LLM reply.

    Models sometimes wrap the array in ```json fences or a paragraph
    of preamble. The regex finds the first bracketed array; we then
    parse it strictly. Any cluster missing required fields is dropped
    rather than guessed.
    """
    if not text:
        return []
    candidates: list[str] = []
    # Try the whole text first (best case: model followed instructions).
    candidates.append(text.strip())
    # Then try extracting the first JSON array.
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(0))

    parsed: list[EvidenceCluster] = []
    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            root_cause = (entry.get("root_cause") or "").strip()
            tasks = entry.get("tasks") or []
            if not root_cause or not isinstance(tasks, list):
                continue
            try:
                severity = int(entry.get("severity", 3))
            except (TypeError, ValueError):
                severity = 3
            severity = max(1, min(5, severity))
            parsed.append(EvidenceCluster(
                root_cause=root_cause,
                tasks=[str(t) for t in tasks],
                severity=severity,
                notes=str(entry.get("notes") or "")[:200],
            ))
        if parsed:
            break
    return parsed
