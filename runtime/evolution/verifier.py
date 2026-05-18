"""Trajectory-level eval grading (AHE paper §3.3, Decision Observability).

The rollout produces ``TaskOutcome``s; that alone tells us whether the
pipeline *ran*, not whether the agent *behaved correctly*. Per the
paper, evaluation is trajectory-based: each (task, response) pair is
judged against the agent's contract (system_prompt + hard rules) and
the prior iteration's claimed_fixes / at_risk_regressions.

This module exposes :func:`verify_trajectory`, called once per case
from :mod:`runtime.evolution.bridge`. It returns a list of
``rubric_result`` dicts shaped exactly like the experiments-runner
output — the same shape the UI's Evals tab already renders.

Cost: one Anthropic call per trajectory (~$0.005 with sonnet-4.6).
For an iteration with 4 tasks × k=2 = 8 trajectories that's ~$0.04.
Skipped silently when ``anthropic_key`` is missing — the bridge
falls back to the mechanical ``pipeline_succeeded`` rubric only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional


logger = logging.getLogger("runtime.evolution.verifier")

_VERIFIER_MODEL_DEFAULT = "claude-sonnet-4-6"
_VERIFIER_MAX_TOKENS = 1200
_RESPONSE_TRUNCATE = 4000
_PROMPT_TRUNCATE = 3000
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def verify_trajectory(
    *,
    task: str,
    response: str,
    success: bool,
    error: Optional[str],
    system_prompt: Optional[str],
    claimed_fixes: list[str],
    at_risk_regressions: list[str],
    anthropic_key: Optional[str],
    model: str = _VERIFIER_MODEL_DEFAULT,
) -> Optional[list[dict[str, Any]]]:
    """Grade one trajectory. Returns rubric_result list or ``None``
    on skip / failure (the caller keeps the mechanical default).

    Rubrics emitted:
      - ``trajectory_quality`` — overall passed/failed verdict + reasoning
      - ``follows_contract`` — does the response respect the system_prompt's persona/policy/rules
      - ``claim_<i>`` (per claimed_fix) — did the trajectory exhibit this fix?
      - ``risk_<i>`` (per at_risk_regression) — does the trajectory show this regression? (passed=NOT observed)
    """
    if not anthropic_key:
        return None
    if not response and not error:
        # Empty response and no error → nothing to grade beyond the
        # mechanical pipeline_succeeded rubric. Skip.
        return None
    try:
        from anthropic import Anthropic
    except Exception as exc:  # pragma: no cover — anthropic is required
        logger.warning("verifier: anthropic SDK missing (%s)", exc)
        return None

    prompt = _build_prompt(
        task=task,
        response=response,
        success=success,
        error=error,
        system_prompt=system_prompt,
        claimed_fixes=claimed_fixes,
        at_risk_regressions=at_risk_regressions,
    )

    try:
        client = Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model=model,
            max_tokens=_VERIFIER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("verifier: call failed (%s)", exc)
        return None

    text = ""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "") or ""

    parsed = _parse_verdict(text)
    if parsed is None:
        logger.info("verifier: could not parse verdict; raw=%s", text[:200])
        return None

    return _verdict_to_rubrics(
        parsed,
        claimed_fixes=claimed_fixes,
        at_risk_regressions=at_risk_regressions,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_prompt(
    *,
    task: str,
    response: str,
    success: bool,
    error: Optional[str],
    system_prompt: Optional[str],
    claimed_fixes: list[str],
    at_risk_regressions: list[str],
) -> str:
    contract = (system_prompt or "(no system prompt available)").strip()
    if len(contract) > _PROMPT_TRUNCATE:
        contract = contract[:_PROMPT_TRUNCATE] + "\n…[truncated]"
    resp_text = (response or "").strip()
    if len(resp_text) > _RESPONSE_TRUNCATE:
        resp_text = resp_text[:_RESPONSE_TRUNCATE] + "\n…[truncated]"

    claims_block = (
        "\n".join(f"  - {c}" for c in claimed_fixes)
        if claimed_fixes else "  (none — prior iteration claimed nothing)"
    )
    risks_block = (
        "\n".join(f"  - {r}" for r in at_risk_regressions)
        if at_risk_regressions else "  (none — prior iteration flagged no risks)"
    )

    error_block = ""
    if error:
        error_block = f"\nPipeline error: {error[:600]}"

    return (
        "You are the AHE Trajectory Verifier (paper §3.3, Decision "
        "Observability). Grade ONE agent trajectory against (a) its "
        "contract, (b) the prior iteration's claimed fixes, (c) the "
        "prior iteration's at-risk regressions.\n\n"
        "Output ONLY a JSON object — no preamble, no fences. Schema:\n"
        "{\n"
        '  "overall_passed": bool,\n'
        '  "overall_reasoning": str (≤200 chars),\n'
        '  "follows_contract": {"passed": bool, "reasoning": str},\n'
        '  "claim_verdicts": [{"index": int, "passed": bool, "reasoning": str}, ...],\n'
        '  "risk_verdicts":  [{"index": int, "observed": bool, "reasoning": str}, ...],\n'
        '  "severity": int (1=polish, 5=breaks contract)\n'
        "}\n\n"
        "Notes:\n"
        '- "follows_contract" = does the response respect the persona, hard rules, '
        "and policies in the contract?\n"
        '- "claim_verdicts[i]" = is the i-th claimed_fix EVIDENT in this trajectory? '
        "(passed=true if the fix is visible / honored)\n"
        '- "risk_verdicts[i]" = is the i-th at_risk_regression OBSERVED in this trajectory? '
        "(observed=true means the regression is happening — BAD)\n"
        '- "overall_passed" = false if follows_contract failed, any claim failed, '
        "or any risk was observed.\n\n"
        f"=== CONTRACT ===\n{contract}\n\n"
        f"=== PRIOR CLAIMED FIXES ===\n{claims_block}\n\n"
        f"=== PRIOR AT-RISK REGRESSIONS ===\n{risks_block}\n\n"
        f"=== TASK ===\n{task!r}\n\n"
        f"=== RESPONSE ===\n{resp_text}{error_block}\n\n"
        "Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# Parse + shape
# ---------------------------------------------------------------------------


def _parse_verdict(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from the model's reply."""
    if not text:
        return None
    text = text.strip()
    candidate = text
    if not text.startswith("{"):
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            return None
        candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _verdict_to_rubrics(
    verdict: dict[str, Any],
    *,
    claimed_fixes: list[str],
    at_risk_regressions: list[str],
) -> list[dict[str, Any]]:
    """Map the parsed verdict into the rubric_results shape the UI reads."""
    rubrics: list[dict[str, Any]] = []

    overall = bool(verdict.get("overall_passed", False))
    overall_reason = str(verdict.get("overall_reasoning", "") or "")[:300]
    severity = int(verdict.get("severity", 3) or 3)
    rubrics.append({
        "rubric": "trajectory_quality",
        "type": "llm_judged",
        "score": 1.0 if overall else 0.0,
        "passed": overall,
        "detail": f"[sev {severity}] {overall_reason}",
    })

    fc = verdict.get("follows_contract") or {}
    fc_passed = bool(fc.get("passed", False))
    rubrics.append({
        "rubric": "follows_contract",
        "type": "llm_judged",
        "score": 1.0 if fc_passed else 0.0,
        "passed": fc_passed,
        "detail": str(fc.get("reasoning", "") or "")[:300],
    })

    claims = verdict.get("claim_verdicts") or []
    claims_by_idx = {int(c.get("index", -1)): c for c in claims if isinstance(c, dict)}
    for i, claim_text in enumerate(claimed_fixes):
        c = claims_by_idx.get(i) or {}
        passed = bool(c.get("passed", False))
        rubrics.append({
            "rubric": f"claim_{i:02d}",
            "type": "claimed_fix",
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "detail": f"{claim_text[:100]} → {str(c.get('reasoning',''))[:200]}",
        })

    risks = verdict.get("risk_verdicts") or []
    risks_by_idx = {int(r.get("index", -1)): r for r in risks if isinstance(r, dict)}
    for i, risk_text in enumerate(at_risk_regressions):
        r = risks_by_idx.get(i) or {}
        # observed=True is BAD → rubric "passed" is the OPPOSITE
        observed = bool(r.get("observed", False))
        rubrics.append({
            "rubric": f"risk_{i:02d}",
            "type": "regression_check",
            "score": 0.0 if observed else 1.0,
            "passed": not observed,
            "detail": f"{risk_text[:100]} → {str(r.get('reasoning',''))[:200]}",
        })

    return rubrics
