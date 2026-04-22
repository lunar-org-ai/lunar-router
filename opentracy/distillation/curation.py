"""
Curation Service — score and select best candidates via LLM-as-Judge.

Replaces the cloud-based curation-worker Lambda.
Uses the Go engine for LLM-as-Judge scoring via ModelInvoker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..evals_common.model_invoker import ModelInvoker
from . import repository as repo
from opentracy._env import env

logger = logging.getLogger(__name__)

JUDGE_MODEL = os.environ.get("DISTILL_JUDGE_MODEL", "openai/gpt-4o-mini")

REWARD_PROMPT = """You are an expert evaluator assessing the quality of an AI assistant's response.

## User Input:
{input}

## Assistant Response:
{output}

## Evaluation Criteria:
Score each dimension from 0.0 to 1.0:

1. **Coherence** (0-1): Is the response logically structured and easy to follow?
2. **Helpfulness** (0-1): Does it address the user's needs effectively?
3. **Correctness** (0-1): Is the information accurate and factually correct?
4. **Format** (0-1): Is the formatting appropriate (code blocks, lists, etc.)?

## Instructions:
Evaluate the response and provide scores for each dimension.
Respond ONLY with valid JSON in this exact format:
{{
    "coherence": <float 0-1>,
    "helpfulness": <float 0-1>,
    "correctness": <float 0-1>,
    "format": <float 0-1>
}}
"""

WEIGHTS = {"coherence": 0.25, "helpfulness": 0.30, "correctness": 0.30, "format": 0.15}

def _data_dir() -> Path:
    """Resolve at call time, not import time — ``ot.distill()`` overrides
    the env var AFTER this module has already been imported."""
    return Path(env("DATA_DIR", "data"))


async def curate_candidates(
    job_id: str,
    tenant_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Score all candidates and select the best per prompt.

    The curated data is written as JSONL to the local filesystem
    for consumption by the training step.

    Returns:
        Summary dict with curation statistics.
    """
    invoker = ModelInvoker()
    judge_model = config.get("judge_model") or config.get("teacher_model") or JUDGE_MODEL

    # Verify judge model connectivity before starting (with retry for rate limits)
    for attempt in range(5):
        try:
            test_result = await asyncio.to_thread(
                invoker.invoke, judge_model, "Say OK", temperature=0.0, max_tokens=8,
            )
            if not test_result.get("output"):
                raise RuntimeError("Empty response from judge model")
            break
        except Exception as e:
            err = str(e)
            if ("429" in err or "rate" in err.lower()) and attempt < 4:
                delay = 10 * (2 ** attempt)
                logger.info("Judge model rate limited, waiting %ds (attempt %d/5)", delay, attempt + 1)
                repo.append_log(tenant_id, job_id, f"Rate limited by judge model, waiting {delay}s...")
                await asyncio.sleep(delay)
                continue
            elif "401" in err or "API key" in err or "Unauthorized" in err:
                raise RuntimeError(
                    f"Curation failed: the judge model '{judge_model}' requires an API key. "
                    f"Please add a valid API key in Settings > API Keys and try again."
                ) from e
            elif "402" in err or "insufficient" in err.lower() or "quota" in err.lower() or "credit" in err.lower():
                raise RuntimeError(
                    f"Curation failed: insufficient credits for '{judge_model}'. "
                    f"Please check your billing/credits with the provider and try again."
                ) from e
            elif "timeout" in err.lower() or "connect" in err.lower():
                raise RuntimeError(
                    f"Curation failed: could not connect to the provider for '{judge_model}'. "
                    f"Please check your network connection."
                ) from e
            else:
                raise RuntimeError(
                    f"Curation failed: judge model '{judge_model}' returned an error: {err}"
                ) from e

    # Fetch all candidates for this job
    candidates = repo.get_candidates(job_id, limit=100_000)
    if not candidates:
        raise ValueError(f"No candidates found for job {job_id}")

    logger.info("Curating %d candidates for job %s", len(candidates), job_id)
    repo.append_log(tenant_id, job_id, f"Curation: scoring {len(candidates)} candidates")

    # Group by prompt_id
    grouped: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        pid = c.get("prompt_id", "")
        grouped.setdefault(pid, []).append(c)

    total_prompts = len(grouped)
    curated: list[dict[str, str]] = []

    _update_phase_progress(tenant_id, job_id, "curation", 5, {
        "total_prompts": total_prompts,
    })

    for idx, (prompt_id, prompt_candidates) in enumerate(grouped.items()):
        best = await _select_best_candidate(
            invoker, judge_model, prompt_id, prompt_candidates
        )
        if best:
            curated.append(best)

        # Progress update (5% → 95%)
        pct = 5 + int((idx + 1) / total_prompts * 90)
        if (idx + 1) % 10 == 0 or idx == total_prompts - 1:
            _update_phase_progress(tenant_id, job_id, "curation", pct, {
                "samples_curated": len(curated),
                "prompts_processed": idx + 1,
                "total_prompts": total_prompts,
            })

    # Write curated JSONL to local filesystem
    job_dir = _data_dir() / "distillation" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    curated_path = job_dir / "curated.jsonl"

    with open(curated_path, "w") as f:
        for example in curated:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    _update_phase_progress(tenant_id, job_id, "curation", 100, {
        "samples_curated": len(curated),
        "prompts_processed": total_prompts,
        "total_prompts": total_prompts,
    })
    repo.append_log(
        tenant_id, job_id,
        f"Curation complete: {len(curated)} examples from {len(candidates)} candidates",
    )

    return {
        "total_candidates": len(candidates),
        "total_prompts": total_prompts,
        "curated_examples": len(curated),
        "curated_path": str(curated_path),
    }


async def _select_best_candidate(
    invoker: ModelInvoker,
    judge_model: str,
    prompt_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Score candidates and return the best one as a training example."""
    best_score = -1.0
    best_candidate = None

    for candidate in candidates:
        prompt_text = candidate.get("prompt", "")
        response_text = candidate.get("response", "")

        score = 0.5
        for attempt in range(5):
            try:
                score = await asyncio.to_thread(
                    _score_response, invoker, judge_model, prompt_text, response_text
                )
                break
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower()) and attempt < 4:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                logger.warning("Failed to score candidate %s: %s", candidate.get("candidate_id", ""), e)
                break

        # Update score in ClickHouse (non-blocking)
        try:
            repo.insert_candidate({
                "candidate_id": candidate.get("candidate_id", ""),
                "job_id": candidate.get("job_id", ""),
                "tenant_id": candidate.get("tenant_id", "default"),
                "prompt_id": prompt_id,
                "candidate_idx": candidate.get("candidate_idx", 0),
                "prompt": prompt_text,
                "system_prompt": candidate.get("system_prompt", ""),
                "response": response_text,
                "model": candidate.get("model", ""),
                "temperature": candidate.get("temperature", 0.8),
                "score": score,
                "selected": 1 if score > best_score else 0,
                "usage": candidate.get("usage", {}),
            })
        except Exception as e:
            logger.warning("Failed to update candidate score: %s", e)

        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        return None

    # Build training example — include tool_calls if present
    example: dict[str, str] = {
        "prompt": best_candidate.get("prompt", ""),
        "response": best_candidate.get("response", ""),
    }

    # Preserve tool_calls and system_prompt for chat-template formatting
    system_prompt = best_candidate.get("system_prompt", "")
    tool_calls = best_candidate.get("tool_calls", "")
    tools = best_candidate.get("tools", "")

    if system_prompt:
        example["system_prompt"] = system_prompt
    if tool_calls:
        example["tool_calls"] = tool_calls
    if tools:
        example["tools"] = tools

    # Fallback plain text (used by raw SFT if no chat template)
    example["text"] = f"{example['prompt']}\n{example['response']}"

    return example


def _score_response(
    invoker: ModelInvoker,
    judge_model: str,
    prompt: str,
    response: str,
) -> float:
    """Invoke LLM-as-Judge to score a single response."""
    eval_prompt = REWARD_PROMPT.format(
        input=prompt[:4000],
        output=response[:4000],
    )

    result = invoker.invoke(
        judge_model,
        eval_prompt,
        temperature=0.3,
        max_tokens=256,
    )

    return _parse_judge_response(result.get("output", ""))


def _parse_judge_response(text: str) -> float:
    """Parse JSON from judge into a weighted 0-1 score."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        result = json.loads(text)
        total = 0.0
        for metric, weight in WEIGHTS.items():
            val = result.get(metric, 0.5)
            total += max(0.0, min(1.0, float(val))) * weight
        return total
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Failed to parse judge response: %s", text[:200])
        return 0.5


def _update_phase_progress(
    tenant_id: str,
    job_id: str,
    phase: str,
    pct: int,
    details: dict[str, Any] | None = None,
) -> None:
    try:
        job = repo.get_job(tenant_id, job_id)
        if not job:
            return
        progress = job.get("progress", {})
        if not isinstance(progress, dict):
            progress = {}
        phase_data = progress.get(phase, {})
        phase_data["status"] = "completed" if pct >= 100 else "running"
        phase_data["progress"] = pct
        if details:
            phase_data.update(details)
        progress[phase] = phase_data
        repo.update_job(tenant_id, job_id, {"progress": progress})
    except Exception as e:
        logger.warning("Failed to update phase progress: %s", e)
