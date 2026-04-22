"""
Pipeline Orchestrator — runs the 4-phase distillation pipeline as a background task.

Replaces AWS Step Functions orchestration.
Phases: data_generation → curation → training → export
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

from . import repository as repo
from .schemas import DistillationConfig
from opentracy._env import env

logger = logging.getLogger(__name__)

DATA_DIR = Path(env("DATA_DIR", "data"))

# Track running pipelines to prevent duplicates
_running_jobs: dict[str, asyncio.Task] = {}


def launch_pipeline(job_id: str, tenant_id: str, config: DistillationConfig) -> None:
    """
    Launch the distillation pipeline as a background asyncio task.
    Safe to call from a sync FastAPI endpoint.
    """
    if job_id in _running_jobs and not _running_jobs[job_id].done():
        logger.warning("Pipeline already running for job %s", job_id)
        return

    loop = asyncio.get_event_loop()
    task = loop.create_task(_run_pipeline(job_id, tenant_id, config))
    _running_jobs[job_id] = task

    # Cleanup when done
    task.add_done_callback(lambda t: _running_jobs.pop(job_id, None))


def cancel_pipeline(job_id: str) -> bool:
    """Cancel a running pipeline."""
    task = _running_jobs.get(job_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


async def _run_pipeline(
    job_id: str,
    tenant_id: str,
    config: DistillationConfig,
) -> None:
    """
    Execute the full distillation pipeline.

    Phases:
        1. Data Generation — generate N candidates per prompt via teacher
        2. Curation — score and select best candidates
        3. Training — fine-tune student model (subprocess)
        4. Export — merge LoRA + GGUF conversion (subprocess)
    """
    config_dict = config.model_dump()

    try:
        repo.update_job_status(
            tenant_id, job_id,
            status="running", phase="data_generation",
        )
        repo.append_log(tenant_id, job_id, "Pipeline started")

        repo.update_job_status(tenant_id, job_id, phase="data_generation")
        repo.append_log(tenant_id, job_id, "Phase 1/4: Data Generation")

        prompts = await _load_prompts(tenant_id, config_dict)
        if not prompts:
            raise ValueError("No prompts available for data generation")

        from .data_gen import generate_data

        gen_result = await generate_data(
            job_id, tenant_id, prompts,
            teacher_model=config.teacher_model,
            n_samples=config.n_samples,
            temperature=config.temperature,
        )

        if gen_result["total_candidates"] == 0:
            last_error = gen_result.get("last_error", "")
            if "401" in last_error or "API key" in last_error or "Unauthorized" in last_error:
                raise RuntimeError(
                    f"Data generation failed: the teacher model '{config.teacher_model}' "
                    f"could not be reached — API key is missing or invalid. "
                    f"Please add a valid API key in Settings > API Keys and try again."
                )
            elif "429" in last_error or "rate" in last_error.lower():
                raise RuntimeError(
                    f"Data generation failed: rate limit exceeded for '{config.teacher_model}'. "
                    f"Please wait a few minutes and try again."
                )
            elif last_error:
                raise RuntimeError(
                    f"Data generation produced no candidates. Last error: {last_error}"
                )
            else:
                raise RuntimeError("Data generation produced no candidates")

        repo.append_log(
            tenant_id, job_id,
            f"Phase 1 complete: {gen_result['total_candidates']} candidates",
        )

        repo.update_job_status(tenant_id, job_id, phase="curation")
        repo.append_log(tenant_id, job_id, "Phase 2/4: Curation")

        from .curation import curate_candidates

        curation_result = await curate_candidates(job_id, tenant_id, config_dict)

        curated_path = curation_result.get("curated_path", "")
        if not curated_path or not Path(curated_path).exists():
            raise RuntimeError("Curation produced no output")

        curated_count = curation_result.get("curated_examples", 0)
        repo.append_log(
            tenant_id, job_id,
            f"Phase 2 complete: {curated_count} curated examples",
        )

        repo.update_job_status(tenant_id, job_id, phase="training")
        repo.append_log(tenant_id, job_id, "Phase 3/4: Training (subprocess)")

        from .trainer import start_training

        train_result = await start_training(
            job_id, tenant_id, config_dict, curated_path,
        )

        adapter_dir = train_result.get("output_dir", "")
        if not adapter_dir or not Path(adapter_dir).exists():
            raise RuntimeError("Training produced no adapter")

        repo.append_log(tenant_id, job_id, "Phase 3 complete: adapter saved")

        # Record the adapter path BEFORE attempting GGUF export — if export
        # fails (missing llama.cpp, OOM, etc.) the adapter is still usable
        # and callers like ``ot.distill()`` can fall back to it.
        repo.update_job(
            tenant_id, job_id,
            {"artifacts": {"adapter_path": adapter_dir}},
        )

        if config.export_gguf:
            repo.update_job_status(tenant_id, job_id, phase="export")
            repo.append_log(tenant_id, job_id, "Phase 4/4: Export (subprocess)")

            from .export import start_export

            export_result = await start_export(
                job_id, tenant_id, config_dict, adapter_dir,
            )

            artifacts = export_result.get("artifacts", {})
            repo.append_log(
                tenant_id, job_id,
                f"Phase 4 complete: {len(artifacts)} GGUF artifacts",
            )
        else:
            # Skip export, just update progress
            job = repo.get_job(tenant_id, job_id)
            if job:
                progress = job.get("progress", {})
                progress["export"] = {"status": "skipped", "progress": 100}
                repo.update_job(tenant_id, job_id, {"progress": progress})
            repo.append_log(tenant_id, job_id, "Phase 4 skipped: export_gguf=false")

        repo.update_job_status(
            tenant_id, job_id,
            status="completed", phase="completed",
        )
        repo.append_log(tenant_id, job_id, "Pipeline completed successfully!")

    except asyncio.CancelledError:
        logger.info("Pipeline cancelled for job %s", job_id)
        repo.update_job_status(
            tenant_id, job_id,
            status="cancelled", error="Pipeline cancelled by user",
        )
        repo.append_log(tenant_id, job_id, "Pipeline cancelled")

    except Exception as e:
        logger.error("Pipeline failed for job %s: %s", job_id, e)
        logger.error(traceback.format_exc())
        repo.update_job_status(
            tenant_id, job_id,
            status="failed", error=str(e),
        )
        repo.append_log(tenant_id, job_id, f"Pipeline failed: {e}")


async def _load_prompts(
    tenant_id: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Load prompts for data generation.

    Sources (in priority order):
    1. dataset_id → load from datasets table
    2. Inline prompts in config
    3. Generate from num_prompts count (placeholder prompts)
    """
    dataset_id = config.get("dataset_id", "")

    if dataset_id:
        try:
            return await _load_from_dataset(tenant_id, dataset_id)
        except Exception as e:
            logger.warning("Failed to load dataset %s: %s", dataset_id, e)

    # Fallback: check if prompts are inline
    inline = config.get("prompts", [])
    if inline:
        return [
            {"id": str(i), "text": p} if isinstance(p, str) else p
            for i, p in enumerate(inline)
        ]

    # No dataset available — try loading from production traces (incl. tool_calls)
    try:
        traces = await _load_tool_call_traces(tenant_id, config)
        if traces:
            return traces
    except Exception as e:
        logger.warning("Failed to load traces: %s", e)

    return []


async def _load_from_dataset(
    tenant_id: str,
    dataset_id: str,
) -> list[dict[str, Any]]:
    """Load prompts from the eval datasets table."""
    from ..evals_common.db import query_rows

    # Load prompts from the eval_samples table
    rows = query_rows(
        """
        SELECT sample_id, input, expected_output, metadata
        FROM eval_samples FINAL
        WHERE dataset_id = {dataset_id:String}
        ORDER BY created_at
        LIMIT 100000
        """,
        {"dataset_id": dataset_id},
    )

    if rows:
        prompts = []
        for r in rows:
            prompts.append({
                "id": r.get("sample_id", ""),
                "text": r.get("input", ""),
                "system": "",
            })
        return prompts

    # Try clustering dataset traces
    # dataset_id might be "run_id/cluster_id" format
    if "/" in dataset_id:
        parts = dataset_id.split("/", 1)
        run_id, cluster_id = parts[0], parts[1]
    else:
        run_id, cluster_id = dataset_id, "0"

    rows = query_rows(
        """
        SELECT trace_id, input, output
        FROM domain_dataset_traces FINAL
        WHERE run_id = {run_id:String}
          AND cluster_id = {cluster_id:UInt32}
        ORDER BY created_at
        LIMIT 100000
        """,
        {"run_id": run_id, "cluster_id": int(cluster_id)},
    )

    prompts = []
    for r in rows:
        prompts.append({
            "id": r.get("trace_id", ""),
            "text": r.get("input", ""),
            "system": "",
        })
    return prompts


async def _load_tool_call_traces(
    tenant_id: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Load successful tool_call traces from llm_traces as training data.

    This enables distilling a model that can make tool_calls by learning
    from real production traces that used tools successfully.
    """
    from ..evals_common.db import query_rows

    model_filter = config.get("tool_call_model", "")
    limit = config.get("tool_call_limit", 1000)
    days = config.get("tool_call_days", 30)

    params: dict[str, Any] = {"limit": limit, "days": days}
    model_clause = ""
    if model_filter:
        model_clause = "AND selected_model = {model:String}"
        params["model"] = model_filter

    rows = query_rows(
        f"""
        SELECT request_id, input_messages, output_message,
               request_tools, response_tool_calls,
               input_text, output_text
        FROM llm_traces FINAL
        WHERE has_tool_calls = 1
          AND is_error = 0
          AND timestamp >= now() - INTERVAL {{days:UInt32}} DAY
          {model_clause}
        ORDER BY timestamp DESC
        LIMIT {{limit:UInt32}}
        """,
        params,
    )

    prompts = []
    for r in rows:
        input_messages = r.get("input_messages", "")
        output_message = r.get("output_message", "")
        request_tools = r.get("request_tools", "")
        response_tool_calls = r.get("response_tool_calls", "")

        prompts.append({
            "id": r.get("request_id", ""),
            "text": r.get("input_text", ""),
            "system": "",
            "input_messages": input_messages,
            "output_message": output_message,
            "tools": request_tools,
            "tool_calls": response_tool_calls,
        })

    logger.info("Loaded %d tool_call traces for distillation", len(prompts))
    return prompts
