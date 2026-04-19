"""
Distillation Router — FastAPI endpoints for the BOND distillation pipeline.

Endpoints match the UI's distillationService.ts expectations:
  GET    /v1/distillation              — list jobs
  POST   /v1/distillation              — create job
  GET    /v1/distillation/{job_id}     — get job details
  DELETE /v1/distillation/{job_id}     — delete job
  POST   /v1/distillation/{job_id}/cancel    — cancel job
  GET    /v1/distillation/{job_id}/logs      — get pipeline logs
  GET    /v1/distillation/{job_id}/candidates — get curation candidates
  GET    /v1/distillation/{job_id}/artifacts  — get GGUF artifacts
  POST   /v1/distillation/estimate           — estimate job cost/time
  GET    /v1/distillation/student-models      — list available student models
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from . import repository as repo
from .schemas import (
    STUDENT_MODEL_MAP,
    MODEL_PARAM_SIZES,
    TEACHER_MODEL_MAP,
    CreateJobRequest,
    DistillationConfig,
    EstimateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["distillation"])

DATA_DIR = Path(os.environ.get("LUNAR_DATA_DIR", "data"))

DEFAULT_TENANT = "default"


def _tenant(tenant_id: str | None) -> str:
    return tenant_id or DEFAULT_TENANT


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    """Serialize a job dict for the API response, matching BackendJobDetails."""
    return {
        "id": job.get("job_id", ""),
        "name": job.get("name", ""),
        "description": job.get("description", ""),
        "tenant_id": job.get("tenant_id", DEFAULT_TENANT),
        "status": job.get("status", "pending"),
        "phase": job.get("phase", "initializing"),
        "config": job.get("config", {}),
        "progress": job.get("progress", {}),
        "results": job.get("results"),
        "artifacts": job.get("artifacts"),
        "error": job.get("error", ""),
        "cost_accrued": float(job.get("cost_accrued", 0)),
        "created_at": _to_str(job.get("created_at", "")),
        "updated_at": _to_str(job.get("updated_at", "")),
        "started_at": _to_str(job.get("started_at")) if job.get("started_at") else None,
        "completed_at": _to_str(job.get("completed_at")) if job.get("completed_at") else None,
    }


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val)



@router.get("/v1/distillation")
async def list_jobs(
    tenant_id: str = Query(DEFAULT_TENANT),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List distillation jobs for a tenant."""
    tid = _tenant(tenant_id)
    jobs, total = repo.list_jobs(tid, status=status, limit=limit, offset=offset)
    return [_serialize_job(j) for j in jobs if j.get("status") != "deleted"]



@router.post("/v1/distillation")
async def create_job(body: dict[str, Any]):
    """Create a new distillation job and start the pipeline."""
    tid = _tenant(body.get("tenant_id"))
    name = body.get("name", "Untitled")
    description = body.get("description", "")
    config_data = body.get("config", {})

    # Parse and validate config
    try:
        config = DistillationConfig(**config_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    # Create job in ClickHouse
    job = repo.create_job(tid, {
        "name": name,
        "description": description,
        "config": config.model_dump(),
    })

    job_id = job["job_id"]
    logger.info("Created distillation job %s for tenant %s", job_id, tid)

    # Launch pipeline in background
    from .pipeline import launch_pipeline
    launch_pipeline(job_id, tid, config)

    return _serialize_job(job)



@router.post("/v1/distillation/estimate")
async def estimate_job(
    body: EstimateRequest,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Estimate job cost and time."""
    model_key = body.student_model
    num_prompts = body.num_prompts
    n_samples = body.n_samples

    # Estimate based on model size and prompts
    param_size = MODEL_PARAM_SIZES.get(model_key, 1.0)

    # Rough cost estimate (local compute, mainly electricity)
    gen_cost = num_prompts * n_samples * 0.001  # ~$0.001 per candidate generation
    train_cost = param_size * 0.01  # ~$0.01 per billion params
    estimated_cost = gen_cost + train_cost

    return {
        "estimated_cost": round(estimated_cost, 2),
        "is_sandbox": False,
        "tier": "local",
        "balance": 999999,
        "sufficient": True,
    }



@router.get("/v1/distillation/teacher-models")
async def list_teacher_models():
    """List available teacher models (API models via Go engine)."""
    models = []
    for m in TEACHER_MODEL_MAP:
        models.append({
            "id": m["id"],
            "name": m["name"],
            "provider": m["provider"],
            "type": "external",
            "available": True,
        })
    return {"models": models}



@router.get("/v1/distillation/student-models")
async def list_available_models():
    """List available student models."""
    models = []
    for key in STUDENT_MODEL_MAP:
        family = key.split("-")[0]
        params = MODEL_PARAM_SIZES.get(key, 0)
        models.append({
            "id": key,
            "name": key,
            "family": family,
            "params": f"{params}B",
            "available": True,
        })
    return {"models": models}



@router.get("/v1/distillation/{job_id}")
async def get_job(
    job_id: str,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Get distillation job details."""
    tid = _tenant(tenant_id)
    job = repo.get_job(tid, job_id)
    if not job or job.get("status") == "deleted":
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job)



@router.delete("/v1/distillation/{job_id}")
async def delete_job(
    job_id: str,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Delete a distillation job."""
    tid = _tenant(tenant_id)

    # Cancel if running
    from .pipeline import cancel_pipeline
    cancel_pipeline(job_id)

    ok = repo.delete_job(tid, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True}



@router.post("/v1/distillation/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Cancel a running distillation job."""
    tid = _tenant(tenant_id)

    job = repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    active_statuses = {"pending", "queued", "running"}
    if job.get("status") in active_statuses:
        from .pipeline import cancel_pipeline
        cancel_pipeline(job_id)

        repo.update_job_status(tid, job_id, status="cancelled", error="Cancelled by user")

        job = repo.get_job(tid, job_id) or job

    return _serialize_job(job)



@router.get("/v1/distillation/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Get pipeline logs for a job."""
    tid = _tenant(tenant_id)
    job = repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = job.get("pipeline_logs", [])
    if not isinstance(logs, list):
        logs = []
    return {"logs": logs}



@router.get("/v1/distillation/{job_id}/candidates")
async def get_job_candidates(
    job_id: str,
    tenant_id: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get curation candidates grouped by prompt."""
    candidates = repo.get_candidates(job_id, limit=limit * 10)

    # Group by prompt_id
    grouped: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        pid = c.get("prompt_id", "")
        grouped.setdefault(pid, []).append(c)

    samples = []
    for prompt_id, prompt_candidates in list(grouped.items())[:limit]:
        prompt_text = prompt_candidates[0].get("prompt", "") if prompt_candidates else ""
        cands = []
        for c in prompt_candidates:
            cands.append({
                "text": c.get("response", ""),
                "scores": {"quality": float(c.get("score", 0))},
                "isWinner": bool(c.get("selected", 0)),
                "candidate_idx": int(c.get("candidate_idx", 0)),
            })
        samples.append({
            "prompt": prompt_text,
            "trace_id": prompt_id,
            "candidates": cands,
        })

    return {"samples": samples, "total": len(grouped)}



@router.get("/v1/distillation/{job_id}/artifacts")
async def get_job_artifacts(
    job_id: str,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Get GGUF artifacts for a completed job."""
    tid = _tenant(tenant_id)
    job = repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    artifacts_data = job.get("artifacts", {})
    gguf_map = artifacts_data.get("gguf", {}) if isinstance(artifacts_data, dict) else {}

    result = []
    for quant_type, file_path in gguf_map.items():
        path = Path(file_path) if file_path else None
        size = path.stat().st_size if path and path.exists() else 0
        result.append({
            "key": quant_type,
            "size": size,
            "url": f"/v1/distillation/{job_id}/download/{quant_type}",
            "expires_in": 0,  # Local file, no expiration
        })

    return result


@router.post("/v1/distillation/{job_id}/deploy")
async def deploy_distilled_job(
    job_id: str,
    body: dict[str, Any] | None = None,
    tenant_id: str = Query(DEFAULT_TENANT),
):
    """Deploy a completed distillation job's GGUF model via llama-server.

    Picks the smallest GGUF artifact (preferring q4_k_m → q5_k_m → q8_0 → first available),
    then launches llama-server in the background. Returns immediately with status='creating';
    the UI polls /v1/deployments/{deployment_id}/status to see when it transitions to in_service.
    """
    tid = _tenant(tenant_id)
    job = repo.get_job(tid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job must be completed before deploy (current status: {job.get('status')})",
        )

    artifacts_data = job.get("artifacts", {}) or {}
    gguf_map = artifacts_data.get("gguf", {}) if isinstance(artifacts_data, dict) else {}
    if not gguf_map:
        raise HTTPException(
            status_code=400,
            detail="No GGUF artifacts found for this job — export must complete successfully first",
        )

    # Pick the best quantization variant (smaller = faster on CPU/T4)
    preferred_order = ["q4_k_m", "q5_k_m", "q4_0", "q5_0", "q8_0", "f16"]
    selected_quant: str | None = None
    selected_path: str | None = None
    for quant in preferred_order:
        if quant in gguf_map and Path(gguf_map[quant]).exists():
            selected_quant = quant
            selected_path = gguf_map[quant]
            break
    if not selected_path:
        # Fall back to the first existing artifact
        for quant, fp in gguf_map.items():
            if fp and Path(fp).exists():
                selected_quant = quant
                selected_path = fp
                break

    if not selected_path:
        raise HTTPException(
            status_code=400,
            detail="GGUF artifacts referenced in job results no longer exist on disk",
        )

    body = body or {}
    instance_type = body.get("instance_type", "cpu-small")
    config = body.get("config", {}) or {}

    # Use the job name (or job_id prefix) as the user-visible model_id
    job_name = job.get("name") or f"distilled-{job_id[:8]}"
    model_id = f"distilled/{job_name}"

    # Pass model_alias so llama-server tags responses with the friendly name
    deploy_config = {
        **config,
        "instance_type": instance_type,
        "model_alias": model_id,
        "source_job_id": job_id,
        "quantization": selected_quant,
    }

    from ..deployment.manager import deploy_model
    result = await deploy_model(
        model_id=model_id,
        model_path=selected_path,
        config=deploy_config,
    )

    return {
        "deployment_id": result["deployment_id"],
        "model_id": result["model_id"],
        "status": result["status"],
        "endpoint_url": result.get("endpoint_url", ""),
        "already_deployed": result.get("already_deployed", False),
        "quantization": selected_quant,
        "model_path": selected_path,
    }


@router.get("/v1/distillation/{job_id}/metrics")
async def get_job_metrics(
    job_id: str,
    tenant_id: str = Query(DEFAULT_TENANT),
    limit: int = Query(5000, ge=1, le=50000),
):
    """Get training metrics for a job (training curves)."""
    metrics = repo.get_metrics(job_id, limit=limit)
    return {"metrics": metrics, "total": len(metrics)}
