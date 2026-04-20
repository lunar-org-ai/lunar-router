"""Serializers for distillation job API responses.

Extracted from the pre-rebrand ``opentracy/distillation/router.py`` after
that module was retired as a dead parallel route registry. The live
endpoints in ``opentracy/api/server.py`` still need a shared serializer
so the response shape doesn't drift between handlers, so that piece
lives here now.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TENANT = "default"


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def serialize_job(job: dict[str, Any]) -> dict[str, Any]:
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
