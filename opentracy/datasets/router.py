"""
Datasets API Router

⚠️ UNUSED — DO NOT ADD LOGIC HERE WITHOUT WIRING IT UP.

This module is never imported anywhere in the app. The live `/v1/datasets/*`
endpoints currently live inline in `opentracy/api/server.py` (search for
`@app.get("/v1/datasets`). Two parallel implementations previously had
different defaults (this one: include_samples=True / limit=1000; server.py:
include_samples=False / limit=50), which caused the dataset detail page to
render "No samples yet" for populated datasets because only the server.py
route was actually registered. Do not add new routes here — edit
`api/server.py` instead — until this module is explicitly mounted via
`app.include_router(...)` and the duplicate routes in server.py are removed.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from opentracy._env import env
from . import repository as repo
from .schemas import (
    AddSamplesRequest,
    AnalyzeTracesRequest,
    AutoCollectConfigIn,
    CreateFromInstructionRequest,
    CreateFromTracesRequest,
    DatasetCreate,
    ImportTracesRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenant(request: Request) -> str:
    """Extract tenant from X-Tenant-Id header (default: 'default')."""
    return request.headers.get("x-tenant-id", "default")


def _extract_value(obj: dict, path: str) -> str:
    """Walk dot-separated path into a dict and return a string."""
    if not path:
        return ""
    parts = path.split(".")
    current: Any = obj
    for p in parts:
        if isinstance(current, dict):
            current = current.get(p)
        else:
            return ""
    if current is None:
        return ""
    if isinstance(current, (dict, list)):
        return json.dumps(current, ensure_ascii=False)
    return str(current)


def _detect_field(sample: dict, candidates: list[str]) -> str:
    """Return the first key in *sample* that appears in *candidates*."""
    for key in candidates:
        if key in sample:
            return key
    return ""



@router.get("")
async def list_datasets(request: Request):
    tenant = _tenant(request)
    datasets = repo.list_datasets(tenant)
    return {"datasets": datasets, "total": len(datasets)}


@router.post("")
async def create_dataset(body: DatasetCreate, request: Request):
    tenant = _tenant(request)
    ds = repo.create_dataset(tenant, name=body.name, description=body.description or "", source=body.source or "manual")
    return {"dataset": ds}



@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    request: Request,
    include_samples: bool = Query(True),
    samples_limit: int = Query(1000),
    samples_offset: int = Query(0),
):
    tenant = _tenant(request)
    ds = repo.get_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    samples: list[dict[str, Any]] = []
    samples_total = 0
    if include_samples:
        samples = repo.get_samples(tenant, dataset_id, limit=samples_limit, offset=samples_offset)
        samples_total = repo.get_samples_count(tenant, dataset_id)
    return {"dataset": ds, "samples": samples, "samples_total": samples_total}


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, request: Request):
    tenant = _tenant(request)
    repo.delete_dataset(tenant, dataset_id)
    return {"success": True}



@router.post("/{dataset_id}/samples")
async def add_samples(dataset_id: str, body: AddSamplesRequest, request: Request):
    tenant = _tenant(request)
    ds = repo.get_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    raw = [s.model_dump() for s in body.samples]
    count = repo.add_samples(tenant, dataset_id, raw)
    return {"message": f"Added {count} samples", "count": count, "added": count}


@router.delete("/{dataset_id}/samples/{sample_id}")
async def delete_sample(dataset_id: str, sample_id: str, request: Request):
    tenant = _tenant(request)
    repo.delete_sample(tenant, dataset_id, sample_id)
    return {"success": True}



@router.post("/from-traces")
async def create_from_traces(body: CreateFromTracesRequest, request: Request):
    tenant = _tenant(request)

    # Fetch requested traces
    if body.trace_ids:
        traces = []
        for tid in body.trace_ids:
            t = repo.get_trace(tid)
            if t:
                traces.append(t)
    else:
        traces, _ = repo.list_traces(
            model_id=body.model_id,
            limit=body.limit or 50,
        )

    if not traces:
        raise HTTPException(status_code=400, detail="No traces found")

    # Create dataset
    ds = repo.create_dataset(
        tenant,
        name=body.name,
        description=body.description or "",
        source="traces",
    )

    # Add each trace as a sample
    samples = [
        {
            "input": t.get("input", ""),
            "expected_output": t.get("output", ""),
            "trace_id": t.get("trace_id", t.get("id", "")),
            "metadata": {
                "model_id": t.get("model_id", ""),
                "provider": t.get("provider", ""),
                "latency_ms": t.get("latency_ms", 0),
                "cost_usd": t.get("cost_usd", 0),
            },
        }
        for t in traces
    ]
    repo.add_samples(tenant, ds["dataset_id"], samples)

    # Re-fetch to get updated counts
    ds = repo.get_dataset(tenant, ds["dataset_id"]) or ds
    return {"dataset": ds}



@router.post("/from-instruction")
async def create_from_instruction(body: CreateFromInstructionRequest, request: Request):
    """Create dataset from instruction by filtering traces that match.

    Simplified implementation: selects recent traces matching model_id.
    A full implementation would use an LLM to filter/curate.
    """
    tenant = _tenant(request)

    traces, _ = repo.list_traces(
        model_id=body.model_id,
        limit=body.limit or 200,
    )

    if not traces:
        raise HTTPException(status_code=400, detail="No traces found matching criteria")

    ds = repo.create_dataset(
        tenant,
        name=body.name,
        description=body.description or body.instruction,
        source="instruction",
    )

    samples = [
        {
            "input": t.get("input", ""),
            "expected_output": t.get("output", ""),
            "trace_id": t.get("trace_id", t.get("id", "")),
            "metadata": {"model_id": t.get("model_id", "")},
        }
        for t in traces[: body.max_samples or 100]
    ]
    repo.add_samples(tenant, ds["dataset_id"], samples)

    ds = repo.get_dataset(tenant, ds["dataset_id"]) or ds
    return {
        "dataset": ds,
        "samples_added": len(samples),
        "traces_analyzed": len(traces),
    }



@router.post("/generate")
async def generate_dataset():
    raise HTTPException(status_code=410, detail="Synthetic generation has been removed")



@router.post("/analyze-traces")
async def analyze_traces_schema(body: AnalyzeTracesRequest):
    """Auto-detect input/output schema from uploaded JSON data."""
    data = body.data
    if not data:
        raise HTTPException(status_code=400, detail="'data' must be a non-empty array")

    sample = data[0]

    input_candidates = [
        "messages", "input", "prompt", "question", "query",
        "user_message", "instruction", "text", "input_text",
    ]
    output_candidates = [
        "output", "response", "answer", "completion", "expected_output",
        "assistant_message", "output_text", "result", "generated_text",
    ]

    source_format = "unknown"
    input_path = ""
    output_path = ""

    if "messages" in sample and isinstance(sample["messages"], list):
        source_format = "openai-messages"
        input_path = "messages"
        output_path = "messages"
    else:
        input_path = _detect_field(sample, input_candidates)
        output_path = _detect_field(sample, output_candidates)
        if input_path and output_path:
            source_format = "input-output"
        elif input_path:
            source_format = "input-only"

    preview = []
    for record in data[:10]:
        inp = ""
        out = ""
        if source_format == "openai-messages":
            msgs = record.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    inp = m.get("content", "")
                elif m.get("role") == "assistant":
                    out = m.get("content", "")
        else:
            inp = _extract_value(record, input_path)
            out = _extract_value(record, output_path)

        meta: dict[str, Any] = {}
        for k, v in record.items():
            if k not in (input_path, output_path, "messages") and isinstance(v, (str, int, float, bool)):
                meta[k] = v

        preview.append({
            "input": inp[:500] if inp else "",
            "expected_output": out[:500] if out else "",
            "metadata": meta,
        })

    mapping = {
        "input": {"path": input_path, "transform": "direct"},
        "output": {"path": output_path, "transform": "direct"},
        "metadata": {},
    }

    return {
        "mapping": mapping,
        "preview": preview,
        "source_format": source_format,
        "total_records": len(data),
    }


@router.post("/import-traces")
async def import_traces(body: ImportTracesRequest, request: Request):
    """Import trace data — transforms records and creates a proper dataset with samples."""
    tenant = _tenant(request)

    name = body.name
    data = body.data
    mapping = body.mapping
    description = body.description or ""

    if not data:
        raise HTTPException(status_code=400, detail="'data' is required")

    input_path = mapping.get("input", {}).get("path", "input") if mapping else "input"
    output_path = mapping.get("output", {}).get("path", "output") if mapping else "output"

    # Create dataset
    ds = repo.create_dataset(tenant, name=name, description=description, source="smart-import")

    samples = []
    skipped = 0
    for record in data:
        inp = ""
        out = ""

        if input_path == "messages" and "messages" in record:
            msgs = record.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    inp = m.get("content", "")
                elif m.get("role") == "assistant":
                    out = m.get("content", "")
        else:
            inp = _extract_value(record, input_path)
            out = _extract_value(record, output_path)

        if not inp and not out:
            skipped += 1
            continue

        meta: dict[str, Any] = {}
        for k, v in record.items():
            if k not in (input_path, output_path, "messages") and isinstance(v, (str, int, float, bool)):
                meta[k] = v

        samples.append({
            "input": inp,
            "expected_output": out,
            "metadata": meta,
        })

    if samples:
        repo.add_samples(tenant, ds["dataset_id"], samples)

    # Also forward to Go engine if LUNAR_ENGINE_URL is set
    engine_url = env("ENGINE_URL", "")
    if engine_url:
        try:
            traces_payload = []
            for s in samples:
                traces_payload.append({
                    "input": s["input"],
                    "output": s["expected_output"],
                    "source": f"import:{name}",
                })
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{engine_url}/v1/traces",
                    json={"traces": traces_payload},
                    timeout=60.0,
                )
                resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to forward traces to Go engine: %s", e)

    return {
        "dataset_id": ds["dataset_id"],
        "name": name,
        "source": "smart-import",
        "samples_count": len(samples),
        "skipped_count": skipped,
    }



@router.get("/{dataset_id}/auto-collect")
async def get_auto_collect(dataset_id: str, request: Request):
    tenant = _tenant(request)
    config = repo.get_auto_collect_config(tenant, dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail="No auto-collect config")
    return config


@router.put("/{dataset_id}/auto-collect")
async def put_auto_collect(dataset_id: str, body: AutoCollectConfigIn, request: Request):
    tenant = _tenant(request)
    ds = repo.get_dataset(tenant, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    config = repo.put_auto_collect_config(tenant, dataset_id, body.model_dump())
    return config


@router.delete("/{dataset_id}/auto-collect")
async def delete_auto_collect(dataset_id: str, request: Request):
    tenant = _tenant(request)
    repo.delete_auto_collect_config(tenant, dataset_id)
    return {"success": True}


@router.get("/{dataset_id}/auto-collect/history")
async def auto_collect_history(
    dataset_id: str,
    request: Request,
    limit: int = Query(20),
):
    tenant = _tenant(request)
    runs = repo.list_collect_runs(tenant, dataset_id, limit=limit)
    return {"runs": runs}


@router.post("/{dataset_id}/auto-collect/run")
async def trigger_auto_collect(dataset_id: str, request: Request):
    """Trigger a manual auto-collect run.

    Collects recent traces matching the config and adds them as samples.
    """
    tenant = _tenant(request)
    config = repo.get_auto_collect_config(tenant, dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail="No auto-collect config for this dataset")

    source_model = config.get("source_model", "")
    max_samples = config.get("max_samples", 100)

    # Get existing trace IDs to avoid duplicates
    collected_ids = repo.get_collected_trace_ids(tenant, dataset_id)

    # Fetch recent traces
    traces, _ = repo.list_traces(
        model_id=source_model or None,
        limit=max_samples * 2,  # fetch extra to account for filtered
    )

    # Filter out already-collected
    new_traces = [t for t in traces if t.get("trace_id") not in collected_ids]
    to_add = new_traces[:max_samples]

    samples = [
        {
            "input": t.get("input", ""),
            "expected_output": t.get("output", ""),
            "trace_id": t.get("trace_id", ""),
            "metadata": {
                "model_id": t.get("model_id", ""),
                "provider": t.get("provider", ""),
                "auto_collected": True,
            },
        }
        for t in to_add
    ]

    added = 0
    if samples:
        added = repo.add_samples(tenant, dataset_id, samples)

    # Save run history
    run = repo.save_collect_run(tenant, dataset_id, {
        "traces_scanned": len(traces),
        "traces_new": len(new_traces),
        "samples_added": added,
        "status": "completed",
    })

    # Update last_collected_at / total_collected in auto-collect config
    existing = repo.get_auto_collect_config(tenant, dataset_id)
    if existing:
        from datetime import datetime, timezone
        existing["last_collected_at"] = datetime.now(timezone.utc)
        existing["total_collected"] = (existing.get("total_collected", 0) or 0) + added
        repo.put_auto_collect_config(tenant, dataset_id, existing)

    return {"run_id": run.get("run_id", ""), "samples_added": added}
