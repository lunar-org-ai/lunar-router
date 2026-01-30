import json
from typing import List
from fastapi import APIRouter, Request
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import time
import uuid
from datetime import datetime
from decimal import Decimal
from ..database.models import TenantStatsModel, ProviderAttempt
from ..database.TenantStatsHandler import TenantStatsHandler
from ..schemas import InferBody, InferResponse, InferRequest
from ..helpers.utils import adapters_for, generate_stream_response
from ..router import HealthFirstPlanner

router = APIRouter(prefix="/v1/infer", tags=["Inference"])
planner = HealthFirstPlanner()

@router.post("/{model_name}", response_model=InferResponse)
async def infer_by_model(model_name: str, body: InferBody, request: Request, background_tasks: BackgroundTasks):
    """
    Health-first routing with fallback.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    adapters = await adapters_for(model_name)
    # Only rank if there are multiple adapters
    ordered = await planner.rank(model_name, adapters) if len(adapters) > 1 else adapters

    last_error = None
    tenant_stats: TenantStatsModel = None
    provider_attempts: List[ProviderAttempt] = []

    for chosen in ordered:
        attempt_start = datetime.utcnow()
        resp, metrics = await chosen.send(
            {
                "tenant": tenant_id,
                "model": model_name,
                "prompt": body.prompt,
                "stream": body.stream,
            }
        )

        # Track this provider attempt
        attempt = ProviderAttempt(
            provider=chosen.name,
            success="error" not in resp,
            error_category=resp.get("error_category") if "error" in resp else None,
            error_message=resp.get("error") if "error" in resp else None,
            latency_ms=metrics.get("latency_ms", 0),
            timestamp=attempt_start.isoformat(),
        )
        provider_attempts.append(attempt)

        tenant_stats = TenantStatsModel(
            TenantId=tenant_id,
            CreationDate=datetime.utcnow().isoformat(),
            Provider=chosen.name,
            Model=model_name,
            TTFT=Decimal(str(metrics.get("ttft_ms", 0.0))),
            Latency=Decimal(str(metrics.get("latency_ms", 0.0))),
            Success=not bool(metrics.get("error", 0.0)),
            InputText=body.prompt,
            OutputText=resp.get("text", ""),
            TotalTokens=int(metrics.get("tokens_in", 0)) + int(metrics.get("tokens_out", 0)),
            # New fields for enhanced error tracking
            ErrorCategory=resp.get("error_category") if "error" in resp else None,
            ErrorMessage=resp.get("error") if "error" in resp else None,
            ProviderAttempts=json.dumps([a.model_dump() for a in provider_attempts]),
            FallbackCount=len(provider_attempts),
            FinalProvider=chosen.name
        )

        if "error" not in resp:
            # Handle streaming response
            if body.stream and "stream" in resp:
                req_id = f"infer_{uuid.uuid4().hex}"
                start_time = time.time()
                
                return StreamingResponse(
                    generate_stream_response(
                        resp["stream"], 
                        req_id, 
                        model_name, 
                        tenant_id, 
                        model_name, 
                        chosen.name, 
                        start_time,
                        tenant_stats,
                        background_tasks,
                        metrics
                    ),
                    media_type="text/plain",
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            
            response = InferResponse(provider=chosen.name,
                model=model_name,
                text=resp.get("text", ""),
                metrics=metrics,
                chosen_by={"strategy": "health-first+fallback"})
            background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
            return response

        last_error = resp.get("error")
        tenant_stats.ErrorType = last_error
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
    raise HTTPException(
        status_code=502,
        detail=f"All providers failed for model '{model_name}': {last_error}",
    )

@router.post("/{model_name}/{provider_id}", response_model=InferResponse)
async def infer_by_model_and_provider(
    model_name: str,
    provider_id: str,
    body: InferBody,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Forces a specific provider for a logical model. No fallback.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    adapters = await adapters_for(model_name)
    forced = next((a for a in adapters if a.name == provider_id), None)
    if not forced:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider_id}' not found for model '{model_name}'",
        )

    attempt_start = datetime.utcnow()
    resp, metrics = await forced.send(
        {
            "tenant": tenant_id,
            "model": model_name,
            "prompt": body.prompt,
            "stream": body.stream,
        }
    )

    # Track this provider attempt
    attempt = ProviderAttempt(
        provider=forced.name,
        success="error" not in resp,
        error_category=resp.get("error_category") if "error" in resp else None,
        error_message=resp.get("error") if "error" in resp else None,
        latency_ms=metrics.get("latency_ms", 0),
        timestamp=attempt_start.isoformat(),
    )

    tenant_stats = TenantStatsModel(
        TenantId=tenant_id,
        CreationDate=datetime.utcnow().isoformat(),
        Provider=forced.name,
        Model=model_name,
        TTFT=Decimal(str(metrics.get("ttft_ms", 0.0))),
        Latency=Decimal(str(metrics.get("latency_ms", 0.0))),
        Success=not bool(metrics.get("error", 0.0)),
        InputText=body.prompt,
        OutputText=resp.get("text", ""),
        TotalTokens=int(metrics.get("tokens_in", 0)) + int(metrics.get("tokens_out", 0)),
        # New fields for enhanced error tracking
        ErrorCategory=resp.get("error_category") if "error" in resp else None,
        ErrorMessage=resp.get("error") if "error" in resp else None,
        ProviderAttempts=json.dumps([attempt.model_dump()]),
        FallbackCount=1,
        FinalProvider=forced.name
    )

    if "error" in resp:
        tenant_stats.ErrorType = resp.get("error")
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
        raise HTTPException(
            status_code=502,
            detail=f"Provider {forced.name} failed: {resp['error']}",
        )

    background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
    return InferResponse(
        provider=forced.name,
        model=model_name,
        text=resp.get("text", ""),
        metrics=metrics,
        chosen_by={"forced_provider": provider_id},
    )

@router.post("/", response_model=InferResponse)
async def infer_legacy(req: InferRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Legacy infer: model comes in the body.
    Still uses health-first + fallback.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    adapters = await adapters_for(req.model)
    ordered = await planner.rank(req.model, adapters)

    last_error = None
    tenant_stats: TenantStatsModel = None
    provider_attempts: List[ProviderAttempt] = []

    for chosen in ordered:
        attempt_start = datetime.utcnow()
        resp, metrics = await chosen.send(
            {
                "tenant": tenant_id,
                "model": req.model,
                "prompt": req.prompt,
                "stream": req.stream,
            }
        )

        # Track this provider attempt
        attempt = ProviderAttempt(
            provider=chosen.name,
            success="error" not in resp,
            error_category=resp.get("error_category") if "error" in resp else None,
            error_message=resp.get("error") if "error" in resp else None,
            latency_ms=metrics.get("latency_ms", 0),
            timestamp=attempt_start.isoformat(),
        )
        provider_attempts.append(attempt)

        tenant_stats = TenantStatsModel(
            TenantId=tenant_id,
            CreationDate=datetime.utcnow().isoformat(),
            Provider=chosen.name,
            Model=req.model,
            TTFT=Decimal(str(metrics.get("ttft_ms", 0.0))),
            Latency=Decimal(str(metrics.get("latency_ms", 0.0))),
            Success=not bool(metrics.get("error", 0.0)),
            InputText=req.prompt,
            OutputText=resp.get("text", ""),
            TotalTokens=int(metrics.get("tokens_in", 0)) + int(metrics.get("tokens_out", 0)),
            # New fields for enhanced error tracking
            ErrorCategory=resp.get("error_category") if "error" in resp else None,
            ErrorMessage=resp.get("error") if "error" in resp else None,
            ProviderAttempts=json.dumps([a.model_dump() for a in provider_attempts]),
            FallbackCount=len(provider_attempts),
            FinalProvider=chosen.name
        )

        if "error" not in resp:
            # Handle streaming response
            if req.stream and "stream" in resp:
                req_id = f"infer_{uuid.uuid4().hex}"
                start_time = time.time()
                
                return StreamingResponse(
                    generate_stream_response(
                        resp["stream"], 
                        req_id, 
                        req.model, 
                        tenant_id, 
                        req.model, 
                        chosen.name, 
                        start_time,
                        tenant_stats,
                        background_tasks,
                        metrics
                    ),
                    media_type="text/plain",
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            
            background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
            return InferResponse(
                provider=chosen.name,
                model=req.model,
                text=resp.get("text", ""),
                metrics=metrics,
                chosen_by={
                    "strategy": "health-first+fallback",
                    "legacy": True,
                },
            )

        last_error = resp.get("error")
        tenant_stats.ErrorType = last_error
        background_tasks.add_task(TenantStatsHandler.insert, tenant_stats)
    raise HTTPException(
        status_code=502,
        detail=f"All providers failed for model '{req.model}': {last_error}",
    )