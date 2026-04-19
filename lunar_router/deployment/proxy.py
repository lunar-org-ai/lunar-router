"""Inference proxy for llama-server deployments.

Forwards OpenAI-compatible requests from the platform API to the local
llama-server instance, captures per-request metrics, and writes them to
ClickHouse for the dashboard.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import inference_metrics, storage

logger = logging.getLogger(__name__)

proxy_router = APIRouter()

# Reusable HTTP client (per-process) — opens a small connection pool.
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    return _http_client


def _resolve_endpoint(deployment_id: str) -> tuple[str, str]:
    """Return (endpoint_url, model_id) or raise 404/503."""
    dep = storage.get_deployment(deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
    if dep.get("status") != "in_service":
        raise HTTPException(
            status_code=503,
            detail=f"Deployment {deployment_id} is {dep.get('status', 'unknown')}, not in_service",
        )
    url = dep.get("endpoint_url") or ""
    if not url:
        raise HTTPException(status_code=503, detail="Deployment has no endpoint URL")
    return url, dep.get("model_id", "")


def _extract_metrics(body: dict[str, Any], duration_ms: float) -> dict[str, Any]:
    """Pull token counts, finish reason, and llama-server timings from a chat/completion response."""
    usage = body.get("usage", {}) or {}
    timings = body.get("timings", {}) or {}
    finish_reason = ""
    choices = body.get("choices", []) or []
    if choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason", "") or ""

    # llama-server provides timings.predict_ms — use that as a proxy for TTFT in non-streaming.
    # For non-streaming there's no real TTFT; we'll record it as 0.
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "prompt_per_second": float(timings.get("prompt_per_second", 0) or 0),
        "predict_per_second": float(timings.get("predict_per_second", 0) or 0),
        "finish_reason": finish_reason,
    }


async def _forward_and_record(
    deployment_id: str,
    endpoint_path: str,
    request: Request,
    endpoint_label: str,
) -> JSONResponse:
    """Forward a non-streaming JSON request to llama-server and record metrics."""
    base_url, model_id = _resolve_endpoint(deployment_id)
    target = base_url.rstrip("/") + endpoint_path

    try:
        body = await request.json()
    except Exception:
        body = {}

    request_id = str(uuid.uuid4())
    temperature = float(body.get("temperature", 0) or 0)
    max_tokens = int(body.get("max_tokens", 0) or 0)
    is_stream = bool(body.get("stream", False))

    if is_stream:
        return await _forward_stream(
            deployment_id=deployment_id,
            model_id=model_id,
            target=target,
            body=body,
            endpoint_label=endpoint_label,
            request_id=request_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    started = time.perf_counter()
    status_code = 500
    error = ""
    response_body: dict[str, Any] = {}

    try:
        resp = await _client().post(
            target,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        status_code = resp.status_code
        try:
            response_body = resp.json()
        except Exception:
            response_body = {"error": "non-json response", "text": resp.text[:500]}
        if status_code >= 400:
            err_obj = response_body.get("error", response_body)
            error = json.dumps(err_obj)[:1000] if isinstance(err_obj, dict) else str(err_obj)[:1000]
    except httpx.HTTPError as e:
        error = f"upstream connection error: {e}"
        status_code = 502
    except Exception as e:
        error = f"proxy error: {e}"
        status_code = 500

    duration_ms = (time.perf_counter() - started) * 1000.0

    metrics = _extract_metrics(response_body, duration_ms) if status_code < 400 else {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_per_second": 0.0, "predict_per_second": 0.0, "finish_reason": "",
    }

    inference_metrics.record_inference(
        deployment_id=deployment_id,
        model_id=model_id,
        request_id=request_id,
        duration_ms=duration_ms,
        status_code=status_code,
        error=error,
        temperature=temperature,
        max_tokens=max_tokens,
        endpoint=endpoint_label,
        **metrics,
    )

    inference_metrics.record_trace(
        request_id=request_id,
        model_id=model_id,
        request_body=body,
        response_body=response_body,
        duration_ms=duration_ms,
        ttft_ms=0.0,
        status_code=status_code,
        error=error,
        request_type=endpoint_label,
        is_stream=False,
    )

    return JSONResponse(content=response_body, status_code=status_code)


async def _forward_stream(
    *,
    deployment_id: str,
    model_id: str,
    target: str,
    body: dict[str, Any],
    endpoint_label: str,
    request_id: str,
    temperature: float,
    max_tokens: int,
) -> StreamingResponse:
    """Forward a streaming SSE request, capture TTFT and final usage."""
    started = time.perf_counter()
    state: dict[str, Any] = {
        "ttft_ms": 0.0,
        "first_chunk": True,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "finish_reason": "",
        "predict_per_second": 0.0,
        "prompt_per_second": 0.0,
        "status": 200,
        "error": "",
        "output_text": "",
    }

    async def iter_chunks():
        try:
            async with _client().stream(
                "POST", target, json=body, headers={"Content-Type": "application/json"}
            ) as resp:
                state["status"] = resp.status_code
                async for raw in resp.aiter_lines():
                    if raw is None:
                        continue
                    if state["first_chunk"]:
                        state["ttft_ms"] = (time.perf_counter() - started) * 1000.0
                        state["first_chunk"] = False
                    line = raw if raw.endswith("\n") else raw + "\n"
                    yield line.encode("utf-8")
                    # Try to extract usage / finish_reason from each SSE chunk
                    if raw.startswith("data:"):
                        payload = raw[5:].strip()
                        if payload and payload != "[DONE]":
                            try:
                                obj = json.loads(payload)
                                _stream_extract(obj, state)
                            except Exception:
                                pass
        except Exception as e:
            state["status"] = 502
            state["error"] = f"stream proxy error: {e}"
            yield f"data: {{\"error\": \"{e}\"}}\n\n".encode("utf-8")
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            inference_metrics.record_inference(
                deployment_id=deployment_id,
                model_id=model_id,
                request_id=request_id,
                duration_ms=duration_ms,
                ttft_ms=state["ttft_ms"],
                prompt_tokens=state["prompt_tokens"],
                completion_tokens=state["completion_tokens"],
                total_tokens=state["total_tokens"],
                prompt_per_second=state["prompt_per_second"],
                predict_per_second=state["predict_per_second"],
                status_code=state["status"],
                error=state["error"],
                finish_reason=state["finish_reason"],
                temperature=temperature,
                max_tokens=max_tokens,
                endpoint=endpoint_label,
            )
            synth_response = {
                "usage": {
                    "prompt_tokens": state["prompt_tokens"],
                    "completion_tokens": state["completion_tokens"],
                    "total_tokens": state["total_tokens"],
                },
                "timings": {"predicted_per_second": state["predict_per_second"]},
                "choices": [
                    {
                        "finish_reason": state["finish_reason"],
                        "message": {"role": "assistant", "content": state["output_text"]},
                    }
                ],
            }
            inference_metrics.record_trace(
                request_id=request_id,
                model_id=model_id,
                request_body=body,
                response_body=synth_response,
                duration_ms=duration_ms,
                ttft_ms=state["ttft_ms"],
                status_code=state["status"],
                error=state["error"],
                request_type=endpoint_label,
                is_stream=True,
            )

    return StreamingResponse(iter_chunks(), media_type="text/event-stream")


def _stream_extract(obj: dict[str, Any], state: dict[str, Any]) -> None:
    """Update streaming state from an SSE chunk payload."""
    usage = obj.get("usage")
    if isinstance(usage, dict):
        state["prompt_tokens"] = int(usage.get("prompt_tokens", state["prompt_tokens"]) or 0)
        state["completion_tokens"] = int(usage.get("completion_tokens", state["completion_tokens"]) or 0)
        state["total_tokens"] = int(usage.get("total_tokens", state["total_tokens"]) or 0)
    timings = obj.get("timings")
    if isinstance(timings, dict):
        state["predict_per_second"] = float(timings.get("predict_per_second", state["predict_per_second"]) or 0)
        state["prompt_per_second"] = float(timings.get("prompt_per_second", state["prompt_per_second"]) or 0)
    choices = obj.get("choices") or []
    if choices and isinstance(choices[0], dict):
        ch = choices[0]
        fr = ch.get("finish_reason")
        if fr:
            state["finish_reason"] = fr
        delta = ch.get("delta") or {}
        if isinstance(delta, dict):
            piece = delta.get("content")
            if isinstance(piece, str):
                state["output_text"] += piece
        elif "text" in ch and isinstance(ch["text"], str):
            state["output_text"] += ch["text"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@proxy_router.post("/{deployment_id}/v1/chat/completions")
async def chat_completions(deployment_id: str, request: Request):
    return await _forward_and_record(
        deployment_id, "/v1/chat/completions", request, endpoint_label="chat",
    )


@proxy_router.post("/{deployment_id}/v1/completions")
async def completions(deployment_id: str, request: Request):
    return await _forward_and_record(
        deployment_id, "/v1/completions", request, endpoint_label="completions",
    )


@proxy_router.post("/{deployment_id}/v1/embeddings")
async def embeddings(deployment_id: str, request: Request):
    return await _forward_and_record(
        deployment_id, "/v1/embeddings", request, endpoint_label="embeddings",
    )
