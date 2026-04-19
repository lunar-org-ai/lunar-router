"""ClickHouse storage for per-request inference metrics from llama-server deployments."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_table_ready = False


def _ensure_table() -> None:
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return

    client.command("""
        CREATE TABLE IF NOT EXISTS inference_metrics (
            request_id          String,
            deployment_id       LowCardinality(String),
            model_id            LowCardinality(String),
            tenant_id           LowCardinality(String) DEFAULT 'local',
            timestamp           DateTime64(3, 'UTC'),
            duration_ms         Float64,
            ttft_ms             Float64 DEFAULT 0,
            prompt_tokens       UInt32 DEFAULT 0,
            completion_tokens   UInt32 DEFAULT 0,
            total_tokens        UInt32 DEFAULT 0,
            prompt_per_second   Float32 DEFAULT 0,
            predict_per_second  Float32 DEFAULT 0,
            status_code         UInt16 DEFAULT 200,
            error               String DEFAULT '',
            finish_reason       LowCardinality(String) DEFAULT '',
            temperature         Float32 DEFAULT 0,
            max_tokens          UInt32 DEFAULT 0,
            endpoint            LowCardinality(String) DEFAULT 'chat'
        )
        ENGINE = MergeTree()
        ORDER BY (deployment_id, timestamp)
        PARTITION BY toYYYYMMDD(timestamp)
        TTL toDateTime(timestamp) + INTERVAL 90 DAY
    """)


def _ch():
    from ..storage.clickhouse_client import get_client

    global _table_ready
    client = get_client()
    if client is None:
        return None
    if not _table_ready:
        _ensure_table()
        _table_ready = True
    return client


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_inference(
    *,
    deployment_id: str,
    model_id: str,
    duration_ms: float,
    ttft_ms: float = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    prompt_per_second: float = 0,
    predict_per_second: float = 0,
    status_code: int = 200,
    error: str = "",
    finish_reason: str = "",
    temperature: float = 0,
    max_tokens: int = 0,
    endpoint: str = "chat",
    tenant_id: str = "local",
    request_id: str | None = None,
) -> None:
    """Insert a single inference event. Failures are swallowed — metrics must never break inference."""
    client = _ch()
    if client is None:
        return
    try:
        client.insert(
            "inference_metrics",
            [[
                request_id or str(uuid.uuid4()),
                deployment_id,
                model_id,
                tenant_id,
                _now(),
                float(duration_ms),
                float(ttft_ms),
                int(prompt_tokens),
                int(completion_tokens),
                int(total_tokens),
                float(prompt_per_second),
                float(predict_per_second),
                int(status_code),
                error[:1000],
                finish_reason[:64],
                float(temperature),
                int(max_tokens),
                endpoint,
            ]],
            column_names=[
                "request_id", "deployment_id", "model_id", "tenant_id", "timestamp",
                "duration_ms", "ttft_ms", "prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_per_second", "predict_per_second", "status_code", "error",
                "finish_reason", "temperature", "max_tokens", "endpoint",
            ],
        )
    except Exception as e:
        logger.warning("Failed to record inference metric: %s", e)


def _truncate(s: str, limit: int) -> str:
    if not s:
        return ""
    return s if len(s) <= limit else s[:limit]


def record_trace(
    *,
    request_id: str,
    model_id: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    duration_ms: float,
    ttft_ms: float,
    status_code: int,
    error: str,
    request_type: str,
    is_stream: bool,
) -> None:
    """Insert a row into llm_traces so proxy inference shows up in the Traces UI.

    Mirrors the subset of columns that the Go engine writes for gateway traffic.
    Failures are swallowed — tracing must never break inference.
    """
    client = _ch()
    if client is None:
        return

    usage = response_body.get("usage", {}) or {}
    timings = response_body.get("timings", {}) or {}
    choices = response_body.get("choices", []) or []

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens))
    tokens_per_s = float(timings.get("predicted_per_second", 0) or 0)

    messages_in = request_body.get("messages") or []
    input_text = ""
    if isinstance(messages_in, list) and messages_in:
        input_text = "\n".join(
            str(m.get("content", "")) for m in messages_in if isinstance(m, dict)
        )
    elif isinstance(request_body.get("prompt"), str):
        input_text = request_body["prompt"]
    elif isinstance(request_body.get("input"), (str, list)):
        inp = request_body["input"]
        input_text = inp if isinstance(inp, str) else json.dumps(inp)

    output_message: dict[str, Any] = {}
    output_text = ""
    finish_reason = ""
    if choices and isinstance(choices[0], dict):
        ch = choices[0]
        finish_reason = ch.get("finish_reason", "") or ""
        msg = ch.get("message") or {}
        if isinstance(msg, dict):
            output_message = msg
            output_text = str(msg.get("content", "") or "")
        elif "text" in ch:
            output_text = str(ch.get("text", "") or "")
            output_message = {"role": "assistant", "content": output_text}

    is_error = 1 if status_code >= 400 else 0
    error_category = ""
    if is_error:
        if status_code == 502:
            error_category = "upstream"
        elif status_code == 503:
            error_category = "unavailable"
        elif 500 <= status_code < 600:
            error_category = "server"
        elif 400 <= status_code < 500:
            error_category = "client"

    try:
        client.insert(
            "llm_traces",
            [[
                request_id,
                _now(),
                model_id or "",
                "llama.cpp",
                float(duration_ms),
                float(ttft_ms),
                int(prompt_tokens),
                int(completion_tokens),
                int(total_tokens),
                int(is_error),
                error_category,
                _truncate(error, 1000),
                request_type,
                1 if is_stream else 0,
                _truncate(input_text, 8000),
                _truncate(output_text, 8000),
                _truncate(json.dumps(messages_in) if messages_in else "", 16000),
                _truncate(json.dumps(output_message) if output_message else "", 8000),
                _truncate(finish_reason, 64),
                float(tokens_per_s),
            ]],
            column_names=[
                "request_id", "timestamp", "selected_model", "provider",
                "latency_ms", "ttft_ms",
                "tokens_in", "tokens_out", "total_tokens",
                "is_error", "error_category", "error_message",
                "request_type", "is_stream",
                "input_text", "output_text", "input_messages", "output_message",
                "finish_reason", "tokens_per_s",
            ],
        )
    except Exception as e:
        logger.warning("Failed to record llm_traces row: %s", e)


def get_aggregate_stats(deployment_id: str, minutes: int = 60) -> dict[str, Any]:
    """Aggregate stats over the last `minutes` for a deployment."""
    client = _ch()
    if client is None:
        return _empty_stats()

    start = _now() - timedelta(minutes=minutes)
    try:
        r = client.query(
            """
            SELECT
                count() AS total,
                countIf(status_code < 400) AS successful,
                countIf(status_code >= 400) AS failed,
                avgIf(duration_ms, status_code < 400) AS avg_latency,
                quantile(0.50)(duration_ms) AS p50,
                quantile(0.95)(duration_ms) AS p95,
                quantile(0.99)(duration_ms) AS p99,
                sum(total_tokens) AS total_tokens,
                sum(prompt_tokens) AS prompt_tokens,
                sum(completion_tokens) AS completion_tokens,
                avg(predict_per_second) AS avg_throughput
            FROM inference_metrics
            WHERE deployment_id = {did:String} AND timestamp >= {start:DateTime64(3)}
            """,
            parameters={"did": deployment_id, "start": start},
        )
        if not r.result_rows:
            return _empty_stats()
        row = r.result_rows[0]
        total = int(row[0] or 0)
        successful = int(row[1] or 0)
        failed = int(row[2] or 0)
        return {
            "total_inferences": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 100.0,
            "avg_latency_ms": float(row[3] or 0),
            "p50_latency_ms": float(row[4] or 0),
            "p95_latency_ms": float(row[5] or 0),
            "p99_latency_ms": float(row[6] or 0),
            "total_tokens": int(row[7] or 0),
            "prompt_tokens": int(row[8] or 0),
            "completion_tokens": int(row[9] or 0),
            "avg_throughput_tps": float(row[10] or 0),
        }
    except Exception as e:
        logger.warning("Failed to query aggregate stats: %s", e)
        return _empty_stats()


def _empty_stats() -> dict[str, Any]:
    return {
        "total_inferences": 0, "successful": 0, "failed": 0,
        "success_rate": 100.0, "avg_latency_ms": 0,
        "p50_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0,
        "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "avg_throughput_tps": 0,
    }


def get_time_series(
    deployment_id: str,
    minutes: int = 60,
    period_seconds: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Return time-series buckets for latency, invocations, tokens, errors."""
    client = _ch()
    if client is None:
        return _empty_time_series()

    start = _now() - timedelta(minutes=minutes)
    bucket_seconds = max(period_seconds, 10)

    try:
        r = client.query(
            f"""
            SELECT
                toStartOfInterval(timestamp, INTERVAL {bucket_seconds} SECOND) AS bucket,
                count() AS invocations,
                avg(duration_ms) AS avg_latency,
                quantile(0.95)(duration_ms) AS p95_latency,
                sum(total_tokens) AS tokens,
                countIf(status_code >= 400) AS errors,
                avg(predict_per_second) AS throughput
            FROM inference_metrics
            WHERE deployment_id = {{did:String}} AND timestamp >= {{start:DateTime64(3)}}
            GROUP BY bucket
            ORDER BY bucket ASC
            """,
            parameters={"did": deployment_id, "start": start},
        )
        invocations: list[dict[str, Any]] = []
        latency: list[dict[str, Any]] = []
        p95: list[dict[str, Any]] = []
        tokens: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        throughput: list[dict[str, Any]] = []
        for row in r.result_rows:
            ts = row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0])
            invocations.append({"timestamp": ts, "value": int(row[1] or 0)})
            latency.append({"timestamp": ts, "value": float(row[2] or 0)})
            p95.append({"timestamp": ts, "value": float(row[3] or 0)})
            tokens.append({"timestamp": ts, "value": int(row[4] or 0)})
            errors.append({"timestamp": ts, "value": int(row[5] or 0)})
            throughput.append({"timestamp": ts, "value": float(row[6] or 0)})
        return {
            "invocations": invocations,
            "model_latency": latency,
            "p95_latency": p95,
            "tokens": tokens,
            "errors": errors,
            "throughput": throughput,
            "cpu_utilization": [],
            "memory_utilization": [],
            "gpu_utilization": [],
            "gpu_memory_utilization": [],
        }
    except Exception as e:
        logger.warning("Failed to query time series: %s", e)
        return _empty_time_series()


def _empty_time_series() -> dict[str, list[dict[str, Any]]]:
    return {
        "invocations": [], "model_latency": [], "p95_latency": [],
        "tokens": [], "errors": [], "throughput": [],
        "cpu_utilization": [], "memory_utilization": [],
        "gpu_utilization": [], "gpu_memory_utilization": [],
    }


def get_latest(deployment_id: str) -> dict[str, Any]:
    """Latest single inference for the deployment, for the 'latest' card."""
    client = _ch()
    if client is None:
        return {}
    try:
        r = client.query(
            """
            SELECT timestamp, duration_ms
            FROM inference_metrics
            WHERE deployment_id = {did:String}
            ORDER BY timestamp DESC LIMIT 1
            """,
            parameters={"did": deployment_id},
        )
        if not r.result_rows:
            return {}
        row = r.result_rows[0]
        ts = row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0])
        return {
            "timestamp": ts,
            "model_latency_ms": float(row[1] or 0),
        }
    except Exception as e:
        logger.warning("Failed to query latest metric: %s", e)
        return {}
