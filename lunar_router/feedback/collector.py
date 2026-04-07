"""
Collect production traces from ClickHouse for the feedback loop.

Bridges the gap between the ClickHouse storage layer and the
TraceToTraining module. Queries real traces and converts them
into TraceRecord objects for Psi vector updates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import logging

from .trace_to_training import TraceRecord

logger = logging.getLogger(__name__)


def collect_traces(
    days: int = 7,
    model: Optional[str] = None,
    limit: int = 10000,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[TraceRecord]:
    """
    Query ClickHouse for recent production traces and convert to TraceRecord.

    Requires LUNAR_CH_ENABLED=true and a running ClickHouse instance.

    Args:
        days: Number of days to look back (ignored if start/end provided).
        model: Optional model filter.
        limit: Maximum traces to fetch.
        start: Custom start time.
        end: Custom end time.

    Returns:
        List of TraceRecord objects ready for TraceToTraining.
    """
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        logger.warning("ClickHouse disabled — returning empty traces. Set LUNAR_CH_ENABLED=true")
        return []

    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=days)

    conditions = [
        "timestamp >= {start:DateTime64(3)}",
        "timestamp <= {end:DateTime64(3)}",
    ]
    params: dict[str, Any] = {"start": start, "end": end, "limit": limit}

    if model:
        conditions.append("selected_model = {model:String}")
        params["model"] = model

    where = " AND ".join(conditions)

    sql = f"""
        SELECT
            request_id,
            selected_model,
            cluster_id,
            is_error,
            latency_ms,
            total_cost_usd,
            input_text,
            output_text,
            error_category
        FROM llm_traces
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT {{limit:UInt32}}
    """

    result = client.query(sql, parameters=params)

    traces = []
    for row in result.result_rows:
        (request_id, selected_model, cluster_id, is_error,
         latency_ms, total_cost_usd, input_text, output_text,
         error_category) = row

        traces.append(TraceRecord(
            request_id=str(request_id),
            selected_model=str(selected_model),
            cluster_id=int(cluster_id) if cluster_id is not None else 0,
            is_error=bool(is_error),
            latency_ms=float(latency_ms or 0),
            total_cost_usd=float(total_cost_usd or 0),
            input_text=str(input_text) if input_text else None,
            output_text=str(output_text) if output_text else None,
            error_category=str(error_category) if error_category else None,
        ))

    logger.info(f"Collected {len(traces)} traces from ClickHouse ({start} to {end})")
    return traces


def collect_quality_flags(
    days: int = 7,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> dict[str, bool]:
    """
    Collect quality issue flags from the memory store (TraceScanner results).

    Returns dict of {request_id: True} for traces flagged as bad quality.
    """
    from ..harness.memory_store import get_memory_store

    store = get_memory_store()
    entries = store.list(category="trace_issues")

    flags: dict[str, bool] = {}
    for entry in entries:
        data = entry.get("data", {})
        request_id = data.get("request_id") or data.get("trace_id")
        if request_id:
            flags[request_id] = True

    logger.info(f"Collected {len(flags)} quality flags from memory store")
    return flags


def collect_trace_embeddings(
    traces: list[TraceRecord],
    embedder=None,
) -> Optional[Any]:
    """
    Embed trace input_text for drift detection.

    Args:
        traces: Traces with input_text.
        embedder: PromptEmbedder instance. If None, returns None.

    Returns:
        numpy array of shape (N, d) or None.
    """
    if embedder is None:
        return None

    import numpy as np

    texts = [t.input_text for t in traces if t.input_text]
    if not texts:
        return None

    embeddings = embedder.embed_batch(texts)
    logger.info(f"Embedded {len(texts)} trace inputs for drift detection")
    return np.array(embeddings)
