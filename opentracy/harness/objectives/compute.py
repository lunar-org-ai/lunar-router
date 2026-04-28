"""Compute functions for the three initial harness objectives.

Each function is deterministic and side-effect-free. When ClickHouse is
disabled or the window has no data, the function returns an empty list —
the caller decides how to render that (null value in dashboards, skipped
scheduler run, etc.).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from opentracy.storage.clickhouse_client import get_client

from .schemas import ObjectiveMeasurement


def _window(window_hours: int) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=window_hours)
    return start, end


def _coerce(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _run(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    client = get_client()
    if client is None:
        return []
    result = client.query(sql, parameters=params)
    cols = result.column_names
    return [dict(zip(cols, row)) for row in result.result_rows]


def cost_per_successful_completion(
    window_hours: int = 168,
) -> list[ObjectiveMeasurement]:
    """USD cost per non-error completion, bucketed by selected_model."""
    start, end = _window(window_hours)
    now = datetime.now(timezone.utc).isoformat()

    rows = _run(
        """
        SELECT
            selected_model,
            sum(total_cost_usd)  AS total_cost,
            countIf(is_error = 0) AS successful
        FROM llm_traces
        WHERE timestamp >= {start:DateTime64(3)}
          AND timestamp <= {end:DateTime64(3)}
        GROUP BY selected_model
        HAVING successful > 0
        """,
        {"start": start, "end": end},
    )

    measurements: list[ObjectiveMeasurement] = []
    for row in rows:
        successful = int(row["successful"])
        value = _coerce(row["total_cost"]) / successful
        measurements.append(
            ObjectiveMeasurement(
                objective_id="cost_per_successful_completion",
                value=round(value, 8),
                unit="USD",
                sample_size=successful,
                window_start=start.isoformat(),
                window_end=end.isoformat(),
                dimension_values={"selected_model": str(row["selected_model"])},
                computed_at=now,
            )
        )
    return measurements


def p95_latency_ms(window_hours: int = 168) -> list[ObjectiveMeasurement]:
    """95th percentile end-to-end latency on non-error traffic per selected_model."""
    start, end = _window(window_hours)
    now = datetime.now(timezone.utc).isoformat()

    rows = _run(
        """
        SELECT
            selected_model,
            quantile(0.95)(latency_ms) AS p95,
            count() AS n
        FROM llm_traces
        WHERE timestamp >= {start:DateTime64(3)}
          AND timestamp <= {end:DateTime64(3)}
          AND is_error = 0
        GROUP BY selected_model
        HAVING n > 0
        """,
        {"start": start, "end": end},
    )

    measurements: list[ObjectiveMeasurement] = []
    for row in rows:
        measurements.append(
            ObjectiveMeasurement(
                objective_id="p95_latency_ms",
                value=round(_coerce(row["p95"]), 2),
                unit="ms",
                sample_size=int(row["n"]),
                window_start=start.isoformat(),
                window_end=end.isoformat(),
                dimension_values={"selected_model": str(row["selected_model"])},
                computed_at=now,
            )
        )
    return measurements


def domain_coverage_ratio(window_hours: int = 168) -> list[ObjectiveMeasurement]:
    """Fraction of traces with a known cluster_id (!= -1) over the window."""
    start, end = _window(window_hours)
    now = datetime.now(timezone.utc).isoformat()

    rows = _run(
        """
        SELECT
            countIf(cluster_id != -1) AS covered,
            count()                   AS total
        FROM llm_traces
        WHERE timestamp >= {start:DateTime64(3)}
          AND timestamp <= {end:DateTime64(3)}
        """,
        {"start": start, "end": end},
    )
    if not rows:
        return []

    row = rows[0]
    total = int(row["total"])
    covered = int(row["covered"] or 0)
    value = (covered / total) if total > 0 else None

    return [
        ObjectiveMeasurement(
            objective_id="domain_coverage_ratio",
            value=round(value, 4) if value is not None else None,
            unit="ratio",
            sample_size=total,
            window_start=start.isoformat(),
            window_end=end.isoformat(),
            computed_at=now,
        )
    ]
