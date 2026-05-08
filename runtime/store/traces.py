"""DuckDB-backed read layer for traces.

Append path is unchanged: runtime/executor/tracing.write_trace still writes
JSONL. This module is read-only — it queries the union of:

  traces/raw/*.jsonl                 — live, today + uncompacted days
  traces/parquet/dt=*/.../*.parquet  — compacted historical partitions

via DuckDB's read_json_auto + read_parquet, glued by UNION ALL BY NAME so a
schema drift (older traces missing session_id/history) becomes NULLs instead
of an error.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "traces" / "raw"
PARQUET_DIR = ROOT / "traces" / "parquet"


def _connect() -> duckdb.DuckDBPyConnection:
    """Fresh in-memory connection per call. Cheap (~microseconds) and keeps
    file metadata fresh; matters because the runtime appends JSONL constantly
    and we don't want to read a stale schema."""
    return duckdb.connect(":memory:")


def _fetch_dicts(cur: duckdb.DuckDBPyConnection) -> list[dict]:
    """Convert a DuckDB cursor result to list[dict] without pulling pyarrow.
    Nested STRUCT/LIST columns come back as nested dicts/lists already."""
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _parquet_dates() -> set[str]:
    if not PARQUET_DIR.exists():
        return set()
    return {p.name[len("dt=") :] for p in PARQUET_DIR.glob("dt=*") if any(p.rglob("*.parquet"))}


def _jsonl_paths_excluding(dates: set[str]) -> list[str]:
    if not RAW_DIR.exists():
        return []
    return [
        p.as_posix()
        for p in sorted(RAW_DIR.glob("*.jsonl"))
        if p.stem not in dates and p.stat().st_size > 0
    ]


def _has_parquet() -> bool:
    return bool(_parquet_dates())


def _has_jsonl() -> bool:
    return RAW_DIR.exists() and any(RAW_DIR.glob("*.jsonl"))


_SCALAR_COLS: dict[str, str] = {
    "trace_id": "VARCHAR",
    "timestamp": "VARCHAR",
    "request": "VARCHAR",
    "response": "VARCHAR",
    "duration_ms": "DOUBLE",
    "success": "BOOLEAN",
    "error": "VARCHAR",
    "agent_version": "VARCHAR",
    "session_id": "VARCHAR",
}


def _source_columns(con: duckdb.DuckDBPyConnection, table_expr: str) -> set[str]:
    """Names of top-level columns the source provides."""
    cur = con.execute(f"DESCRIBE SELECT * FROM {table_expr} LIMIT 0")
    return {row[0] for row in cur.fetchall()}


def _normalized_projection(con: duckdb.DuckDBPyConnection, table_expr: str) -> str:
    """Cast every column to a stable, source-agnostic schema before union.

    DuckDB infers `error` (and nested `stages.error`) as JSON when read from
    JSONL but as VARCHAR from Parquet. UNION ALL BY NAME of mismatched types
    can drag a non-JSON string into a JSON-typed column and fail at
    materialization time. Force all string-ish columns to VARCHAR up-front,
    rebuild stages/history with explicit field casts so the inner STRUCT
    types match too. Missing columns are filled with typed NULLs so the
    projection works against older sources that lack session_id/history."""
    have = _source_columns(con, table_expr)

    parts: list[str] = []
    for name, sqltype in _SCALAR_COLS.items():
        if name in have:
            parts.append(f"CAST({name} AS {sqltype}) AS {name}")
        else:
            parts.append(f"CAST(NULL AS {sqltype}) AS {name}")

    if "stages" in have:
        parts.append(
            "[STRUCT_PACK("
            "stage := CAST(s.stage AS VARCHAR),"
            "technique := CAST(s.technique AS VARCHAR),"
            "variant := CAST(s.variant AS VARCHAR),"
            "duration_ms := CAST(s.duration_ms AS DOUBLE),"
            "docs_in := CAST(s.docs_in AS BIGINT),"
            "docs_out := CAST(s.docs_out AS BIGINT),"
            "response_set := CAST(s.response_set AS BOOLEAN),"
            "routing_model := CAST(s.routing_model AS VARCHAR),"
            "error := CAST(s.error AS VARCHAR)"
            ") FOR s IN COALESCE(stages, [])] AS stages"
        )
    else:
        parts.append("CAST([] AS STRUCT(stage VARCHAR)[]) AS stages")

    if "history" in have:
        parts.append(
            "[STRUCT_PACK("
            "role := CAST(h.role AS VARCHAR),"
            "content := CAST(h.content AS VARCHAR)"
            ") FOR h IN COALESCE(history, [])] AS history"
        )
    else:
        parts.append("CAST([] AS STRUCT(role VARCHAR, content VARCHAR)[]) AS history")

    if "metadata" in have:
        parts.append("CAST(metadata AS JSON) AS metadata")
    else:
        parts.append("CAST(NULL AS JSON) AS metadata")

    return f"SELECT {', '.join(parts)} FROM {table_expr}"


def _traces_view_sql(con: duckdb.DuckDBPyConnection) -> str | None:
    """Build the unified-source SQL for the given DuckDB connection.

    Per-day source picker: a day with a Parquet partition is read from
    Parquet; otherwise from JSONL. JSONL is kept on disk after compaction
    as an audit trail, so we must NOT read it twice. Each source is
    individually projected into a normalized schema BEFORE the union, so
    type mismatches (JSON vs VARCHAR for `error`) and missing columns
    (older traces lacking `session_id`/`history`) don't poison the read."""
    pq_dates = _parquet_dates()
    pq_glob = str(PARQUET_DIR / "**" / "*.parquet")
    jsonl_paths = _jsonl_paths_excluding(pq_dates)

    parts: list[str] = []
    if pq_dates:
        parts.append(
            _normalized_projection(
                con, f"read_parquet('{pq_glob}', union_by_name=true)"
            )
        )
    if jsonl_paths:
        list_lit = "[" + ", ".join(f"'{p}'" for p in jsonl_paths) + "]"
        parts.append(
            _normalized_projection(
                con, f"read_json_auto({list_lit}, union_by_name=true)"
            )
        )

    if not parts:
        return None

    return " UNION ALL BY NAME ".join(f"({p})" for p in parts)


def available_dates() -> list[str]:
    """Filesystem-driven (faster than DuckDB) — covers JSONL and Parquet
    partitions. Returned newest-first."""
    dates: set[str] = set()
    if RAW_DIR.exists():
        for p in RAW_DIR.glob("*.jsonl"):
            dates.add(p.stem)
    if PARQUET_DIR.exists():
        for p in PARQUET_DIR.glob("dt=*"):
            stem = p.name[len("dt=") :]
            dates.add(stem)
    return sorted(dates, reverse=True)


def _pick_routing_model(stages: list[dict] | None) -> Optional[str]:
    if not stages:
        return None
    for s in stages:
        if not isinstance(s, dict):
            continue
        if s.get("stage") == "route" and s.get("routing_model"):
            return str(s["routing_model"])
    return None


def _s(v: Any) -> Optional[str]:
    """Coerce DuckDB scalars (UUID, datetime, etc.) to str. None stays None."""
    return None if v is None else str(v)


def _ts(v: Any) -> Optional[str]:
    """Normalize timestamp to ISO-Z form, matching the JSONL on-disk shape.

    DuckDB infers TIMESTAMP for ISO-shaped strings and renders them as
    'YYYY-MM-DD HH:MM:SS[.ffffff]' on cast. The runtime API contract has
    always been 'YYYY-MM-DDTHH:MM:SS.fffZ'; the UI parses with new Date(...)
    which accepts both, but we preserve the original format so existing
    string equality (e.g. session.started_at == trace.timestamp) keeps
    working."""
    if v is None:
        return None
    s = str(v)
    if "T" in s and s.endswith("Z"):
        return s
    # 'YYYY-MM-DD HH:MM:SS[.ffffff]' -> 'YYYY-MM-DDTHH:MM:SS.fffZ'
    body = s.replace(" ", "T")
    if "." in body:
        head, frac = body.split(".", 1)
        frac = (frac + "000")[:3]  # truncate/pad to milliseconds
        body = f"{head}.{frac}"
    return body + "Z"


def _row_to_summary(row: dict) -> dict:
    stages = row.get("stages") or []
    history = row.get("history") or []
    return {
        "trace_id": _s(row.get("trace_id")) or "",
        "timestamp": _ts(row.get("timestamp")) or "",
        "request": row.get("request") or "",
        "response": row.get("response"),
        "duration_ms": float(row.get("duration_ms") or 0),
        "success": bool(row.get("success") or False),
        "error": _s(row.get("error")),
        "agent_version": _s(row.get("agent_version")),
        "n_stages": len(stages),
        "routing_model": _pick_routing_model(stages),
        "session_id": _s(row.get("session_id")),
        "n_turns": len(history) + 1,
    }


def _parse_metadata(v: Any) -> dict:
    """metadata is projected as JSON to keep the union type-stable across
    parquet (MAP) and JSONL (STRUCT) sources. DuckDB returns JSON-typed
    columns as Python strings; parse back to a dict for the API contract."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            import json as _json
            parsed = _json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _row_to_detail(row: dict) -> dict:
    summary = _row_to_summary(row)
    stages_raw = row.get("stages") or []
    history_raw = row.get("history") or []
    stages = [
        {
            "stage": _s(s.get("stage")),
            "technique": _s(s.get("technique")) or "",
            "variant": _s(s.get("variant")) or "",
            "duration_ms": float(s.get("duration_ms") or 0),
            "docs_in": int(s.get("docs_in") or 0),
            "docs_out": int(s.get("docs_out") or 0),
            "response_set": s.get("response_set"),
            "routing_model": _s(s.get("routing_model")),
            "error": _s(s.get("error")),
        }
        for s in stages_raw
        if isinstance(s, dict)
    ]
    history = [
        {"role": h.get("role") or "", "content": h.get("content") or ""}
        for h in history_raw
        if isinstance(h, dict)
    ]
    return {
        **summary,
        "stages": stages,
        "metadata": _parse_metadata(row.get("metadata")),
        "history": history,
    }


def query_traces(
    *,
    date: Optional[str] = None,
    success: Optional[bool] = None,
    agent_version: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return (page, total_filtered). Newest-first."""
    with _connect() as con:
        src = _traces_view_sql(con)
        if src is None:
            return [], 0

        where: list[str] = []
        params: list[Any] = []
        if date:
            where.append("substr(timestamp, 1, 10) = ?")
            params.append(date)
        if success is not None:
            where.append("COALESCE(success, false) = ?")
            params.append(bool(success))
        if agent_version:
            where.append("agent_version = ?")
            params.append(agent_version)
        if q:
            needle = f"%{q.lower()}%"
            where.append(
                "(LOWER(COALESCE(request, '')) LIKE ? OR LOWER(COALESCE(response, '')) LIKE ?)"
            )
            params.extend([needle, needle])

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        base = f"WITH t AS ({src}) SELECT * FROM t{where_sql}"
        count_sql = f"WITH t AS ({src}) SELECT COUNT(*) FROM t{where_sql}"

        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        page_sql = f"{base} ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"

        total = con.execute(count_sql, params).fetchone()[0]
        cur = con.execute(page_sql, params)
        rows = _fetch_dicts(cur)

    return [_row_to_summary(r) for r in rows], int(total)


def get_trace(trace_id: str) -> Optional[dict]:
    with _connect() as con:
        src = _traces_view_sql(con)
        if src is None:
            return None
        sql = f"WITH t AS ({src}) SELECT * FROM t WHERE trace_id = ? LIMIT 1"
        cur = con.execute(sql, [trace_id])
        rows = _fetch_dicts(cur)
    if not rows:
        return None
    return _row_to_detail(rows[0])


def get_session_turns(session_id: str) -> list[dict]:
    """Return chronological list of turn-level dicts for a session_id."""
    with _connect() as con:
        src = _traces_view_sql(con)
        if src is None:
            return []
        sql = (
            f"WITH t AS ({src}) "
            "SELECT trace_id, timestamp, request, response, success, error, "
            "agent_version, duration_ms FROM t "
            "WHERE session_id = ? ORDER BY timestamp ASC"
        )
        cur = con.execute(sql, [session_id])
        rows = _fetch_dicts(cur)
    return [
        {
            "trace_id": _s(r.get("trace_id")) or "",
            "timestamp": _ts(r.get("timestamp")) or "",
            "request": r.get("request") or "",
            "response": r.get("response"),
            "success": bool(r.get("success") or False),
            "error": _s(r.get("error")),
            "agent_version": _s(r.get("agent_version")),
            "duration_ms": float(r.get("duration_ms") or 0),
        }
        for r in rows
    ]


def metrics_traces_window(window_days: int = 7) -> dict:
    """Aggregate today + last `window_days` for /metrics/overview.
    Returns: today_count, active_5min, resolution_rate, avg_latency_ms."""
    with _connect() as con:
        src = _traces_view_sql(con)
        if src is None:
            return {
                "today_count": 0,
                "active_5min": 0,
                "resolution_rate": None,
                "avg_latency_ms": None,
            }

        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        five_min_ago = (
            (now - timedelta(minutes=5))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        window_start = (now - timedelta(days=window_days)).date().isoformat()

        sql = f"""
        WITH t AS ({src}),
             today AS (
                SELECT * FROM t WHERE substr(timestamp, 1, 10) = ?
             ),
             windowed AS (
                SELECT * FROM t WHERE substr(timestamp, 1, 10) >= ?
             )
        SELECT
          (SELECT COUNT(*) FROM today)                                       AS today_count,
          (SELECT COUNT(*) FROM today WHERE timestamp > ?)                   AS active_5min,
          (SELECT
             CASE WHEN COUNT(*) = 0 THEN NULL
                  ELSE SUM(CASE WHEN response IS NOT NULL
                                AND NOT EXISTS (
                                  SELECT 1 FROM UNNEST(COALESCE(stages, [])) AS u(s)
                                  WHERE s.error IS NOT NULL
                                )
                           THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
             END
           FROM windowed)                                                    AS resolution_rate,
          (SELECT
             CASE WHEN COUNT(*) = 0 THEN NULL
                  ELSE AVG(COALESCE(duration_ms, 0))
             END
           FROM windowed)                                                    AS avg_latency_ms
        """

        cur = con.execute(sql, [today, window_start, five_min_ago])
        rows = _fetch_dicts(cur)
    row = rows[0] if rows else {}

    return {
        "today_count": int(row.get("today_count") or 0),
        "active_5min": int(row.get("active_5min") or 0),
        "resolution_rate": (
            float(row["resolution_rate"]) if row.get("resolution_rate") is not None else None
        ),
        "avg_latency_ms": (
            float(row["avg_latency_ms"]) if row.get("avg_latency_ms") is not None else None
        ),
    }
