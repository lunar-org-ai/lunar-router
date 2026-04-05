"""ClickHouse storage for local deployments."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_tables_ready = False


def _ensure_tables() -> None:
    from ..storage.clickhouse_client import get_client

    client = get_client()
    if client is None:
        return

    client.command("""
        CREATE TABLE IF NOT EXISTS local_deployments (
            id              String,
            model_id        String,
            model_path      String,
            engine          LowCardinality(String) DEFAULT 'vllm',
            status          LowCardinality(String),
            endpoint_name   String DEFAULT '',
            endpoint_url    String DEFAULT '',
            pid             UInt64 DEFAULT 0,
            port            UInt32 DEFAULT 0,
            instance_type   String DEFAULT 'local-gpu',
            config          String DEFAULT '{}',
            error_message   String DEFAULT '',
            error_code      String DEFAULT '',
            tenant_id       String DEFAULT 'local',
            scaling         String DEFAULT '{}',
            created_at      DateTime64(3, 'UTC'),
            updated_at      DateTime64(3, 'UTC')
        ) ENGINE = ReplacingMergeTree(updated_at) ORDER BY (id)
    """)


def _ch():
    from ..storage.clickhouse_client import get_client

    global _tables_ready
    client = get_client()
    if client is None:
        return None
    if not _tables_ready:
        _ensure_tables()
        _tables_ready = True
    return client


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_dt(v: Any) -> str | None:
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _row_to_dict(column_names: list[str], row: tuple) -> dict:
    d = dict(zip(column_names, row))
    for field in ("config", "scaling"):
        if isinstance(d.get(field), str) and d[field]:
            try:
                d[field] = json.loads(d[field])
            except Exception:
                pass
    for field in ("created_at", "updated_at"):
        d[field] = _serialize_dt(d.get(field))
    # Ensure numeric types
    d["pid"] = int(d.get("pid", 0))
    d["port"] = int(d.get("port", 0))
    return d


def insert_deployment(
    deployment_id: str,
    model_id: str,
    model_path: str,
    port: int,
    instance_type: str = "local-gpu",
    config: dict | None = None,
    scaling: dict | None = None,
) -> None:
    client = _ch()
    if client is None:
        return
    now = _now()
    client.insert(
        "local_deployments",
        [[
            deployment_id, model_id, model_path, "vllm",
            "creating",  # initial status
            model_id,    # endpoint_name
            "",          # endpoint_url (set once healthy)
            0,           # pid (set once launched)
            port,
            instance_type,
            json.dumps(config or {}),
            "", "",      # error_message, error_code
            "local",     # tenant_id
            json.dumps(scaling or {}),
            now, now,
        ]],
        column_names=[
            "id", "model_id", "model_path", "engine",
            "status", "endpoint_name", "endpoint_url",
            "pid", "port", "instance_type", "config",
            "error_message", "error_code", "tenant_id", "scaling",
            "created_at", "updated_at",
        ],
    )


def update_deployment(deployment_id: str, **fields: Any) -> None:
    client = _ch()
    if client is None:
        return

    existing = get_deployment(deployment_id)
    if existing is None:
        return

    now = _now()
    existing["updated_at"] = now

    for k, v in fields.items():
        if k in ("config", "scaling") and isinstance(v, dict):
            v = json.dumps(v)
        existing[k] = v

    cols = [
        "id", "model_id", "model_path", "engine",
        "status", "endpoint_name", "endpoint_url",
        "pid", "port", "instance_type", "config",
        "error_message", "error_code", "tenant_id", "scaling",
        "created_at", "updated_at",
    ]
    row = []
    for c in cols:
        val = existing.get(c)
        if c in ("config", "scaling") and isinstance(val, dict):
            val = json.dumps(val)
        row.append(val)

    client.insert("local_deployments", [row], column_names=cols)


def get_deployment(deployment_id: str) -> Optional[dict]:
    client = _ch()
    if client is None:
        return None

    r = client.query(
        "SELECT * FROM local_deployments FINAL WHERE id = {did:String}",
        parameters={"did": deployment_id},
    )
    if not r.result_rows:
        return None
    return _row_to_dict(r.column_names, r.result_rows[0])


def list_deployments(statuses: list[str] | None = None) -> list[dict]:
    client = _ch()
    if client is None:
        return []

    if statuses:
        placeholders = ", ".join(f"'{s}'" for s in statuses)
        q = (
            f"SELECT * FROM local_deployments FINAL "
            f"WHERE status IN ({placeholders}) "
            f"ORDER BY created_at DESC"
        )
    else:
        q = "SELECT * FROM local_deployments FINAL ORDER BY created_at DESC"

    r = client.query(q)
    return [_row_to_dict(r.column_names, row) for row in r.result_rows]


def delete_deployment(deployment_id: str) -> None:
    client = _ch()
    if client is None:
        return
    client.command(
        f"ALTER TABLE local_deployments DELETE WHERE id = '{deployment_id}'"
    )
