"""SQLite-backed append-only ledger store.

One table, three indices matching the three real access patterns:
    (objective_id, ts)   — time-series per objective
    (parent_id)          — chain reconstruction
    (type, ts)           — typed windowed queries

Public API is append-only. `get`, `chain`, `time_series`, and `recent`
are read helpers; there is no update or delete on purpose.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union

from opentracy._env import env

from .entry import LedgerEntry


_DEFAULT_PATH = Path.home() / ".opentracy" / "harness_ledger.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS harness_ledger (
    id              TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    type            TEXT NOT NULL,
    objective_id    TEXT,
    subject         TEXT,
    agent           TEXT,
    parameters_in   TEXT NOT NULL DEFAULT '{}',
    data            TEXT NOT NULL DEFAULT '{}',
    parent_id       TEXT,
    tags            TEXT NOT NULL DEFAULT '[]',
    duration_ms     INTEGER,
    cost_usd        REAL,
    outcome         TEXT
);

CREATE INDEX IF NOT EXISTS idx_ledger_objective_ts ON harness_ledger (objective_id, ts);
CREATE INDEX IF NOT EXISTS idx_ledger_parent       ON harness_ledger (parent_id);
CREATE INDEX IF NOT EXISTS idx_ledger_type_ts      ON harness_ledger (type, ts);
"""


class LedgerStore:
    """Append-only SQLite ledger. Thread-safe via a per-instance lock."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None):
        if db_path is None:
            override = env("LEDGER_DB", "")
            db_path = override or str(_DEFAULT_PATH)
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def append(self, entry: LedgerEntry) -> str:
        """Insert one entry. Returns the entry id."""
        row = (
            entry.id,
            entry.ts,
            entry.type,
            entry.objective_id,
            entry.subject,
            entry.agent,
            json.dumps(entry.parameters_in, default=str),
            json.dumps(entry.data, default=str),
            entry.parent_id,
            json.dumps(entry.tags),
            entry.duration_ms,
            entry.cost_usd,
            entry.outcome,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO harness_ledger (
                    id, ts, type, objective_id, subject, agent,
                    parameters_in, data, parent_id, tags,
                    duration_ms, cost_usd, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            self._conn.commit()
        return entry.id

    def get(self, entry_id: str) -> Optional[LedgerEntry]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM harness_ledger WHERE id = ?", (entry_id,)
            ).fetchone()
        return _row_to_entry(row) if row else None

    def chain(self, root_id: str) -> list[LedgerEntry]:
        """Return the full causal chain rooted at `root_id` in BFS order.

        Order within a level is (ts ASC, rowid ASC) so the result is stable
        across equal timestamps.
        """
        visited: list[LedgerEntry] = []
        seen: set[str] = set()
        frontier = [root_id]

        with self._lock:
            while frontier:
                placeholders = ",".join("?" * len(frontier))
                rows = self._conn.execute(
                    f"SELECT * FROM harness_ledger WHERE id IN ({placeholders}) "
                    f"ORDER BY ts ASC, rowid ASC",
                    tuple(frontier),
                ).fetchall()
                for row in rows:
                    if row["id"] in seen:
                        continue
                    visited.append(_row_to_entry(row))
                    seen.add(row["id"])

                child_rows = self._conn.execute(
                    f"SELECT id FROM harness_ledger WHERE parent_id IN ({placeholders}) "
                    f"ORDER BY ts ASC, rowid ASC",
                    tuple(frontier),
                ).fetchall()
                frontier = [r["id"] for r in child_rows if r["id"] not in seen]
        return visited

    def time_series(
        self,
        objective_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10000,
    ) -> list[LedgerEntry]:
        """Entries for an objective in a time window, oldest-first."""
        conditions = ["objective_id = ?"]
        params: list = [objective_id]
        if start:
            conditions.append("ts >= ?")
            params.append(start)
        if end:
            conditions.append("ts <= ?")
            params.append(end)
        where = " AND ".join(conditions)
        sql = (
            f"SELECT * FROM harness_ledger WHERE {where} "
            f"ORDER BY ts ASC, rowid ASC LIMIT ?"
        )
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_entry(r) for r in rows]

    def recent(self, limit: int = 100) -> list[LedgerEntry]:
        """Most recently inserted entries, newest-first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM harness_ledger ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        id=row["id"],
        ts=row["ts"],
        type=row["type"],
        objective_id=row["objective_id"],
        subject=row["subject"],
        agent=row["agent"],
        parameters_in=json.loads(row["parameters_in"] or "{}"),
        data=json.loads(row["data"] or "{}"),
        parent_id=row["parent_id"],
        tags=json.loads(row["tags"] or "[]"),
        duration_ms=row["duration_ms"],
        cost_usd=row["cost_usd"],
        outcome=row["outcome"],
    )
