"""Trace writer — persists ExecutionRecord as JSONL.

Format: one JSON object per line in traces/raw/<YYYY-MM-DD>.jsonl. Each line is
an envelope (trace_id, timestamp) wrapping the ExecutionRecord. This schema is
the input to evals (read these to score the agent), to the ledger (causal
chains start here), and to the harness proposer (failed traces become
candidates for proposals).

write_trace also publishes a summary event to TraceBus, the in-process
pub/sub for live SSE subscribers (the runtime's /traces/stream endpoint).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from runtime.executor.pipeline import ExecutionRecord

logger = logging.getLogger(__name__)

TRACES_DIR = Path("traces/raw")


class TraceBus:
    """In-process pub/sub for live trace events.

    write_trace runs in a sync FastAPI thread (uvicorn calls handlers in a
    threadpool), so publish() must be safe to call without an async context.
    Each subscriber gets a bounded asyncio.Queue (drop-oldest on overflow,
    so a slow client can never block trace writes). The asyncio loop is
    captured at startup via attach_loop()."""

    def __init__(self, *, max_queue: int = 200) -> None:
        self._subs: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_queue = max_queue

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once from the FastAPI lifespan so publish() can hop
        from the writer thread back onto the running event loop."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict[str, Any]) -> None:
        """Fan-out to all subscribers. Bounded — drops the oldest item if a
        subscriber is full."""
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            subs = list(self._subs)
        if not subs:
            return
        loop.call_soon_threadsafe(self._fanout, subs, event)

    @staticmethod
    def _fanout(subs: list[asyncio.Queue[dict[str, Any]]], event: dict[str, Any]) -> None:
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest to make room — slow consumers don't block writes.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass


bus = TraceBus()


def _summary_event(env: dict[str, Any]) -> dict[str, Any]:
    """Compact payload for SSE — never the full request/response bodies.
    The UI fetches detail on click via /traces/{trace_id}."""
    stages = env.get("stages") or []
    history = env.get("history") or []
    request = env.get("request") or ""
    return {
        "trace_id": env.get("trace_id") or "",
        "timestamp": env.get("timestamp") or "",
        "session_id": env.get("session_id"),
        "agent_version": env.get("agent_version"),
        "duration_ms": float(env.get("duration_ms") or 0),
        "success": bool(env.get("success") or False),
        "error": env.get("error"),
        "n_stages": len(stages),
        "n_turns": len(history) + 1,
        "request_preview": (request[:200] + "…") if len(request) > 200 else request,
    }


def _new_trace_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def envelope(
    record: ExecutionRecord,
    trace_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the on-disk dict from a record + optional ids."""
    return {
        "trace_id": trace_id or _new_trace_id(),
        "timestamp": timestamp or _now_iso(),
        **record.to_dict(),
    }


def write_trace(
    record: ExecutionRecord,
    traces_dir: Path | str = TRACES_DIR,
    trace_id: str | None = None,
) -> str:
    """Append the trace as JSONL, then publish a summary event to TraceBus.
    Returns the trace_id."""
    traces_dir = Path(traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    env = envelope(record, trace_id=trace_id)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = traces_dir / f"{today}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(env, ensure_ascii=False) + "\n")

    try:
        bus.publish(_summary_event(env))
    except Exception:
        # The pub/sub fan-out must never break the write path.
        logger.exception("TraceBus publish failed for %s", env.get("trace_id"))

    return env["trace_id"]
