"""ToolKit — tools available to harness agents.

Each tool is an async callable that returns a dict.
Agents request tool calls; the runner executes them and feeds results back.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ToolKit:
    """Tools available to harness agents for data access."""

    def __init__(self, engine_url: Optional[str] = None):
        self._engine_url = engine_url or os.environ.get("LUNAR_ENGINE_URL", "http://localhost:8080")
        self._tools: dict[str, Callable] = {
            "query_traces": self.query_traces,
            "query_summary": self.query_summary,
            "embed_texts": self.embed_texts,
            "cluster_texts": self.cluster_texts,
            "read_secrets": self.read_secrets,
            "list_datasets": self.list_datasets,
            "list_models": self.list_models,
            "add_traces": self.add_traces,
            "read_memory": self.read_memory,
            "write_memory": self.write_memory,
            "read_feedback": self.read_feedback,
        }
        self._descriptions: dict[str, str] = {
            "query_traces": "Query recent LLM traces. Args: days (int=7), model (str=''), limit (int=50)",
            "query_summary": "Get aggregated stats. Args: days (int=7)",
            "embed_texts": "Generate MiniLM embeddings. Args: texts (list[str])",
            "cluster_texts": "KMeans cluster texts. Args: texts (list[str]), k (int=5)",
            "read_secrets": "List configured API key providers. No args.",
            "list_datasets": "List domain datasets. Args: status (str='')",
            "list_models": "List available LLM models. No args.",
            "add_traces": "Add traces to ClickHouse. Args: traces (list[dict]) each with input/output/model",
            "read_memory": "Query memory entries. Args: agent (str=''), category (str=''), tags (list[str]=[]), limit (int=10)",
            "write_memory": "Save a memory note. Args: agent (str), content (str), tags (list[str]=[]), category (str='note')",
            "read_feedback": "Query user feedback (false positive dismissals). Args: issue_type (str=''), model (str=''), limit (int=20)",
        }

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def available(self) -> list[dict[str, str]]:
        """Return tool descriptions for the LLM."""
        return [{"name": k, "description": v} for k, v in self._descriptions.items()]

    async def query_traces(self, days: int = 7, model: str = "", limit: int = 50) -> dict:
        """Query recent traces from ClickHouse."""
        from ..storage.clickhouse_client import query_traces, get_client

        if get_client() is None:
            return {"error": "ClickHouse not available", "traces": []}
        start = datetime.now(timezone.utc) - timedelta(days=days)
        traces = query_traces(model=model or None, start=start, limit=limit)
        return {"traces": traces[:limit], "count": len(traces)}

    async def query_summary(self, days: int = 7) -> dict:
        """Get aggregated stats from ClickHouse."""
        from ..storage.clickhouse_client import query_summary, get_client

        if get_client() is None:
            return {"error": "ClickHouse not available"}
        start = datetime.now(timezone.utc) - timedelta(days=days)
        return query_summary(start=start)

    async def embed_texts(self, texts: list[str]) -> dict:
        """Generate MiniLM embeddings for texts."""
        from ..core.embeddings import PromptEmbedder, SentenceTransformerProvider

        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        embedder = PromptEmbedder(provider, cache_enabled=True)
        embeddings = embedder.embed_batch(texts)
        return {
            "count": len(texts),
            "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
        }

    async def cluster_texts(self, texts: list[str], k: int = 5) -> dict:
        """KMeans cluster a list of texts by semantic similarity."""
        from ..core.embeddings import PromptEmbedder, SentenceTransformerProvider
        from sklearn.cluster import KMeans

        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        embedder = PromptEmbedder(provider, cache_enabled=True)
        embeddings = embedder.embed_batch(texts)

        k = min(k, len(texts))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)

        clusters: dict[int, list[str]] = {}
        for i, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(texts[i][:100])

        return {"k": k, "clusters": {str(k): v for k, v in clusters.items()}}

    async def read_secrets(self) -> dict:
        """List configured API key providers."""
        from ..storage.secrets import list_configured_providers

        return {"configured_providers": list_configured_providers()}

    async def list_datasets(self, status: str = "") -> dict:
        """List domain datasets from clustering."""
        from ..storage.clickhouse_client import get_client

        client = get_client()
        if client is None:
            return {"datasets": []}
        query = "SELECT run_id, cluster_id, status, domain_label, trace_count FROM cluster_datasets ORDER BY trace_count DESC LIMIT 50"
        r = client.query(query)
        return {"datasets": [dict(zip(r.column_names, row)) for row in r.result_rows]}

    async def list_models(self) -> dict:
        """List available LLM models from the Go engine."""
        import httpx

        try:
            r = httpx.get(f"{self._engine_url}/v1/models", timeout=5.0)
            return r.json()
        except Exception:
            return {"models": []}

    async def add_traces(self, traces: list[dict]) -> dict:
        """Add traces to ClickHouse via the Go engine."""
        import httpx

        try:
            r = httpx.post(
                f"{self._engine_url}/v1/traces",
                json={"traces": traces},
                timeout=30.0,
            )
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    async def read_memory(
        self,
        agent: str = "",
        category: str = "",
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> dict:
        """Query memory entries for context."""
        from .memory_store import get_memory_store

        store = get_memory_store()
        entries = store.query(
            agent=agent or None,
            category=category or None,
            tags=tags,
            limit=limit,
        )
        return {
            "count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }

    async def write_memory(
        self,
        agent: str,
        content: str,
        tags: list[str] | None = None,
        category: str = "note",
    ) -> dict:
        """Save a memory note (agent-authored)."""
        from .memory_store import MemoryEntry, get_memory_store

        store = get_memory_store()
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            agent=agent,
            category=category,
            created_at=datetime.now(timezone.utc).isoformat(),
            body=content,
            tags=tags or [],
        )
        path = store.save(entry)
        return {"saved": True, "id": entry.id, "path": str(path)}

    async def read_feedback(
        self,
        issue_type: str = "",
        model: str = "",
        limit: int = 20,
    ) -> dict:
        """Query user feedback (false positive dismissals)."""
        from .memory_store import get_memory_store

        store = get_memory_store()
        tags = []
        if issue_type:
            tags.append(issue_type)
        if model:
            tags.append(model)

        entries = store.query(
            agent="trace_scanner",
            category="user_feedback",
            tags=tags or None,
            limit=limit,
        )
        return {
            "count": len(entries),
            "feedback": [
                {
                    "issue_type": e.evaluation.get("issue_type"),
                    "model_id": e.evaluation.get("model_id"),
                    "feedback_type": e.evaluation.get("feedback_type"),
                    "trace_input_preview": e.evaluation.get("trace_input_preview", "")[:200],
                    "reason": e.evaluation.get("reason", ""),
                    "created_at": e.created_at,
                }
                for e in entries
            ],
        }
