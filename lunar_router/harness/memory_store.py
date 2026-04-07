"""MemoryStore — file-backed persistent memory for harness agent runs.

Memory entries are .md files with YAML frontmatter (structured metadata)
and a markdown body (human-readable context). Mirrors the AgentRegistry
pattern: directory-backed, scan on init, singleton accessor.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path(__file__).parent / "memory"

# Known evaluation-relevant keys to extract from agent result data
_EVAL_KEYS = {"confidence", "coherence", "same_domain", "success", "quality_assessment"}

# Singleton instance
_instance: Optional[MemoryStore] = None


@dataclass
class MemoryEntry:
    """A single memory record parsed from a .md file."""

    id: str
    agent: str
    category: str  # "run_result" | "tool_stats" | "evaluation" | "decision" | "note"
    created_at: str
    body: str = ""
    model: str = ""
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    retried: bool = False
    tags: list[str] = field(default_factory=list)
    context_hash: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "category": self.category,
            "created_at": self.created_at,
            "body": self.body,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_calls": self.tool_calls,
            "retried": self.retried,
            "tags": self.tags,
            "context_hash": self.context_hash,
            "evaluation": self.evaluation,
        }

    def to_markdown(self) -> str:
        """Serialize to .md file content (YAML frontmatter + body)."""
        meta = {
            "id": self.id,
            "agent": self.agent,
            "category": self.category,
            "created_at": self.created_at,
            "model": self.model,
            "duration_ms": round(self.duration_ms, 1),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_calls": self.tool_calls,
            "retried": self.retried,
            "tags": self.tags,
            "context_hash": self.context_hash,
            "evaluation": self.evaluation,
        }
        # Drop empty optional fields
        meta = {k: v for k, v in meta.items() if v or isinstance(v, (bool, int, float))}
        frontmatter = yaml.dump(meta, default_flow_style=False, sort_keys=False).strip()
        return f"---\n{frontmatter}\n---\n\n{self.body}\n"

    @classmethod
    def from_agent_result(
        cls,
        result: Any,
        user_input: str,
        category: str = "run_result",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Create a MemoryEntry from an AgentResult."""
        context_hash = hashlib.sha256(user_input.encode()).hexdigest()[:12]

        # Extract evaluation-relevant fields from result.data
        evaluation: dict[str, Any] = {}
        data = result.data if hasattr(result, "data") else {}
        for key in _EVAL_KEYS:
            if key in data:
                evaluation[key] = data[key]
        if "parse_error" not in data:
            evaluation.setdefault("success", True)

        # Build human-readable body
        body_parts = ["## Run Summary\n"]
        input_preview = user_input[:300]
        if len(user_input) > 300:
            input_preview += "..."
        body_parts.append(f"**Input preview:** {input_preview}\n")

        body_parts.append("## Output\n")
        for key, value in data.items():
            if key.startswith("_"):
                continue
            body_parts.append(f"- **{key}:** {value}")

        body = "\n".join(body_parts)

        return cls(
            id=str(uuid.uuid4()),
            agent=result.agent,
            category=category,
            created_at=datetime.now(timezone.utc).isoformat(),
            body=body,
            model=result.model,
            duration_ms=result.duration_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            tool_calls=result.tool_calls,
            retried=result.retried,
            tags=tags or [],
            context_hash=context_hash,
            evaluation=evaluation,
        )


def _parse_memory_file(path: Path) -> MemoryEntry:
    """Parse a .md file with YAML frontmatter into MemoryEntry."""
    text = path.read_text()

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter_str = parts[1].strip()
            body = parts[2].strip()
        else:
            frontmatter_str = ""
            body = text
    else:
        frontmatter_str = ""
        body = text

    meta = yaml.safe_load(frontmatter_str) if frontmatter_str else {}
    if not isinstance(meta, dict):
        meta = {}

    return MemoryEntry(
        id=meta.get("id", path.stem),
        agent=meta.get("agent", ""),
        category=meta.get("category", "note"),
        created_at=meta.get("created_at", ""),
        body=body,
        model=meta.get("model", ""),
        duration_ms=float(meta.get("duration_ms", 0)),
        tokens_in=int(meta.get("tokens_in", 0)),
        tokens_out=int(meta.get("tokens_out", 0)),
        tool_calls=int(meta.get("tool_calls", 0)),
        retried=bool(meta.get("retried", False)),
        tags=meta.get("tags", []) or [],
        context_hash=meta.get("context_hash", ""),
        evaluation=meta.get("evaluation", {}) or {},
        file_path=str(path),
    )


def _safe_mean(values: list[float | int]) -> float:
    """Mean that returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


class MemoryStore:
    """File-backed CRUD + query store for memory entries."""

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, MemoryEntry] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan memory directory and load all .md files."""
        self._entries.clear()
        for path in self.memory_dir.glob("*.md"):
            try:
                entry = _parse_memory_file(path)
                self._entries[entry.id] = entry
            except Exception as e:
                logger.warning(f"Failed to load memory {path.name}: {e}")
        logger.info(f"Loaded {len(self._entries)} memory entries from {self.memory_dir}")

    # --- CRUD ---

    def save(self, entry: MemoryEntry) -> Path:
        """Write a memory entry to disk and cache it."""
        filename = self._generate_filename(entry)
        path = self.memory_dir / filename
        path.write_text(entry.to_markdown())
        entry.file_path = str(path)
        self._entries[entry.id] = entry
        return path

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry from disk and cache."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        if entry.file_path:
            path = Path(entry.file_path)
            if path.exists():
                path.unlink()
        del self._entries[entry_id]
        return True

    def list_all(self) -> list[MemoryEntry]:
        """Return all entries sorted by created_at descending."""
        return sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )

    # --- Querying ---

    def query(
        self,
        agent: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        since: Optional[str] = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Filter and return memory entries."""
        results = list(self._entries.values())

        if agent:
            results = [e for e in results if e.agent == agent]
        if category:
            results = [e for e in results if e.category == category]
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set & set(e.tags)]
        if since:
            results = [e for e in results if e.created_at >= since]

        results.sort(key=lambda e: e.created_at, reverse=True)
        return results[:limit]

    def agent_summary(self, agent_name: str) -> dict[str, Any]:
        """Aggregate stats for an agent across all run_result entries."""
        entries = self.query(agent=agent_name, category="run_result", limit=10000)
        if not entries:
            return {"agent": agent_name, "runs": 0}

        durations = [e.duration_ms for e in entries]
        tokens_in = [e.tokens_in for e in entries]
        tokens_out = [e.tokens_out for e in entries]
        tool_counts = [e.tool_calls for e in entries]
        success_count = sum(1 for e in entries if e.evaluation.get("success"))
        retry_count = sum(1 for e in entries if e.retried)
        all_tags = [tag for e in entries for tag in e.tags]

        return {
            "agent": agent_name,
            "runs": len(entries),
            "avg_duration_ms": round(_safe_mean(durations), 1),
            "avg_tokens_in": round(_safe_mean(tokens_in)),
            "avg_tokens_out": round(_safe_mean(tokens_out)),
            "success_rate": round(success_count / len(entries), 3),
            "retry_rate": round(retry_count / len(entries), 3),
            "tool_call_avg": round(_safe_mean(tool_counts), 1),
            "common_tags": Counter(all_tags).most_common(10),
            "last_run": max(e.created_at for e in entries),
        }

    def tool_effectiveness(self, agent_name: str) -> dict[str, Any]:
        """Compare runs with/without tools for an agent."""
        entries = self.query(agent=agent_name, category="run_result", limit=10000)
        with_tools = [e for e in entries if e.tool_calls > 0]
        without_tools = [e for e in entries if e.tool_calls == 0]

        def _stats(group: list[MemoryEntry]) -> dict[str, Any]:
            if not group:
                return {"count": 0, "avg_duration_ms": 0, "success_rate": 0}
            success = sum(1 for e in group if e.evaluation.get("success"))
            return {
                "count": len(group),
                "avg_duration_ms": round(_safe_mean([e.duration_ms for e in group]), 1),
                "success_rate": round(success / len(group), 3),
            }

        return {
            "agent": agent_name,
            "with_tools": _stats(with_tools),
            "without_tools": _stats(without_tools),
        }

    def prune(self, max_entries: int = 500) -> int:
        """Delete oldest entries beyond max_entries threshold. Returns count deleted."""
        if len(self._entries) <= max_entries:
            return 0

        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        to_delete = sorted_entries[max_entries:]
        for entry in to_delete:
            self.delete(entry.id)

        return len(to_delete)

    # --- Internal ---

    def _generate_filename(self, entry: MemoryEntry) -> str:
        """Generate a filename like cluster_labeler_20260330T153022_a1b2c3.md."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        short_id = entry.id[:8]
        agent = entry.agent.replace(" ", "_") if entry.agent else "unknown"
        return f"{agent}_{ts}_{short_id}.md"


def get_memory_store(memory_dir: Optional[Path] = None) -> MemoryStore:
    """Get or create the singleton MemoryStore."""
    global _instance
    if _instance is None:
        _instance = MemoryStore(memory_dir)
    return _instance
