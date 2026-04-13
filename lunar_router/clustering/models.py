"""Data models for the clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


def _f(v: Any) -> float:
    """Coerce numpy/other numeric types to plain Python float."""
    return float(v) if v is not None else 0.0


@dataclass
class ClusterLabel:
    """Rich structured label produced by LLM for a cluster."""

    domain_label: str  # "Programming", "Medical Q&A", etc.
    short_description: str  # one sentence covering what the cluster is about
    inclusion_rule: str  # what types of prompts belong here
    exclusion_rule: str  # what types of prompts do NOT belong
    confidence: float  # 0-1, LLM's self-assessed confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_label": self.domain_label,
            "short_description": self.short_description,
            "inclusion_rule": self.inclusion_rule,
            "exclusion_rule": self.exclusion_rule,
            "confidence": _f(self.confidence),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterLabel:
        return cls(
            domain_label=d.get("domain_label", "Unknown"),
            short_description=d.get("short_description", ""),
            inclusion_rule=d.get("inclusion_rule", ""),
            exclusion_rule=d.get("exclusion_rule", ""),
            confidence=d.get("confidence", 0.0),
        )

    @classmethod
    def unknown(cls) -> ClusterLabel:
        return cls("Unknown", "", "", "", 0.0)


@dataclass
class TraceRow:
    """A single trace extracted from ClickHouse with full metadata."""

    request_id: str
    timestamp: str
    input_text: str
    output_text: str
    selected_model: str
    provider: str
    router_cluster_id: int  # cluster_id from routing (-1 if none)
    latency_ms: float
    ttft_ms: float
    total_cost_usd: float
    tokens_in: int
    tokens_out: int
    is_error: bool
    error_category: str
    is_stream: bool
    cache_hit: bool
    request_type: str
    expected_error: float
    cost_adjusted_score: float
    input_messages: str  # JSON
    output_message: str  # JSON


@dataclass
class CandidateDataset:
    """A cluster that may or may not be promoted to a qualified dataset."""

    cluster_id: int
    label: ClusterLabel
    trace_ids: list[str] = field(default_factory=list)
    trace_count: int = 0
    status: str = "candidate"  # "candidate" | "qualified" | "rejected"

    # Quality metrics
    coherence_score: float = 0.0  # LLM-rated 0-1
    diversity_score: float = 0.0  # embedding variance within cluster
    noise_rate: float = 0.0  # % of traces flagged as outliers
    avg_success_rate: float = 0.0  # from is_error field
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0

    # Enrichment metadata
    top_models: list[str] = field(default_factory=list)
    top_providers: list[str] = field(default_factory=list)
    has_tool_usage: bool = False
    sample_prompts: list[str] = field(default_factory=list)

    def is_qualified(self) -> bool:
        return self.status == "qualified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": int(self.cluster_id),
            "label": self.label.to_dict(),
            "trace_count": int(self.trace_count),
            "status": self.status,
            "coherence_score": _f(self.coherence_score),
            "diversity_score": _f(self.diversity_score),
            "noise_rate": _f(self.noise_rate),
            "avg_success_rate": _f(self.avg_success_rate),
            "avg_latency_ms": _f(self.avg_latency_ms),
            "avg_cost_usd": _f(self.avg_cost_usd),
            "top_models": self.top_models,
            "top_providers": self.top_providers,
            "has_tool_usage": bool(self.has_tool_usage),
            "sample_prompts": self.sample_prompts,
        }


@dataclass
class DatasetVersion:
    """Full provenance for a clustering run."""

    version: str  # "v1", "v2", ...
    run_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    source_window_start: Optional[datetime] = None
    source_window_end: Optional[datetime] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    clustering_config: dict[str, Any] = field(default_factory=dict)
    labeler_model: str = "mistral-small-latest"
    trace_count: int = 0
    num_clusters: int = 0
    silhouette_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "source_window_start": self.source_window_start.isoformat() if self.source_window_start else None,
            "source_window_end": self.source_window_end.isoformat() if self.source_window_end else None,
            "embedding_model": self.embedding_model,
            "embedding_dim": int(self.embedding_dim),
            "clustering_config": self.clustering_config,
            "labeler_model": self.labeler_model,
            "trace_count": int(self.trace_count),
            "num_clusters": int(self.num_clusters),
            "silhouette_score": _f(self.silhouette_score),
        }


@dataclass
class MergeSuggestion:
    """Suggestion to merge two clusters — requires human confirmation."""

    cluster_a: int
    cluster_b: int
    similarity_score: float  # cosine similarity of centroids
    llm_agrees: bool  # LLM thinks they're the same domain
    reason: str  # why they might merge

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_a": int(self.cluster_a),
            "cluster_b": int(self.cluster_b),
            "similarity_score": _f(self.similarity_score),
            "llm_agrees": bool(self.llm_agrees),
            "reason": self.reason,
        }


@dataclass
class ClusteringResult:
    """Full result of a clustering pipeline run."""

    version: DatasetVersion
    datasets: list[CandidateDataset] = field(default_factory=list)
    merge_suggestions: list[MergeSuggestion] = field(default_factory=list)

    @property
    def qualified_count(self) -> int:
        return sum(1 for d in self.datasets if d.status == "qualified")

    @property
    def candidate_count(self) -> int:
        return sum(1 for d in self.datasets if d.status == "candidate")

    @property
    def rejected_count(self) -> int:
        return sum(1 for d in self.datasets if d.status == "rejected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version.to_dict(),
            "datasets": [d.to_dict() for d in self.datasets],
            "merge_suggestions": [m.to_dict() for m in self.merge_suggestions],
            "summary": {
                "total": len(self.datasets),
                "qualified": self.qualified_count,
                "candidate": self.candidate_count,
                "rejected": self.rejected_count,
            },
        }
