"""Clustering pipeline: traces → domain datasets.

Extracts traces from ClickHouse, embeds them, clusters with adaptive KMeans,
labels with LLM, applies quality gates, and stores results.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from .labeler import ClusterLabeler
from .models import (
    CandidateDataset,
    ClusteringResult,
    DatasetVersion,
    MergeSuggestion,
    TraceRow,
)
from .quality import (
    QualityThresholds,
    apply_quality_gates,
    compute_diversity_score,
    compute_noise_rate,
)

logger = logging.getLogger(__name__)


def _adaptive_k_candidates(n_traces: int) -> list[int]:
    """Select K candidates based on data volume."""
    if n_traces < 100:
        return [5]
    elif n_traces < 500:
        return [5, 10, 15]
    elif n_traces < 5000:
        return [10, 20, 50]
    else:
        return [20, 50, 100]


class ClusteringPipeline:
    """Main pipeline: extract → embed → cluster → label → qualify → store."""

    def __init__(
        self,
        strategy: str = "auto",
        llm_provider: str = "mistral",
        llm_model: str = "mistral-small-latest",
        engine_url: str = "http://localhost:8080",
        thresholds: Optional[QualityThresholds] = None,
    ):
        """
        Args:
            strategy: "auto" (adaptive KMeans) or "router-aligned" (reuse router cluster_ids)
            llm_provider: provider name for labeling (key must exist in secrets)
            llm_model: specific model for labeling
            engine_url: Go engine URL for LLM calls
            thresholds: quality gate thresholds (uses defaults if None)
        """
        self.strategy = strategy
        self.llm_model = llm_model
        self.engine_url = engine_url
        self.thresholds = thresholds or QualityThresholds()
        self.labeler = ClusterLabeler(model=llm_model, engine_url=engine_url)

    async def run(
        self,
        days: int = 30,
        min_traces: int = 50,
    ) -> ClusteringResult:
        """Execute the full clustering pipeline."""
        run_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=days)

        logger.info(f"[clustering:{run_id}] Starting pipeline (strategy={self.strategy}, days={days})")

        # Step 1: Extract traces from ClickHouse
        traces = self._extract_traces(window_start, now)
        logger.info(f"[clustering:{run_id}] Extracted {len(traces)} traces with content")

        if len(traces) < min_traces:
            logger.warning(f"[clustering:{run_id}] Only {len(traces)} traces, need {min_traces}. Aborting.")
            return ClusteringResult(
                version=DatasetVersion(
                    version="v0", run_id=run_id, trace_count=len(traces),
                    source_window_start=window_start, source_window_end=now,
                ),
            )

        # Step 2: Route-aligned or auto clustering
        if self.strategy == "router-aligned":
            cluster_assignments, num_clusters, silhouette, embeddings = self._cluster_router_aligned(traces)
        else:
            cluster_assignments, num_clusters, silhouette, embeddings = self._cluster_auto(traces)

        logger.info(f"[clustering:{run_id}] Clustered into {num_clusters} clusters (silhouette={silhouette:.3f})")

        # Step 3: Group traces by cluster
        cluster_groups: dict[int, list[int]] = {}  # cluster_id → trace indices
        for i, cid in enumerate(cluster_assignments):
            cluster_groups.setdefault(cid, []).append(i)

        # Step 4: Build candidate datasets + label + quality score
        datasets: list[CandidateDataset] = []
        for cid_raw, indices in sorted(cluster_groups.items()):
            cid = int(cid_raw)  # ensure native int, not numpy
            cluster_traces = [traces[i] for i in indices]
            cluster_embeddings = embeddings[indices] if embeddings is not None else None
            prompts = [t.input_text for t in cluster_traces]

            # Sample prompts for LLM
            sample_size = min(10, len(prompts))
            rng = np.random.default_rng(cid)
            sample_idx = rng.choice(len(prompts), size=sample_size, replace=False)
            sample_prompts = [prompts[i] for i in sample_idx]

            # LLM labeling
            label = await self.labeler.label_cluster(sample_prompts)

            # LLM coherence
            coherence = await self.labeler.score_coherence(sample_prompts)

            # LLM outlier detection
            outlier_idx = await self.labeler.detect_outliers(sample_prompts)
            noise_rate = compute_noise_rate(outlier_idx, len(prompts))

            # Diversity from embeddings
            diversity = 0.0
            if cluster_embeddings is not None and len(cluster_embeddings) >= 2:
                diversity = compute_diversity_score(cluster_embeddings)

            # Trace-level stats
            success_count = sum(1 for t in cluster_traces if not t.is_error)
            avg_success = success_count / len(cluster_traces) if cluster_traces else 0
            avg_latency = sum(t.latency_ms for t in cluster_traces) / len(cluster_traces) if cluster_traces else 0
            avg_cost = sum(t.total_cost_usd for t in cluster_traces) / len(cluster_traces) if cluster_traces else 0

            # Top models/providers
            model_counts = Counter(t.selected_model for t in cluster_traces)
            provider_counts = Counter(t.provider for t in cluster_traces)

            candidate = CandidateDataset(
                cluster_id=cid,
                label=label,
                trace_ids=[t.request_id for t in cluster_traces],
                trace_count=len(cluster_traces),
                coherence_score=coherence,
                diversity_score=diversity,
                noise_rate=noise_rate,
                avg_success_rate=avg_success,
                avg_latency_ms=avg_latency,
                avg_cost_usd=avg_cost,
                top_models=[m for m, _ in model_counts.most_common(3)],
                top_providers=[p for p, _ in provider_counts.most_common(3)],
                sample_prompts=sample_prompts,
            )

            # Apply quality gates
            candidate.status = apply_quality_gates(candidate, self.thresholds)
            datasets.append(candidate)

        logger.info(
            f"[clustering:{run_id}] Results: "
            f"{sum(1 for d in datasets if d.status == 'qualified')} qualified, "
            f"{sum(1 for d in datasets if d.status == 'candidate')} candidate, "
            f"{sum(1 for d in datasets if d.status == 'rejected')} rejected"
        )

        # Step 5: Generate merge suggestions (conservative)
        merge_suggestions = await self._find_merge_suggestions(datasets, embeddings, cluster_assignments)

        # Step 6: Build version info
        # Convert numpy scalars to native Python types for JSON serialization
        silhouette_py = float(silhouette)
        num_clusters_py = int(num_clusters)

        config = {
            "strategy": self.strategy,
            "k": num_clusters_py,
            "silhouette": round(silhouette_py, 4),
            "min_traces": min_traces,
            "days": days,
        }

        version = DatasetVersion(
            version="v1",
            run_id=run_id,
            source_window_start=window_start,
            source_window_end=now,
            clustering_config=config,
            labeler_model=self.llm_model,
            trace_count=len(traces),
            num_clusters=num_clusters_py,
            silhouette_score=silhouette_py,
        )

        result = ClusteringResult(
            version=version,
            datasets=datasets,
            merge_suggestions=merge_suggestions,
        )

        # Step 7: Store to ClickHouse
        self._store_results(result, traces, cluster_assignments)

        logger.info(f"[clustering:{run_id}] Pipeline complete")
        return result

    def _extract_traces(self, start: datetime, end: datetime) -> list[TraceRow]:
        """Extract traces with content from ClickHouse."""
        from ..storage.clickhouse_client import get_client

        client = get_client()
        if client is None:
            return []

        sql = """
            SELECT
                request_id, toString(timestamp), input_text, output_text,
                selected_model, provider, cluster_id,
                latency_ms, ttft_ms, total_cost_usd,
                tokens_in, tokens_out, is_error, error_category,
                is_stream, cache_hit, request_type,
                expected_error, cost_adjusted_score,
                input_messages, output_message
            FROM llm_traces
            WHERE input_text != '' AND timestamp >= {start:DateTime64(3)} AND timestamp <= {end:DateTime64(3)}
            ORDER BY timestamp DESC
        """
        result = client.query(sql, parameters={"start": start, "end": end})

        traces = []
        for row in result.result_rows:
            traces.append(TraceRow(
                request_id=str(row[0]),
                timestamp=str(row[1]),
                input_text=row[2] or "",
                output_text=row[3] or "",
                selected_model=row[4] or "",
                provider=row[5] or "",
                router_cluster_id=int(row[6]) if row[6] is not None else -1,
                latency_ms=float(row[7] or 0),
                ttft_ms=float(row[8] or 0),
                total_cost_usd=float(row[9] or 0),
                tokens_in=int(row[10] or 0),
                tokens_out=int(row[11] or 0),
                is_error=bool(row[12]),
                error_category=row[13] or "",
                is_stream=bool(row[14]),
                cache_hit=bool(row[15]),
                request_type=row[16] or "",
                expected_error=float(row[17] or 0),
                cost_adjusted_score=float(row[18] or 0),
                input_messages=row[19] or "",
                output_message=row[20] or "",
            ))

        return traces

    def _cluster_auto(self, traces: list[TraceRow]) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Embed prompts and cluster with adaptive KMeans."""
        from ..core.embeddings import PromptEmbedder, SentenceTransformerProvider
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Embed all prompts
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        embedder = PromptEmbedder(provider, cache_enabled=True)

        prompts = [t.input_text for t in traces]
        embeddings = embedder.embed_batch(prompts)

        # Adaptive K selection
        k_candidates = _adaptive_k_candidates(len(traces))
        best_k, best_score, best_labels = k_candidates[0], -1.0, None

        for k in k_candidates:
            if k >= len(traces):
                continue
            km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
            logger.info(f"  K={k}: silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        if best_labels is None:
            # Fallback: single cluster
            best_labels = np.zeros(len(traces), dtype=int)
            best_k, best_score = 1, 0.0

        return best_labels, best_k, best_score, embeddings

    def _cluster_router_aligned(self, traces: list[TraceRow]) -> tuple[np.ndarray, int, float, Optional[np.ndarray]]:
        """Reuse existing router cluster_ids — no re-clustering needed."""
        labels = np.array([t.router_cluster_id for t in traces])

        # Filter out traces without cluster assignments
        valid = labels >= 0
        if not valid.any():
            logger.warning("No traces have router cluster_ids, falling back to auto")
            return self._cluster_auto(traces)

        num_clusters = len(set(labels[valid]))
        return labels, num_clusters, 0.0, None  # no embeddings in this mode

    async def _find_merge_suggestions(
        self,
        datasets: list[CandidateDataset],
        embeddings: Optional[np.ndarray],
        assignments: np.ndarray,
    ) -> list[MergeSuggestion]:
        """Find pairs of clusters that might be the same domain."""
        if embeddings is None or len(datasets) < 2:
            return []

        # Compute centroids
        centroids = {}
        for ds in datasets:
            mask = assignments == ds.cluster_id
            if mask.any():
                centroids[ds.cluster_id] = embeddings[mask].mean(axis=0)

        suggestions = []
        checked = set()

        for ds_a in datasets:
            if ds_a.cluster_id not in centroids:
                continue
            for ds_b in datasets:
                if ds_b.cluster_id not in centroids:
                    continue
                pair = tuple(sorted([ds_a.cluster_id, ds_b.cluster_id]))
                if pair in checked or ds_a.cluster_id == ds_b.cluster_id:
                    continue
                checked.add(pair)

                # Cosine similarity
                a, b = centroids[ds_a.cluster_id], centroids[ds_b.cluster_id]
                sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

                # Only suggest if very similar (conservative)
                if sim > 0.85:
                    llm_agrees, reason = await self.labeler.check_merge(
                        ds_a.sample_prompts[:5], ds_b.sample_prompts[:5]
                    )
                    suggestions.append(MergeSuggestion(
                        cluster_a=ds_a.cluster_id,
                        cluster_b=ds_b.cluster_id,
                        similarity_score=round(sim, 4),
                        llm_agrees=llm_agrees,
                        reason=reason,
                    ))

        return suggestions

    def _store_results(
        self,
        result: ClusteringResult,
        traces: list[TraceRow],
        assignments: np.ndarray,
    ) -> None:
        """Persist clustering results to ClickHouse."""
        import json
        from ..storage.clickhouse_client import get_client

        client = get_client()
        if client is None:
            logger.warning("ClickHouse not available, skipping result storage")
            return

        v = result.version

        # Insert clustering run
        run_columns = [
            "run_id", "created_at", "strategy", "num_clusters", "silhouette_score",
            "source_window_start", "source_window_end", "total_traces",
            "embedding_model", "labeler_model", "config",
        ]
        client.insert("clustering_runs", [[
            v.run_id,
            v.created_at,
            v.clustering_config.get("strategy", "auto"),
            v.num_clusters,
            v.silhouette_score,
            v.source_window_start,
            v.source_window_end,
            v.trace_count,
            v.embedding_model,
            v.labeler_model,
            json.dumps(v.clustering_config),
        ]], column_names=run_columns)

        # Insert cluster datasets
        ds_columns = [
            "run_id", "cluster_id", "status", "domain_label", "short_description",
            "inclusion_rule", "exclusion_rule", "label_confidence", "trace_count",
            "coherence_score", "diversity_score", "noise_rate", "avg_success_rate",
            "avg_latency_ms", "avg_cost_usd", "top_models", "top_providers",
            "sample_prompts", "version",
        ]
        rows = []
        for ds in result.datasets:
            rows.append([
                v.run_id,
                ds.cluster_id,
                ds.status,
                ds.label.domain_label,
                ds.label.short_description,
                ds.label.inclusion_rule,
                ds.label.exclusion_rule,
                ds.label.confidence,
                ds.trace_count,
                ds.coherence_score,
                ds.diversity_score,
                ds.noise_rate,
                ds.avg_success_rate,
                ds.avg_latency_ms,
                ds.avg_cost_usd,
                json.dumps(ds.top_models),
                json.dumps(ds.top_providers),
                json.dumps(ds.sample_prompts),
                v.version,
            ])

        if rows:
            client.insert("cluster_datasets", rows, column_names=ds_columns)

        # Batch insert trace → cluster mapping
        map_rows = []
        for i, trace in enumerate(traces):
            cid = int(assignments[i])
            map_rows.append([v.run_id, trace.request_id, cid, trace.output_text])

        if map_rows:
            client.insert(
                "trace_cluster_map",
                map_rows,
                column_names=["run_id", "request_id", "cluster_id", "output_text"],
            )

        logger.info(f"[clustering:{v.run_id}] Stored {len(result.datasets)} clusters, {len(map_rows)} mappings")
