"""Router proposer — the auto_trainer pipeline disassembled into AHE shape.

The reference's ``training/auto_trainer.py`` is a monolith that fits +
recomputes Psi + promotes in one pass. AHE wants every change to travel
through a uniform pipeline: proposer → critic → approver → executor →
ledger. This module owns the **proposer half**: pull traces, augment,
fit clusters, recompute Psi, package an inline candidate payload, and
emit a ``Proposal``.

The candidate payload rides on ``Mutation.value`` (a dict). The critic
in ``harness/critics/router_critic.py`` reads it back and scores via
``RouterEvaluator``. The executor in ``harness/executor/promote.py``
reads it again to write the on-disk artifact and flip the current
pointer.

This module does **not** call the LLM judge or run model generation
itself. The proposer accepts an already-prepared
``preference_dataset`` (typically built by P15.3.5's GoldenAugmenter
ahead of time, persisted to ``evals/preference_pairs/``). When the
preference dataset is empty/None the proposer still produces a valid
candidate — just without the preference-refinement signal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from experiments.types import Mutation
from harness.proposer.router_psi_compute import compute_blended_psi
from harness.types import Prediction, Proposal
from router.config_io import get_current_version, now_iso
from router.core.embeddings import PromptEmbedder
from router.errors import NotEnoughDataError
from router.feedback.store_adapter import iter_traces_since
from router.feedback.trace_to_training import TraceRecord
from router.models.llm_registry import LLMRegistry
from router.training.gate import DEFAULT_MIN_CORPUS_SIZE, check_first_fit_eligibility
from router.training.kmeans import KMeansTrainer
from router.training.result import KMeansTrainResult


logger = logging.getLogger("harness.proposer.router_proposer")


DEFAULT_PRODUCTION_ALPHA = 0.3   # locked by P15.3 roadmap
DEFAULT_PROPOSAL_SOURCE = "claude_code"


@dataclass
class RouterProposerConfig:
    """Knobs for one ``RouterProposer.propose()`` invocation."""

    min_corpus_size: int = DEFAULT_MIN_CORPUS_SIZE
    target_k: Optional[int] = None       # None → sqrt(N/2) heuristic
    max_augmentation_samples: int = 500  # reserved for future use
    production_alpha: float = DEFAULT_PRODUCTION_ALPHA
    proposal_source: str = DEFAULT_PROPOSAL_SOURCE


@dataclass
class _ProposeMaterials:
    """Internal — what we accumulated to build the Proposal."""

    fit_result: KMeansTrainResult
    traces: list[TraceRecord]
    psi_table: dict[str, list[float]]


class RouterProposer:
    """Builds a router_config candidate Proposal.

    Single-shot — the harness/MCP layer (P15.3.9) decides when to call
    ``propose()``. This class doesn't run any loop.

    Construction takes pre-built dependencies (embedder, registry, cache,
    optional preference dataset) so the harness wiring layer can hand-pick
    them. ``propose()`` does the rest:

      1. Stream traces since the current config's ``created_at``.
      2. Apply the first-fit gate (raises ``NotEnoughDataError`` if too few).
      3. Pick K (heuristic or constructor-supplied).
      4. Fit clusters.
      5. Re-embed + re-assign each trace against the new clusters.
      6. Compute blended Psi.
      7. Build the candidate payload (centroids + Psi + drift_baseline).
      8. Return a ``Proposal`` ready for the critic.
    """

    def __init__(
        self,
        *,
        embedder: PromptEmbedder,
        registry: LLMRegistry,
        cache: Any = None,           # ResponseCache, reserved for future Psi math
        preference_dataset: Any = None,
        cfg: Optional[RouterProposerConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.registry = registry
        self.cache = cache
        self.preference_dataset = preference_dataset
        self.cfg = cfg or RouterProposerConfig()

    # ------------------------------------------------------------------

    def propose(
        self,
        *,
        since_iso: Optional[str] = None,
    ) -> Proposal:
        """Build one candidate. Raises ``NotEnoughDataError`` when the
        trace corpus is below ``min_corpus_size`` (the harness brain in
        P15.3.9 is expected to handle this and skip).
        """
        materials = self._gather(since_iso=since_iso)
        candidate_payload = self._build_payload(materials)

        next_version = candidate_payload["version"]
        target_path = f"versions/router_config_v{next_version}.json"

        proposal = Proposal(
            mutations=[
                Mutation(
                    file=target_path,
                    path="<inline_payload>",
                    value=candidate_payload,
                )
            ],
            description=(
                f"UniRoute router_config v{next_version}: "
                f"K={materials.fit_result.k}, N={materials.fit_result.n_samples}, "
                f"silhouette={materials.fit_result.silhouette:.3f}"
            ),
            source=self.cfg.proposal_source,
            metadata={
                "candidate_payload_inline": True,
                "fitted_from": dict(materials.fit_result.fitted_from),
                "n_traces": len(materials.traces),
                "production_alpha": self.cfg.production_alpha,
                "preference_pairs": _len_or_zero(self.preference_dataset),
            },
            prediction=Prediction(
                rubric="overall",
                expected_delta=max(0.0, materials.fit_result.silhouette - 0.0),
                rationale=(
                    f"Fit K={materials.fit_result.k} clusters on "
                    f"{materials.fit_result.n_samples} production traces "
                    f"(silhouette={materials.fit_result.silhouette:.3f}). "
                    "Expecting AUROC lift over the current config's baseline."
                ),
                confidence=0.4,
            ),
        )
        logger.info(
            "proposed router_config v%d: K=%d N=%d silhouette=%.3f",
            next_version,
            materials.fit_result.k,
            materials.fit_result.n_samples,
            materials.fit_result.silhouette,
        )
        return proposal

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gather(self, *, since_iso: Optional[str]) -> _ProposeMaterials:
        # 1. Pull cold-start traces (cluster_id=-1 sentinel — used for fitting).
        traces_for_fit: list[TraceRecord] = list(
            iter_traces_since(
                since_iso=since_iso,
                embedder=None,
                assigner=None,
            )
        )

        # 2. First-fit gate.
        eligible, reason = check_first_fit_eligibility(
            corpus_size=len(traces_for_fit),
            min_corpus_size=self.cfg.min_corpus_size,
            requested_k=self.cfg.target_k,
        )
        if not eligible:
            raise NotEnoughDataError(reason)

        # 3. Pick K — caller override or sqrt(N/2) heuristic, clamped [4, 32].
        k = self.cfg.target_k or _sqrt_k_heuristic(len(traces_for_fit))

        # 4. Fit clusters.
        prompts = [t.input_text for t in traces_for_fit if t.input_text]
        trainer = KMeansTrainer(self.embedder, num_clusters=k)
        fitted_from = {
            "source": "production_traces",
            "n_traces": len(traces_for_fit),
            "earliest": _safe_metadata_iso(traces_for_fit[0]) if traces_for_fit else None,
            "latest": _safe_metadata_iso(traces_for_fit[-1]) if traces_for_fit else None,
            "since_iso": since_iso,
        }
        fit_result = trainer.train(
            prompts,
            fitted_from=fitted_from,
            min_corpus_size=self.cfg.min_corpus_size,
        )

        # 5. Re-assign each trace against the new clusters so Psi math has
        # cluster_id ∈ [0, K) instead of the cold-start sentinel.
        traces_with_clusters: list[TraceRecord] = []
        for t in traces_for_fit:
            if not t.input_text:
                continue
            emb = self.embedder.embed(t.input_text)
            cid = int(fit_result.assigner.assign(emb).cluster_id)
            traces_with_clusters.append(_with_cluster(t, cid))

        # 6. Blend Psi.
        psi_table = compute_blended_psi(
            assigner=fit_result.assigner,
            registry=self.registry,
            traces=traces_with_clusters,
            preference_dataset=self.preference_dataset,
            cache=self.cache,
            production_alpha=self.cfg.production_alpha,
        )

        return _ProposeMaterials(
            fit_result=fit_result,
            traces=traces_with_clusters,
            psi_table=psi_table,
        )

    def _build_payload(self, m: _ProposeMaterials) -> dict:
        next_version = (get_current_version() or 0) + 1
        intra_dist = _intra_cluster_distance(m)
        embedder_model = m.fit_result.embedder_model_id
        embedding_dim = int(m.fit_result.assigner.embedding_dim)

        # Per-model Psi as a dict; centroids go in the sidecar .npz the
        # executor writes (atomic). Keep "centroids" as None inline.
        model_psi: dict[str, dict] = {}
        for profile in self.registry.get_all():
            psi_vec = m.psi_table.get(profile.model_id)
            if psi_vec is None:
                continue
            model_psi[profile.model_id] = {
                "psi_vector": list(psi_vec),
                "cost_per_1k_tokens": float(profile.cost_per_1k_tokens),
                "cluster_sample_counts": list(
                    map(int, profile.cluster_sample_counts.tolist())
                )
                if profile.cluster_sample_counts.size == m.fit_result.k
                else [int(profile.num_validation_samples / max(m.fit_result.k, 1))]
                * m.fit_result.k,
                "metadata": dict(profile.metadata),
            }

        return {
            "version": next_version,
            "k": int(m.fit_result.k),
            "centroids": None,             # written to sidecar .npz by the executor
            "model_psi": model_psi,
            "cost_weight": 0.0,
            "embedder_model": embedder_model,
            "embedding_dim": embedding_dim,
            "min_corpus_size": int(self.cfg.min_corpus_size),
            "created_at": m.fit_result.fitted_at,
            "fitted_from": dict(m.fit_result.fitted_from),
            "drift_baseline": float(intra_dist) if intra_dist is not None else None,
            "metadata": {
                "phase": "P15.3.7",
                "stage": "proposer_candidate",
                "silhouette": float(m.fit_result.silhouette),
                "inertia": float(m.fit_result.inertia),
                "n_samples": int(m.fit_result.n_samples),
                "production_alpha": float(self.cfg.production_alpha),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sqrt_k_heuristic(n: int) -> int:
    """K ≈ sqrt(N/2), clamped to [4, 32]."""
    raw = int(math.sqrt(max(n, 0) / 2))
    return max(4, min(raw, 32)) if raw else 4


def _intra_cluster_distance(m: _ProposeMaterials) -> Optional[float]:
    """Mean nearest-centroid distance over the fit corpus.

    P15.3.9 uses this as the persisted ``drift_baseline`` so DriftDetector
    doesn't self-baseline on its first arbitrary embedding batch. The
    fix flagged in P15.3.4 lives here.
    """
    if not m.traces:
        return None
    distances = []
    for t in m.traces:
        if not t.input_text:
            continue
        # We can't recover the original embedding here without re-embedding,
        # which we already did above. The proposer keeps the trade-off
        # cheap: cluster_id is set, recompute once with the cached embedder.
        # The math: distance between the trace's embedding and its assigned
        # centroid. We approximate via the assigner's centroids stored in
        # the fit_result.
        # (Re-embedding is fine here — embedder is cached process-wide.)
        # Avoid a circular import; deferred to _intra_cluster_for_traces.
        return _intra_cluster_for_traces(m)
    return None


def _intra_cluster_for_traces(m: _ProposeMaterials) -> float:
    import numpy as np

    centroids = m.fit_result.assigner.centroids
    embedder = None  # not stored on materials; the proposer's _gather already used it

    # We re-embed lazily here. To avoid a runtime dep, use the proposer's
    # KMeansTrainResult.assigner internal centroids and approximate the
    # mean distance via the inertia: inertia = sum of squared distances,
    # so sqrt(inertia / N) is the RMS distance. This avoids re-running
    # the embedder.
    inertia = float(m.fit_result.inertia)
    n = max(int(m.fit_result.n_samples), 1)
    return float(np.sqrt(inertia / n))


def _with_cluster(trace: TraceRecord, cluster_id: int) -> TraceRecord:
    """Return a copy of ``trace`` with ``cluster_id`` set."""
    return TraceRecord(
        request_id=trace.request_id,
        selected_model=trace.selected_model,
        cluster_id=cluster_id,
        is_error=trace.is_error,
        latency_ms=trace.latency_ms,
        total_cost_usd=trace.total_cost_usd,
        input_text=trace.input_text,
        output_text=trace.output_text,
        error_category=trace.error_category,
        metadata=dict(trace.metadata),
    )


def _safe_metadata_iso(trace: TraceRecord) -> Optional[str]:
    ts = trace.metadata.get("timestamp")
    return str(ts) if ts else None


def _len_or_zero(obj: Any) -> int:
    try:
        return len(obj) if obj is not None else 0
    except TypeError:
        return 0
