"""Cluster-only snapshot writer.

Writes a partial ``router_config_<n>.json`` artifact carrying the fitted
centroids but with empty ``model_psi`` and **without** updating the
``current`` pointer. Only the executor in P15.3.7 promotes a fully-stitched
config (centroids + Psi + cost_weight) to current.

Why partial vs full? The reference's ``auto_trainer.py`` is a monolith that
refits + recomputes Psi + promotes in one pass. We split it: P15.3.3 lands
the cluster fit, P15.3.5/6 fill Psi via judge + eval, P15.3.7 stitches and
promotes through the AHE pipeline. The snapshot here is the hand-off
artifact between P15.3.3 and P15.3.7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from router.config_io import get_current_version, save_config
from router.training.result import KMeansTrainResult


def snapshot_clusters_only(
    result: KMeansTrainResult,
    *,
    cost_weight: float = 0.0,
    versions_dir: Optional[Path] = None,
) -> Path:
    """Persist a cluster-only router_config snapshot.

    The artifact has:
      * version = (current_version or 0) + 1
      * centroids set in a sibling .npz file
      * model_psi = {} (P15.3.6 fills it)
      * cost_weight = 0.0 (default; P15.3.7 picks a real one)
      * fitted_from = result.fitted_from (provenance)
      * metadata.stage = "clusters_only" (so loaders know Psi is empty)

    The current pointer is NOT updated. P15.3.7's executor promotes when
    Psi is computed and the critic + approver pass.

    Returns:
        The path to the written JSON artifact.
    """
    next_version = (get_current_version(versions_dir=versions_dir) or 0) + 1
    payload = {
        "version": next_version,
        "k": result.k,
        # Centroids live in the sidecar .npz — JSON references it implicitly
        # via the version number (router_config_v<n>_centroids.npz). For
        # readability we also omit them here rather than duplicate.
        "centroids": None,
        "model_psi": {},
        "cost_weight": float(cost_weight),
        "embedder_model": result.embedder_model_id,
        "embedding_dim": int(result.assigner.embedding_dim),
        "min_corpus_size": 200,
        "created_at": result.fitted_at,
        "fitted_from": dict(result.fitted_from),
        "drift_baseline": None,
        "metadata": {
            "phase": "P15.3.3",
            "stage": "clusters_only",
            "silhouette": float(result.silhouette),
            "inertia": float(result.inertia),
            "n_samples": int(result.n_samples),
        },
    }
    return save_config(
        payload,
        centroids=result.assigner.centroids,
        versions_dir=versions_dir,
        update_pointer=False,
    )
