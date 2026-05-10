"""K-Means trainer.

Implements Section 5.1 of the UniRoute paper: unsupervised K-Means
clustering on the training set S_tr to define K representative prompt
clusters.

Differences vs the reference implementation:
- accepts a flat ``list[str]`` (P15.3.4 will retrofit a PromptDataset
  overload via a one-line guard)
- returns a ``KMeansTrainResult`` dataclass with quality metrics +
  provenance (so the proposer in P15.3.7 has everything it needs to
  log to the ledger without re-deriving anything)
- requires a ``fitted_from`` provenance argument — the caller must say
  where the corpus came from (production traces, golden seed, synthetic)
- silhouette computed on a subsample (capped 5000) to keep the O(N²)
  cost bounded
- emits INFO-level logs through ``logging.getLogger("router.training.kmeans")``
  instead of ``verbose=True print()``
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from router.core.clustering import KMeansClusterAssigner
from router.core.embeddings import PromptEmbedder
from router.errors import KMeansFitError, NotEnoughDataError
from router.training.gate import DEFAULT_MIN_CORPUS_SIZE, check_first_fit_eligibility
from router.training.result import KMeansTrainResult

logger = logging.getLogger("router.training.kmeans")


SILHOUETTE_SAMPLE_CAP = 5000


class KMeansTrainer:
    """Trains K-Means clusters on prompt embeddings.

    This is the unsupervised approach from Section 5.1. Prompts are
    clustered purely based on their embedding similarity, no labels.
    """

    def __init__(self, embedder: PromptEmbedder, num_clusters: int = 100):
        self.embedder = embedder
        self.num_clusters = num_clusters

    # ------------------------------------------------------------------
    # Single-K fit
    # ------------------------------------------------------------------

    def train(
        self,
        prompts,
        *,
        fitted_from: dict,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        silhouette_sample: int = SILHOUETTE_SAMPLE_CAP,
        min_corpus_size: int = DEFAULT_MIN_CORPUS_SIZE,
    ) -> KMeansTrainResult:
        """Fit K-Means on ``prompts`` and return a KMeansTrainResult.

        Args:
            prompts: Either a flat list of prompt strings, or a
                ``PromptDataset`` (the trainer pulls ``.get_prompts()``
                under the hood). The PromptDataset overload was added in
                P15.3.4; before that the trainer was list-only.
            fitted_from: Provenance dict — must say where the corpus came
                from. The proposer / ledger reads this back when
                explaining a router_config promotion. Required (no default)
                so callers can't forget.
            random_state: Random seed for reproducibility.
            n_init: Number of K-Means initializations.
            max_iter: Max iterations per initialization.
            silhouette_sample: Cap for the silhouette calculation
                (sklearn's ``silhouette_score`` is O(N²)). When N is
                larger we sample uniformly without replacement.
            min_corpus_size: Floor below which we refuse to fit.

        Raises:
            NotEnoughDataError: When corpus_size < min_corpus_size or
                N < 2 * K.
            KMeansFitError: When sklearn's KMeans hits max_iter without
                converging.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError as e:
            raise ImportError(
                "scikit-learn required for K-Means training. "
                "Install with: uv sync --extra router"
            ) from e

        # P15.3.4 polymorphism: accept PromptDataset by pulling .get_prompts().
        # Lazy import to avoid a hard router.training -> router.data dep at
        # module load time.
        from router.data.dataset import PromptDataset
        if isinstance(prompts, PromptDataset):
            prompts = prompts.get_prompts()

        eligible, reason = check_first_fit_eligibility(
            corpus_size=len(prompts),
            min_corpus_size=min_corpus_size,
            requested_k=self.num_clusters,
        )
        if not eligible:
            raise NotEnoughDataError(reason)

        embedder_model_id = _embedder_model_id(self.embedder)
        logger.info(
            "fit start n=%d k=%d embedder=%s",
            len(prompts),
            self.num_clusters,
            embedder_model_id,
        )

        embeddings = self.embedder.embed_batch(prompts)

        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            verbose=0,
        )
        kmeans.fit(embeddings)

        # Convergence check. KMeans sets n_iter_ to the number of iterations
        # the BEST init used; if it equals max_iter the best run hit the
        # ceiling — flag it.
        if int(kmeans.n_iter_) >= max_iter:
            raise KMeansFitError(
                f"KMeans did not converge in {max_iter} iters "
                f"(n_iter_={kmeans.n_iter_}, inertia={kmeans.inertia_:.4f})"
            )

        labels = np.asarray(kmeans.labels_)
        cluster_sizes = _labels_to_cluster_sizes(labels, k=self.num_clusters)
        sil = _silhouette(embeddings, labels, silhouette_sample, random_state)

        result = KMeansTrainResult(
            assigner=KMeansClusterAssigner(kmeans.cluster_centers_),
            k=self.num_clusters,
            n_samples=len(prompts),
            silhouette=sil,
            inertia=float(kmeans.inertia_),
            cluster_sizes=cluster_sizes,
            embedder_model_id=embedder_model_id,
            fitted_at=_now_iso(),
            fitted_from=dict(fitted_from),
        )
        logger.info("fit done %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Multi-K sweep with silhouette-based selection
    # ------------------------------------------------------------------

    def train_with_validation(
        self,
        train_prompts,
        val_prompts,
        k_values: list[int],
        *,
        fitted_from: dict,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        silhouette_sample: int = SILHOUETTE_SAMPLE_CAP,
        min_corpus_size: int = DEFAULT_MIN_CORPUS_SIZE,
    ) -> KMeansTrainResult:
        """Try several K and pick the one with highest silhouette on val.

        Returns the KMeansTrainResult of the best K. The chosen K's
        ``self.num_clusters`` is preserved through the result.

        Both ``train_prompts`` and ``val_prompts`` accept either a flat
        ``list[str]`` or a ``PromptDataset`` (P15.3.4 polymorphism).
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError as e:
            raise ImportError(
                "scikit-learn required for K-Means training. "
                "Install with: uv sync --extra router"
            ) from e

        from router.data.dataset import PromptDataset
        if isinstance(train_prompts, PromptDataset):
            train_prompts = train_prompts.get_prompts()
        if isinstance(val_prompts, PromptDataset):
            val_prompts = val_prompts.get_prompts()

        eligible, reason = check_first_fit_eligibility(
            corpus_size=len(train_prompts),
            min_corpus_size=min_corpus_size,
        )
        if not eligible:
            raise NotEnoughDataError(reason)

        # K candidates must all satisfy N >= 2K and K >= 2.
        eligible_ks = [
            k for k in k_values if k >= 2 and len(train_prompts) >= 2 * k
        ]
        if not eligible_ks:
            raise NotEnoughDataError(
                f"no eligible K in {k_values} given train_size={len(train_prompts)}"
            )

        embedder_model_id = _embedder_model_id(self.embedder)
        train_emb = self.embedder.embed_batch(train_prompts)
        val_emb = self.embedder.embed_batch(val_prompts)

        best: Optional[KMeansTrainResult] = None
        for k in eligible_ks:
            logger.info("sweep k=%d", k)
            km = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
            )
            km.fit(train_emb)

            val_labels = km.predict(val_emb)
            score = _silhouette(val_emb, val_labels, silhouette_sample, random_state)
            train_labels = np.asarray(km.labels_)
            sizes = _labels_to_cluster_sizes(train_labels, k=k)
            logger.info("k=%d silhouette=%.4f inertia=%.2f", k, score, km.inertia_)

            candidate = KMeansTrainResult(
                assigner=KMeansClusterAssigner(km.cluster_centers_),
                k=k,
                n_samples=len(train_prompts),
                silhouette=score,
                inertia=float(km.inertia_),
                cluster_sizes=sizes,
                embedder_model_id=embedder_model_id,
                fitted_at=_now_iso(),
                fitted_from=dict(fitted_from),
            )
            # Treat NaN as worse than any real score.
            if (
                best is None
                or (np.isnan(best.silhouette) and not np.isnan(candidate.silhouette))
                or (
                    not np.isnan(candidate.silhouette)
                    and candidate.silhouette > best.silhouette
                )
            ):
                best = candidate

        assert best is not None  # eligible_ks is non-empty here
        # Sync self.num_clusters to the chosen K so subsequent .train() calls
        # on this trainer use the picked value.
        self.num_clusters = best.k
        logger.info("best %s", best.summary())
        return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class KMeansPlusPlusInitializer:
    """K-Means++ initialization for better centroid selection.

    Can be used to initialize centroids before training, or to create a
    custom KMeans implementation. Kept here for parity with the reference;
    the trainer uses sklearn's built-in init.
    """

    @staticmethod
    def initialize(
        embeddings: np.ndarray,
        k: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        n_samples, _ = embeddings.shape

        first_idx = rng.integers(0, n_samples)
        centroids = [embeddings[first_idx]]

        for _ in range(1, k):
            centroids_arr = np.array(centroids)
            distances = np.min(
                np.linalg.norm(
                    embeddings[:, np.newaxis] - centroids_arr,
                    axis=2,
                ),
                axis=1,
            )
            distances_sq = distances**2
            probs = distances_sq / distances_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(embeddings[next_idx])

        return np.array(centroids)


def analyze_clusters(
    prompts: list[str],
    assigner: KMeansClusterAssigner,
    embedder: PromptEmbedder,
    top_n: int = 3,
) -> dict:
    """Group prompts by cluster + return size statistics + examples.

    Used by the UI Router config drawer (P15.3.10) and by debug surfaces.
    """
    embeddings = embedder.embed_batch(prompts)
    results = assigner.assign_batch(embeddings)

    clusters: dict[int, list[str]] = {i: [] for i in range(assigner.num_clusters)}
    for prompt, result in zip(prompts, results):
        clusters[result.cluster_id].append(prompt)

    sizes = [len(v) for v in clusters.values()]
    return {
        "num_clusters": assigner.num_clusters,
        "num_samples": len(prompts),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "cluster_examples": {k: v[:top_n] for k, v in clusters.items()},
        "size_stats": {
            "min": int(min(sizes)) if sizes else 0,
            "max": int(max(sizes)) if sizes else 0,
            "mean": float(np.mean(sizes)) if sizes else 0.0,
            "std": float(np.std(sizes)) if sizes else 0.0,
            "empty_clusters": sum(1 for s in sizes if s == 0),
        },
    }


def _silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_cap: int,
    random_state: int,
) -> float:
    """Silhouette score with subsampling for large N. Returns NaN when
    sklearn rejects the input (single cluster, etc.).
    """
    from sklearn.metrics import silhouette_score

    n = len(labels)
    if n < 2 or len(set(labels.tolist())) < 2:
        return float("nan")

    if n > sample_cap:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_cap, replace=False)
        emb_sub = embeddings[idx]
        lab_sub = labels[idx]
    else:
        emb_sub = embeddings
        lab_sub = labels

    if len(set(lab_sub.tolist())) < 2:
        return float("nan")

    try:
        return float(silhouette_score(emb_sub, lab_sub))
    except ValueError:
        return float("nan")


def _labels_to_cluster_sizes(labels: np.ndarray, k: int) -> dict[int, int]:
    sizes = {i: 0 for i in range(k)}
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        sizes[int(u)] = int(c)
    return sizes


def _embedder_model_id(embedder: PromptEmbedder) -> str:
    """Extract a stable model_id from whatever provider the embedder wraps."""
    provider = getattr(embedder, "provider", None)
    name = getattr(provider, "model_name", None)
    if isinstance(name, str) and name:
        return name
    # MockEmbeddingProvider has no model_name; mark explicitly so logs are honest.
    return "mock"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
