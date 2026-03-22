"""
Lunar Router Loader: High-level functions for loading pre-trained routers.

Provides simple functions to:
1. Download weights and load a ready-to-use router
2. Create routers from local weights
3. Load routers from state managers
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

from .core.embeddings import PromptEmbedder, SentenceTransformerProvider
from .core.clustering import load_cluster_assigner, ClusterAssigner
from .models.llm_registry import LLMRegistry
from .models.llm_profile import LLMProfile
from .router.uniroute import UniRouteRouter
from .storage.state_manager import StateManager
from .weights import download_weights, get_weights_path

logger = logging.getLogger(__name__)


def load_router(
    weights_path: Optional[Path] = None,
    weights_name: str = "default",
    embedding_model: str = "all-MiniLM-L6-v2",
    cost_weight: float = 0.0,
    use_soft_assignment: bool = True,
    allowed_models: Optional[List[str]] = None,
    download_if_missing: bool = True,
    verbose: bool = True,
) -> UniRouteRouter:
    """
    Load a UniRoute router from pre-trained weights.

    This is the main entry point for using UniRoute with pre-trained weights.

    Args:
        weights_path: Path to weights directory. If None, downloads from registry.
        weights_name: Name of weights to download if weights_path is None.
        embedding_model: SentenceTransformers model name for embeddings.
        cost_weight: Lambda parameter for cost-quality tradeoff.
        use_soft_assignment: Use soft cluster probabilities.
        allowed_models: Restrict routing to these models.
        download_if_missing: Download weights if not found locally.
        verbose: Print progress information.

    Returns:
        Configured UniRouteRouter ready for use.

    Example:
        >>> from lunar_router import load_router
        >>>
        >>> # Load with default weights (downloads if needed)
        >>> router = load_router()
        >>>
        >>> # Route a prompt
        >>> decision = router.route("What is machine learning?")
        >>> print(f"Best model: {decision.selected_model}")
        >>>
        >>> # Load with specific weights and cost preference
        >>> router = load_router(weights_name="mmlu-v1", cost_weight=0.5)
    """
    # Determine weights path
    if weights_path is None:
        # Use download_weights which checks bundled weights first
        if download_if_missing:
            weights_path = download_weights(weights_name, verbose=verbose)
        else:
            weights_path = get_weights_path(weights_name)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Weights not found at {weights_path}. "
                    f"Set download_if_missing=True to download automatically."
                )

    weights_path = Path(weights_path)

    if verbose:
        print(f"Loading router from: {weights_path}")

    # 1. Initialize embedder
    if verbose:
        print(f"  Initializing embedder: {embedding_model}")
    provider = SentenceTransformerProvider(model_name=embedding_model)
    embedder = PromptEmbedder(provider, cache_enabled=True)

    # 2. Load cluster assigner
    # Try different possible locations for cluster file
    cluster_candidates = [
        weights_path / "clusters" / "mmlu_full.npz",
        weights_path / "clusters" / "default.npz",
        weights_path / "clusters.npz",
    ]

    cluster_assigner = None
    for cluster_path in cluster_candidates:
        if cluster_path.exists():
            if verbose:
                print(f"  Loading clusters: {cluster_path.name}")
            cluster_assigner = load_cluster_assigner(cluster_path)
            break

    if cluster_assigner is None:
        raise FileNotFoundError(
            f"No cluster file found in {weights_path}. "
            f"Tried: {[str(p) for p in cluster_candidates]}"
        )

    # 3. Load profiles into registry
    profiles_path = weights_path / "profiles"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles directory not found: {profiles_path}")

    if verbose:
        print(f"  Loading profiles from: {profiles_path}")

    registry = LLMRegistry()
    profile_count = 0
    for profile_file in profiles_path.glob("*.json"):
        try:
            profile = LLMProfile.load(profile_file)
            registry.register(profile)
            profile_count += 1
        except Exception as e:
            logger.warning(f"Failed to load profile {profile_file}: {e}")

    if profile_count == 0:
        raise ValueError(f"No valid profiles found in {profiles_path}")

    if verbose:
        profiles_loaded = [p.model_id for p in registry.get_all()]
        print(f"  Loaded {profile_count} profiles: {profiles_loaded}")

    # 4. Create router
    router = UniRouteRouter(
        embedder=embedder,
        cluster_assigner=cluster_assigner,
        registry=registry,
        cost_weight=cost_weight,
        use_soft_assignment=use_soft_assignment,
        allowed_models=allowed_models,
    )

    if verbose:
        print(f"Router ready! Clusters: {cluster_assigner.num_clusters}, "
              f"Models: {len(registry)}")

    return router


def create_router(
    cluster_assigner: ClusterAssigner,
    registry: LLMRegistry,
    embedding_model: str = "all-MiniLM-L6-v2",
    cost_weight: float = 0.0,
    use_soft_assignment: bool = True,
    allowed_models: Optional[List[str]] = None,
) -> UniRouteRouter:
    """
    Create a router from components.

    Useful when you have custom-trained clusters and profiles.

    Args:
        cluster_assigner: Pre-trained cluster assigner.
        registry: Registry with model profiles.
        embedding_model: SentenceTransformers model name.
        cost_weight: Lambda parameter.
        use_soft_assignment: Use soft cluster probabilities.
        allowed_models: Restrict routing to these models.

    Returns:
        Configured UniRouteRouter.

    Example:
        >>> from lunar_router import create_router, KMeansTrainer
        >>> from lunar_router import PromptEmbedder, LLMRegistry
        >>>
        >>> # After training
        >>> router = create_router(
        ...     cluster_assigner=trained_assigner,
        ...     registry=trained_registry,
        ...     cost_weight=0.3,
        ... )
    """
    provider = SentenceTransformerProvider(model_name=embedding_model)
    embedder = PromptEmbedder(provider, cache_enabled=True)

    return UniRouteRouter(
        embedder=embedder,
        cluster_assigner=cluster_assigner,
        registry=registry,
        cost_weight=cost_weight,
        use_soft_assignment=use_soft_assignment,
        allowed_models=allowed_models,
    )


def load_router_from_state(
    state_path: Path,
    cluster_name: str = "default",
    embedding_model: str = "all-MiniLM-L6-v2",
    cost_weight: float = 0.0,
    use_soft_assignment: bool = True,
    allowed_models: Optional[List[str]] = None,
    verbose: bool = True,
) -> UniRouteRouter:
    """
    Load a router from a StateManager directory.

    Args:
        state_path: Path to StateManager directory.
        cluster_name: Name of cluster assigner to load.
        embedding_model: SentenceTransformers model name.
        cost_weight: Lambda parameter.
        use_soft_assignment: Use soft cluster probabilities.
        allowed_models: Restrict routing to these models.
        verbose: Print progress information.

    Returns:
        Configured UniRouteRouter.
    """
    state = StateManager(state_path)

    if verbose:
        summary = state.get_summary()
        print(f"Loading from StateManager: {state_path}")
        print(f"  Available clusters: {summary['cluster_assigners']}")
        print(f"  Available profiles: {summary['profiles']}")

    # Load components
    cluster_assigner = state.load_cluster_assigner(cluster_name)
    registry = state.load_registry()

    # Create router
    return create_router(
        cluster_assigner=cluster_assigner,
        registry=registry,
        embedding_model=embedding_model,
        cost_weight=cost_weight,
        use_soft_assignment=use_soft_assignment,
        allowed_models=allowed_models,
    )
