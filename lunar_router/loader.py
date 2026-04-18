"""
Lunar Router Loader: High-level functions for loading pre-trained routers.

Provides simple functions to:
1. Download weights and load a ready-to-use router
2. Create routers from local weights
3. Load routers from state managers

Supports two backends:
- "python": Pure Python (default if Go engine not available)
- "go": Go engine (faster, auto-detected)
- "auto": Use Go if available, fall back to Python
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

from .core.embeddings import PromptEmbedder, SentenceTransformerProvider
from .core.clustering import load_cluster_assigner, ClusterAssigner
from .models.llm_registry import LLMRegistry
from .models.llm_profile import LLMProfile
from .router.uniroute import UniRouteRouter, RoutingDecision, RoutingStats
from .storage.state_manager import StateManager
from .weights import download_weights, get_weights_path

logger = logging.getLogger(__name__)


def _go_engine_available() -> bool:
    """Check if the Go engine binary is available."""
    try:
        from .engine import _find_binary
        _find_binary()
        return True
    except (ImportError, FileNotFoundError):
        return False


def load_router(
    weights_path: Optional[Path] = None,
    weights_name: str = "default",
    embedding_model: str = "all-MiniLM-L6-v2",
    cost_weight: float = 0.0,
    use_soft_assignment: bool = True,
    allowed_models: Optional[List[str]] = None,
    download_if_missing: bool = True,
    verbose: bool = True,
    engine: str = "go",
) -> UniRouteRouter:
    """Load a UniRoute router from pre-trained weights.

    Main entry point for the auto-routing layer. Downloads weights on first
    run and returns a router you can call ``.route(prompt)`` on.

    Args:
        weights_path: Path to weights directory. If None, downloads from registry.
        weights_name: Name of weights to download if weights_path is None.
        embedding_model: SentenceTransformers model name (only used by Python backend).
        cost_weight: Lambda parameter for cost-quality tradeoff.
        use_soft_assignment: Use soft cluster probabilities (Python backend only).
        allowed_models: Restrict routing to these models.
        download_if_missing: Download weights if not found locally.
        verbose: Print progress information.
        engine: Backend engine. **Default: ``"go"``** — the Go engine is the
            canonical/production path. Pass ``"auto"`` only if you want a
            silent fallback to Python when the Go binary is missing; pass
            ``"python"`` only for the research/offline path.
            - ``"go"``: Use the Go engine. Raises if binary is unavailable. (Default.)
            - ``"auto"``: Prefer Go, fall back to Python if Go binary is missing.
            - ``"python"``: Force the pure-Python backend.

    Returns:
        Configured UniRouteRouter ready for use.

    Example:
        >>> from lunar_router import load_router
        >>> router = load_router()                       # Go engine (default)
        >>> decision = router.route("What is machine learning?")
        >>> print(f"Best model: {decision.selected_model}")
        >>>
        >>> router = load_router(engine="python")        # research/offline path
        >>> router = load_router(engine="auto")          # silent-fallback mode
    """
    # Determine weights path
    if weights_path is None:
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

    # Decide which engine to use
    use_go = False
    if engine == "go":
        use_go = True
    elif engine == "auto":
        use_go = _go_engine_available()
    elif engine != "python":
        raise ValueError(f"Unknown engine: {engine!r}. Use 'auto', 'go', or 'python'.")

    if use_go:
        return _load_go_router(
            weights_path=weights_path,
            cost_weight=cost_weight,
            allowed_models=allowed_models,
            verbose=verbose,
        )

    return _load_python_router(
        weights_path=weights_path,
        embedding_model=embedding_model,
        cost_weight=cost_weight,
        use_soft_assignment=use_soft_assignment,
        allowed_models=allowed_models,
        verbose=verbose,
    )


def _load_go_router(
    weights_path: Path,
    cost_weight: float,
    allowed_models: Optional[List[str]],
    verbose: bool,
) -> "GoBackedRouter":
    """Load a router backed by the Go engine."""
    from .engine import GoEngine

    if verbose:
        print(f"Loading Go engine with weights: {weights_path}")

    engine = GoEngine(
        weights_path=str(weights_path),
    )
    engine.start()

    if verbose:
        health = engine.health()
        print(
            f"Go engine ready! "
            f"Models: {health['num_models']}, "
            f"Clusters: {health['num_clusters']}, "
            f"Embedder: {health['embedder_ready']}"
        )

    return GoBackedRouter(
        engine=engine,
        cost_weight=cost_weight,
        allowed_models=allowed_models,
    )


def _load_python_router(
    weights_path: Path,
    embedding_model: str,
    cost_weight: float,
    use_soft_assignment: bool,
    allowed_models: Optional[List[str]],
    verbose: bool,
) -> UniRouteRouter:
    """Load a router using the pure Python backend."""
    if verbose:
        print(f"Loading router from: {weights_path}")

    # 1. Initialize embedder
    if verbose:
        print(f"  Initializing embedder: {embedding_model}")
    provider = SentenceTransformerProvider(model_name=embedding_model)
    embedder = PromptEmbedder(provider, cache_enabled=True)

    # 2. Load cluster assigner
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


class GoBackedRouter:
    """
    Router backed by the Go engine.

    Exposes the same interface as UniRouteRouter so it's a drop-in replacement.
    The Go engine runs as a subprocess and communicates via localhost HTTP.
    """

    def __init__(
        self,
        engine,
        cost_weight: float = 0.0,
        allowed_models: Optional[List[str]] = None,
    ):
        self._engine = engine
        self.cost_weight = cost_weight
        self.allowed_models = allowed_models
        self._stats = RoutingStats()

    @property
    def stats(self) -> RoutingStats:
        """Return routing statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = RoutingStats()
        self._engine.metrics_reset()

    def route(
        self,
        prompt: str,
        available_models: Optional[List[str]] = None,
        cost_weight_override: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Route a prompt to the best LLM.

        Args:
            prompt: The input prompt text.
            available_models: Optional list of model IDs to consider.
            cost_weight_override: Override the default cost_weight for this request.

        Returns:
            RoutingDecision with the selected model and metrics.
        """
        import numpy as np

        models = available_models if available_models is not None else self.allowed_models
        cw = cost_weight_override if cost_weight_override is not None else self.cost_weight

        result = self._engine.route(
            prompt=prompt,
            available_models=models,
            cost_weight=cw,
        )

        decision = RoutingDecision(
            selected_model=result["selected_model"],
            expected_error=result["expected_error"],
            cost_adjusted_score=result["cost_adjusted_score"],
            all_scores=result.get("all_scores", {}),
            cluster_id=result["cluster_id"],
            cluster_probabilities=np.array(result.get("cluster_probabilities", [])),
            reasoning=result.get("reasoning"),
        )

        self._stats.update(decision)
        return decision

    def route_embedding(
        self,
        embedding: List[float],
        available_models: Optional[List[str]] = None,
        cost_weight_override: Optional[float] = None,
    ) -> RoutingDecision:
        """Route a pre-computed embedding to the best model."""
        import numpy as np

        models = available_models if available_models is not None else self.allowed_models
        cw = cost_weight_override if cost_weight_override is not None else self.cost_weight

        result = self._engine.route_embedding(
            embedding=embedding,
            available_models=models,
            cost_weight=cw,
        )

        decision = RoutingDecision(
            selected_model=result["selected_model"],
            expected_error=result["expected_error"],
            cost_adjusted_score=result["cost_adjusted_score"],
            all_scores=result.get("all_scores", {}),
            cluster_id=result["cluster_id"],
            cluster_probabilities=np.array(result.get("cluster_probabilities", [])),
            reasoning=result.get("reasoning"),
        )

        self._stats.update(decision)
        return decision

    def route_batch(
        self,
        prompts: List[str],
        available_models: Optional[List[str]] = None,
        cost_weight_override: Optional[float] = None,
    ) -> List[RoutingDecision]:
        """Route multiple prompts."""
        return [
            self.route(prompt, available_models, cost_weight_override)
            for prompt in prompts
        ]

    # --- Chat Completions (Gateway) ---

    def vision(
        self,
        image: str,
        prompt: str = "Describe this image.",
        model: str = "auto",
        max_tokens: Optional[int] = None,
        detail: str = "auto",
    ) -> dict:
        """
        Send an image to a vision model and get a response.

        Args:
            image: Base64-encoded image string, data URI, or URL.
            prompt: Text prompt to accompany the image.
            model: Model name, or "auto" for semantic routing.
            max_tokens: Maximum tokens to generate.
            detail: Image detail level ("low", "high", "auto").

        Returns:
            OpenAI-compatible chat completion response dict.

        Example:
            >>> import base64
            >>> with open("photo.jpg", "rb") as f:
            ...     img = base64.b64encode(f.read()).decode()
            >>> resp = router.vision(img, "What is this?", model="gpt-4o")
            >>> print(resp["choices"][0]["message"]["content"])
        """
        return self._engine.vision(
            image=image,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            detail=detail,
        )

    def chat(
        self,
        messages: List[dict],
        model: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """
        Send a chat completion request via the Go gateway.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}].
            model: Model name (e.g. "gpt-4o-mini"), or "auto" for semantic routing.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stream: If True, returns a generator of SSE chunks.
            **kwargs: Additional OpenAI-compatible params (top_p, stop, etc.).

        Returns:
            OpenAI-compatible chat completion response dict.

        Example:
            >>> router = load_router()
            >>> resp = router.chat(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     model="gpt-4o-mini",
            ... )
            >>> print(resp["choices"][0]["message"]["content"])
        """
        return self._engine.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        model: str = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate a text response. Convenience wrapper.

        Args:
            prompt: The user prompt.
            model: Model name, or "auto" for semantic routing.

        Returns:
            The generated text string.

        Example:
            >>> router = load_router()
            >>> text = router.generate("What is 2+2?", model="gpt-4o-mini")
            >>> print(text)
        """
        return self._engine.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def smart_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cost_weight: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """
        Route to the best model and generate a response in one call.

        Args:
            prompt: The user prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            cost_weight: Override cost weight for routing.

        Returns:
            Dict with 'text', 'model', 'routing', and 'usage'.
        """
        # Step 1: Route
        decision = self.route(prompt, cost_weight_override=cost_weight)

        # Step 2: Generate with selected model
        resp = self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=decision.selected_model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return {
            "text": resp["choices"][0]["message"]["content"],
            "model": decision.selected_model,
            "routing": decision.to_dict(),
            "usage": resp.get("usage"),
        }

    # --- Models & Metrics ---

    def list_models(self) -> List[dict]:
        """List all available models with profiles."""
        result = self._engine.models()
        return result.get("models", [])

    def model(self, model_id: str) -> dict:
        """Get info for a specific model."""
        return self._engine.model(model_id)

    def metrics(self) -> dict:
        """Get Go engine metrics (latency, tokens, cost, errors, per-provider/model)."""
        return self._engine.metrics()

    def metrics_recent(self, n: int = 20) -> List[dict]:
        """Get last N raw request metrics."""
        return self._engine.metrics_recent(n)

    def health(self) -> dict:
        """Check engine health."""
        return self._engine.health()

    # --- Cache ---

    def cache_stats(self) -> dict:
        """Get routing cache statistics (size, hits, misses, hit_rate)."""
        return self._engine.cache_stats()

    def cache_clear(self) -> None:
        """Clear the routing cache."""
        self._engine.cache_clear()

    # --- Lifecycle ---

    def close(self) -> None:
        """Stop the Go engine."""
        self._engine.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        health = {}
        try:
            health = self._engine.health()
        except Exception:
            pass
        return (
            f"GoBackedRouter("
            f"models={health.get('num_models', '?')}, "
            f"clusters={health.get('num_clusters', '?')}, "
            f"engine={self._engine.base_url})"
        )


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
    Always uses the Python backend (Go engine requires weight files on disk).

    Args:
        cluster_assigner: Pre-trained cluster assigner.
        registry: Registry with model profiles.
        embedding_model: SentenceTransformers model name.
        cost_weight: Lambda parameter.
        use_soft_assignment: Use soft cluster probabilities.
        allowed_models: Restrict routing to these models.

    Returns:
        Configured UniRouteRouter.
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
