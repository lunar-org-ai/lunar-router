"""
Full Training Pipeline: End-to-end training for UniRoute.

Provides high-level functions to train a complete UniRoute system:
1. Train clusters from prompt embeddings
2. Profile models on validation data
3. Export weights for distribution
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable
import json

from ..core.embeddings import PromptEmbedder, SentenceTransformerProvider
from ..core.clustering import ClusterAssigner, KMeansClusterAssigner
from ..data.dataset import PromptDataset
from ..models.llm_client import LLMClient
from ..models.llm_profile import LLMProfile
from ..models.llm_registry import LLMRegistry
from ..storage.state_manager import StateManager
from .kmeans_trainer import KMeansTrainer, analyze_clusters


@dataclass
class TrainingConfig:
    """Configuration for training a UniRoute system."""

    # Cluster training
    num_clusters: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    random_state: int = 42

    # Profiling
    profiling_max_tokens: int = 256
    profiling_temperature: float = 0.0
    checkpoint_every: int = 10

    # Output
    output_dir: Optional[Path] = None
    weights_name: str = "custom"
    weights_version: str = "1.0.0"
    weights_description: str = "Custom trained UniRoute weights"

    # Model metadata
    models_to_profile: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "embedding_model": self.embedding_model,
            "kmeans_n_init": self.kmeans_n_init,
            "kmeans_max_iter": self.kmeans_max_iter,
            "random_state": self.random_state,
            "profiling_max_tokens": self.profiling_max_tokens,
            "profiling_temperature": self.profiling_temperature,
            "weights_name": self.weights_name,
            "weights_version": self.weights_version,
            "weights_description": self.weights_description,
            "models_to_profile": self.models_to_profile,
        }


@dataclass
class TrainingResult:
    """Result of training a UniRoute system."""

    cluster_assigner: ClusterAssigner
    profiles: List[LLMProfile]
    registry: LLMRegistry
    embedder: PromptEmbedder
    config: TrainingConfig
    output_path: Optional[Path] = None

    @property
    def num_clusters(self) -> int:
        return self.cluster_assigner.num_clusters

    @property
    def num_models(self) -> int:
        return len(self.profiles)


def train_clusters(
    training_data: PromptDataset,
    num_clusters: int = 100,
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    verbose: bool = True,
) -> tuple[KMeansClusterAssigner, PromptEmbedder]:
    """
    Train K-Means clusters from prompt data.

    This is Step 1 of the UniRoute training pipeline.

    Args:
        training_data: Dataset of prompts for clustering.
        num_clusters: Number of clusters K to create.
        embedding_model: SentenceTransformers model name.
        random_state: Random seed for reproducibility.
        n_init: Number of K-Means initializations.
        max_iter: Maximum iterations per initialization.
        verbose: Whether to print progress.

    Returns:
        Tuple of (cluster_assigner, embedder).

    Example:
        >>> from lunar_router.training import train_clusters
        >>> from lunar_router import PromptDataset
        >>>
        >>> # Load your training data
        >>> data = PromptDataset.from_jsonl("training_prompts.jsonl")
        >>>
        >>> # Train clusters
        >>> assigner, embedder = train_clusters(data, num_clusters=100)
        >>> print(f"Trained {assigner.num_clusters} clusters")
    """
    if verbose:
        print(f"Training clusters with K={num_clusters}")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Training samples: {len(training_data)}")

    # Initialize embedder
    provider = SentenceTransformerProvider(model_name=embedding_model)
    embedder = PromptEmbedder(provider, cache_enabled=True)

    # Train K-Means
    trainer = KMeansTrainer(embedder, num_clusters=num_clusters)
    cluster_assigner = trainer.train(
        training_data,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
    )

    return cluster_assigner, embedder


def profile_models(
    llm_clients: List[LLMClient],
    validation_data: PromptDataset,
    cluster_assigner: ClusterAssigner,
    embedder: PromptEmbedder,
    max_tokens: int = 256,
    temperature: float = 0.0,
    checkpoint_every: int = 10,
    verbose: bool = True,
) -> List[LLMProfile]:
    """
    Profile multiple LLMs to compute their Psi vectors.

    This is Step 2 of the UniRoute training pipeline.

    Args:
        llm_clients: List of LLM clients to profile.
        validation_data: Dataset with (prompt, ground_truth) pairs.
        cluster_assigner: Pre-trained cluster assigner.
        embedder: Embedder matching the cluster training.
        max_tokens: Max tokens for generation.
        temperature: Temperature for generation.
        checkpoint_every: Save checkpoint every N samples.
        verbose: Whether to print progress.

    Returns:
        List of LLMProfile objects.

    Example:
        >>> from lunar_router.training import profile_models
        >>> from lunar_router import LLMClient
        >>>
        >>> # Create clients for models to profile
        >>> clients = [
        ...     LLMClient("gpt-4o", api_key=openai_key),
        ...     LLMClient("gpt-4o-mini", api_key=openai_key),
        ... ]
        >>>
        >>> # Profile models
        >>> profiles = profile_models(
        ...     clients, validation_data, assigner, embedder
        ... )
    """
    from ..profiler.model_profiler import ModelProfiler

    if verbose:
        print(f"Profiling {len(llm_clients)} models")
        print(f"  Validation samples: {len(validation_data)}")
        print(f"  Clusters: {cluster_assigner.num_clusters}")

    profiler = ModelProfiler(
        embedder=embedder,
        cluster_assigner=cluster_assigner,
        validation_set=validation_data,
    )

    profiles = []
    for i, client in enumerate(llm_clients):
        if verbose:
            print(f"\n[{i+1}/{len(llm_clients)}] Profiling {client.model_id}...")

        profile = profiler.profile(
            client,
            show_progress=verbose,
            max_tokens=max_tokens,
            temperature=temperature,
            checkpoint_every=checkpoint_every,
        )
        profiles.append(profile)

        if verbose:
            print(f"  Accuracy: {profile.overall_accuracy:.2%}")

    return profiles


def export_weights(
    cluster_assigner: ClusterAssigner,
    profiles: List[LLMProfile],
    output_dir: Path,
    name: str = "custom",
    version: str = "1.0.0",
    description: str = "Custom trained UniRoute weights",
    embedding_model: str = "all-MiniLM-L6-v2",
    verbose: bool = True,
) -> Path:
    """
    Export trained weights to a directory for distribution.

    Creates a weights package with:
    - clusters/default.npz - Cluster centroids
    - profiles/*.json - Model profiles
    - weights_config.json - Metadata

    Args:
        cluster_assigner: Trained cluster assigner.
        profiles: List of model profiles.
        output_dir: Directory to export to.
        name: Weights package name.
        version: Version string.
        description: Package description.
        embedding_model: Embedding model used for training.
        verbose: Whether to print progress.

    Returns:
        Path to the exported weights directory.

    Example:
        >>> from lunar_router.training import export_weights
        >>>
        >>> export_weights(
        ...     assigner, profiles,
        ...     output_dir=Path("./my_weights"),
        ...     name="my-router-v1",
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Exporting weights to: {output_dir}")

    # 1. Save clusters
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)
    cluster_path = clusters_dir / "default.npz"
    cluster_assigner.save(cluster_path)
    if verbose:
        print(f"  Saved clusters: {cluster_path}")

    # 2. Save profiles
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    for profile in profiles:
        profile_path = profiles_dir / f"{profile.model_id.replace('/', '_')}.json"
        profile.save(profile_path)
        if verbose:
            print(f"  Saved profile: {profile.model_id}")

    # 3. Save metadata
    config = {
        "name": name,
        "version": version,
        "description": description,
        "source_type": "local",
        "source_path": str(output_dir),
        "clusters_file": "clusters/default.npz",
        "profiles_dir": "profiles",
        "num_clusters": cluster_assigner.num_clusters,
        "embedding_model": embedding_model,
        "embedding_dim": cluster_assigner.centroids.shape[1] if hasattr(cluster_assigner, 'centroids') else 384,
        "models_profiled": [p.model_id for p in profiles],
    }

    config_path = output_dir / "weights_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"  Saved config: {config_path}")
        print(f"Export complete! {len(profiles)} models, {cluster_assigner.num_clusters} clusters")

    return output_dir


def full_training_pipeline(
    training_data: PromptDataset,
    validation_data: PromptDataset,
    llm_clients: List[LLMClient],
    config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> TrainingResult:
    """
    Run the complete UniRoute training pipeline.

    This function runs the full training process:
    1. Train K-Means clusters on training data
    2. Profile each LLM on validation data
    3. Export weights to the specified directory

    Args:
        training_data: Dataset of prompts for cluster training (S_tr).
        validation_data: Dataset with (prompt, ground_truth) for profiling (S_val).
        llm_clients: List of LLM clients to profile.
        config: Training configuration. If None, uses defaults.
        verbose: Whether to print progress.

    Returns:
        TrainingResult with all trained components.

    Example:
        >>> from lunar_router.training import full_training_pipeline, TrainingConfig
        >>> from lunar_router import PromptDataset, LLMClient
        >>>
        >>> # Load data
        >>> train_data = PromptDataset.from_jsonl("train.jsonl")
        >>> val_data = PromptDataset.from_jsonl("val.jsonl")
        >>>
        >>> # Create LLM clients
        >>> clients = [
        ...     LLMClient("gpt-4o", api_key=os.environ["OPENAI_API_KEY"]),
        ...     LLMClient("gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
        ... ]
        >>>
        >>> # Configure training
        >>> config = TrainingConfig(
        ...     num_clusters=100,
        ...     output_dir=Path("./weights"),
        ...     weights_name="my-router",
        ... )
        >>>
        >>> # Run full pipeline
        >>> result = full_training_pipeline(
        ...     train_data, val_data, clients, config
        ... )
        >>>
        >>> # Use the trained router
        >>> from lunar_router import load_router
        >>> router = load_router(weights_path=result.output_path)
    """
    if config is None:
        config = TrainingConfig()

    if verbose:
        print("=" * 60)
        print("UniRoute Training Pipeline")
        print("=" * 60)
        print(f"Training samples: {len(training_data)}")
        print(f"Validation samples: {len(validation_data)}")
        print(f"Models to profile: {len(llm_clients)}")
        print(f"Clusters: {config.num_clusters}")
        print("=" * 60)

    # Step 1: Train clusters
    if verbose:
        print("\n[Step 1/3] Training Clusters")
        print("-" * 40)

    cluster_assigner, embedder = train_clusters(
        training_data,
        num_clusters=config.num_clusters,
        embedding_model=config.embedding_model,
        random_state=config.random_state,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        verbose=verbose,
    )

    # Step 2: Profile models
    if verbose:
        print("\n[Step 2/3] Profiling Models")
        print("-" * 40)

    profiles = profile_models(
        llm_clients,
        validation_data,
        cluster_assigner,
        embedder,
        max_tokens=config.profiling_max_tokens,
        temperature=config.profiling_temperature,
        checkpoint_every=config.checkpoint_every,
        verbose=verbose,
    )

    # Create registry
    registry = LLMRegistry()
    for profile in profiles:
        registry.register(profile)

    # Step 3: Export weights
    output_path = None
    if config.output_dir:
        if verbose:
            print("\n[Step 3/3] Exporting Weights")
            print("-" * 40)

        output_path = export_weights(
            cluster_assigner,
            profiles,
            config.output_dir,
            name=config.weights_name,
            version=config.weights_version,
            description=config.weights_description,
            embedding_model=config.embedding_model,
            verbose=verbose,
        )

    if verbose:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Clusters: {cluster_assigner.num_clusters}")
        print(f"Models profiled: {len(profiles)}")
        for profile in profiles:
            print(f"  - {profile.model_id}: {profile.overall_accuracy:.2%} accuracy")
        if output_path:
            print(f"Weights exported to: {output_path}")
        print("=" * 60)

    return TrainingResult(
        cluster_assigner=cluster_assigner,
        profiles=profiles,
        registry=registry,
        embedder=embedder,
        config=config,
        output_path=output_path,
    )


def quick_train(
    prompts: List[str],
    prompt_answers: List[str],
    llm_clients: List[LLMClient],
    num_clusters: int = 50,
    train_split: float = 0.8,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> TrainingResult:
    """
    Quick training from raw lists of prompts and answers.

    Convenience function for simple training scenarios.

    Args:
        prompts: List of prompt strings.
        prompt_answers: List of ground truth answers.
        llm_clients: List of LLM clients to profile.
        num_clusters: Number of clusters.
        train_split: Fraction of data for training vs validation.
        output_dir: Optional directory to export weights.
        verbose: Whether to print progress.

    Returns:
        TrainingResult with trained components.

    Example:
        >>> from lunar_router.training import quick_train
        >>>
        >>> prompts = ["What is 2+2?", "Explain photosynthesis", ...]
        >>> answers = ["4", "Photosynthesis is...", ...]
        >>> clients = [LLMClient("gpt-4o", api_key=key)]
        >>>
        >>> result = quick_train(prompts, answers, clients)
    """
    import random

    # Create dataset
    samples = list(zip(prompts, prompt_answers))
    random.shuffle(samples)

    split_idx = int(len(samples) * train_split)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Create datasets
    train_data = PromptDataset([
        {"prompt": p, "ground_truth": a} for p, a in train_samples
    ])
    val_data = PromptDataset([
        {"prompt": p, "ground_truth": a} for p, a in val_samples
    ])

    # Configure and train
    config = TrainingConfig(
        num_clusters=num_clusters,
        output_dir=output_dir,
    )

    return full_training_pipeline(
        train_data, val_data, llm_clients, config, verbose
    )
