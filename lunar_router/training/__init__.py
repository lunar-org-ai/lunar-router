"""
UniRoute Training Module.

Provides tools for training a complete UniRoute routing system:

1. Cluster Training (K-Means):
   - KMeansTrainer: Train clusters from prompt embeddings
   - analyze_clusters: Analyze cluster composition

2. Model Profiling:
   - Use profiler.ModelProfiler to compute Psi vectors

3. High-Level Pipeline:
   - full_training_pipeline: End-to-end training
   - train_clusters: Train just the clusters
   - profile_models: Profile just the models
   - export_weights: Export for distribution

Example:
    >>> from lunar_router.training import full_training_pipeline, TrainingConfig
    >>> from lunar_router import PromptDataset, LLMClient
    >>>
    >>> # Load data
    >>> train_data = PromptDataset.from_jsonl("train.jsonl")
    >>> val_data = PromptDataset.from_jsonl("val.jsonl")
    >>>
    >>> # Create LLM clients
    >>> clients = [LLMClient("gpt-4o", api_key=key)]
    >>>
    >>> # Train
    >>> result = full_training_pipeline(train_data, val_data, clients)
"""

from .kmeans_trainer import KMeansTrainer, KMeansPlusPlusInitializer, analyze_clusters
from .learned_map_trainer import LearnedMapTrainer
from .pipeline import (
    full_training_pipeline,
    train_clusters,
    profile_models,
    export_weights,
    quick_train,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    # K-Means training
    "KMeansTrainer",
    "KMeansPlusPlusInitializer",
    "analyze_clusters",
    # Learned Map training
    "LearnedMapTrainer",
    # High-level pipeline
    "full_training_pipeline",
    "train_clusters",
    "profile_models",
    "export_weights",
    "quick_train",
    "TrainingConfig",
    "TrainingResult",
]
