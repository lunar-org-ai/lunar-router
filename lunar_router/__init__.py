"""
Lunar Router: Intelligent LLM Routing for Efficient Inference

Based on the paper "Universal Model Routing for Efficient LLM Inference"
(arXiv:2502.08773)

The system decides dynamically which LLM to use for each prompt,
optimizing the trade-off between quality and cost.

Supported Providers:
- OpenAI (GPT-4o, GPT-4, GPT-3.5, O1, O3)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini 2.0, Gemini 1.5)
- Groq (Llama 3.3, Mixtral, Gemma)
- Mistral (Large, Small, Codestral)
- vLLM (local models)

Quick Start:
    >>> from lunar_router import load_router, download_weights
    >>>
    >>> # Load router with pre-trained weights (downloads if needed)
    >>> router = load_router()
    >>>
    >>> # Route a prompt
    >>> decision = router.route("Explain quantum computing")
    >>> print(f"Selected model: {decision.selected_model}")

Using Clients Directly:
    >>> from lunar_router import create_client, OpenAIClient, GroqClient
    >>>
    >>> # Create clients
    >>> openai = create_client("openai", "gpt-4o-mini")
    >>> groq = create_client("groq", "llama-3.1-8b-instant")
    >>> google = create_client("google", "gemini-1.5-flash")

Training:
    >>> from lunar_router import full_training_pipeline, TrainingConfig
    >>> from lunar_router import PromptDataset, create_client
    >>>
    >>> # Load data
    >>> train_data = PromptDataset.from_jsonl("train.jsonl")
    >>> val_data = PromptDataset.from_jsonl("val.jsonl")
    >>>
    >>> # Create LLM clients for profiling
    >>> clients = [
    ...     create_client("openai", "gpt-4o"),
    ...     create_client("groq", "llama-3.1-70b-versatile"),
    ... ]
    >>>
    >>> # Run full training pipeline
    >>> config = TrainingConfig(num_clusters=100, output_dir="./weights")
    >>> result = full_training_pipeline(train_data, val_data, clients, config)
"""

__version__ = "0.1.0"

# Core components
from .core.embeddings import PromptEmbedder, SentenceTransformerProvider, EmbeddingProvider
from .core.clustering import (
    ClusterAssigner,
    ClusterResult,
    KMeansClusterAssigner,
    LearnedMapClusterAssigner,
    load_cluster_assigner,
)

# Models
from .models.llm_profile import LLMProfile
from .models.llm_registry import LLMRegistry
from .models.llm_client import (
    LLMClient,
    LLMResponse,
    OpenAIClient,
    AnthropicClient,
    MistralClient,
    GoogleClient,
    GroqClient,
    VLLMClient,
    MockLLMClient,
    create_client,
)

# Router
from .router.uniroute import UniRouteRouter, RoutingDecision, RoutingStats

# Storage
from .storage.state_manager import StateManager

# Data
from .data.dataset import PromptDataset, PromptSample

# Weights management
from .weights import (
    download_weights,
    download_from_url,
    download_from_s3,
    download_from_huggingface,
    get_weights_path,
    list_available_weights,
    WeightsConfig,
    WEIGHTS_REGISTRY,
)

# Training (optional - requires scikit-learn)
try:
    from .training.kmeans_trainer import KMeansTrainer, analyze_clusters
    from .training.learned_map_trainer import LearnedMapTrainer
    from .training.pipeline import (
        full_training_pipeline,
        train_clusters,
        profile_models,
        export_weights,
        quick_train,
        TrainingConfig,
        TrainingResult,
    )
    from .profiler.model_profiler import ModelProfiler

    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False

# High-level convenience functions
from .loader import load_router, create_router, load_router_from_state

# SDK - Unified LLM interface (like LiteLLM / OpenRouter)
from .sdk import (
    completion, acompletion, Router, parse_model, ModelResponse, StreamChunk,
    add_trace, add_traces, import_traces,
)
from .model_prices import model_cost, get_model_info, supported_models

# Hub - Download manager (like NLTK, spaCy)
from .hub import download, list_packages, info as package_info, remove, path, Hub, LUNAR_DATA_HOME


__all__ = [
    # Version
    "__version__",
    # Core
    "PromptEmbedder",
    "SentenceTransformerProvider",
    "EmbeddingProvider",
    "ClusterAssigner",
    "ClusterResult",
    "KMeansClusterAssigner",
    "LearnedMapClusterAssigner",
    "load_cluster_assigner",
    # Models
    "LLMProfile",
    "LLMRegistry",
    "LLMClient",
    "LLMResponse",
    # LLM Clients (7 providers)
    "OpenAIClient",
    "AnthropicClient",
    "MistralClient",
    "GoogleClient",
    "GroqClient",
    "VLLMClient",
    "MockLLMClient",
    "create_client",
    # Router
    "UniRouteRouter",
    "RoutingDecision",
    "RoutingStats",
    # Storage
    "StateManager",
    # Data
    "PromptDataset",
    "PromptSample",
    # Weights
    "download_weights",
    "download_from_url",
    "download_from_s3",
    "download_from_huggingface",
    "get_weights_path",
    "list_available_weights",
    "WeightsConfig",
    "WEIGHTS_REGISTRY",
    # High-level
    "load_router",
    "create_router",
    "load_router_from_state",
    # SDK (LiteLLM-style)
    "completion",
    "acompletion",
    "Router",
    "parse_model",
    "ModelResponse",
    "StreamChunk",
    "model_cost",
    "get_model_info",
    "supported_models",
    # Trace ingestion
    "add_trace",
    "add_traces",
    "import_traces",
    # Hub (like NLTK download)
    "download",
    "list_packages",
    "package_info",
    "remove",
    "path",
    "Hub",
    "LUNAR_DATA_HOME",
]

# Add training exports if available
if _TRAINING_AVAILABLE:
    __all__.extend([
        # Training classes
        "KMeansTrainer",
        "LearnedMapTrainer",
        "ModelProfiler",
        "analyze_clusters",
        # Training pipeline
        "full_training_pipeline",
        "train_clusters",
        "profile_models",
        "export_weights",
        "quick_train",
        "TrainingConfig",
        "TrainingResult",
    ])
