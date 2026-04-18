"""Lunar: the auto-distillation layer for your LLM calls.

Drop-in OpenAI-compatible SDK. Every request becomes a trace; traces become
datasets; datasets become distilled custom models; the routing layer swaps
those models in under your app via aliases — so your cost curve goes down
over time without code changes.

Quick start:
    >>> import lunar_router as lr
    >>> resp = lr.completion(
    ...     model="openai/gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello"}],
    ... )
    >>> print(resp.choices[0].message.content)
    >>> print(f"cost: ${resp._cost:.6f}  latency: {resp._latency_ms:.0f}ms")

Routing with fallbacks:
    >>> router = lr.Router(
    ...     model_list=[
    ...         {"model_name": "smart", "model": "openai/gpt-4o"},
    ...         {"model_name": "smart", "model": "anthropic/claude-sonnet-4-6"},
    ...     ],
    ...     fallbacks=[{"smart": ["deepseek/deepseek-chat"]}],
    ... )
    >>> resp = router.completion(model="smart", messages=[...])

Drop-in replacement for the OpenAI SDK (zero code changes in existing apps):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
    # All 13 providers routed through the Lunar engine; every request is a trace.

Distillation — what makes Lunar different from a plain LLM gateway:
    >>> from lunar_router import Distiller
    >>> d = Distiller()
    # Submit a dataset, pick teacher + student, and Lunar trains the distilled
    # model and serves it behind a routing alias you can point traffic at.

Providers (13):
    OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, Together, Fireworks,
    Cerebras, Sambanova, Perplexity, Cohere, Bedrock.
"""

__version__ = "0.2.1"

# ---------------------------------------------------------------------------
# Public SDK — eager imports only. Everything here must be cheap to load.
# ---------------------------------------------------------------------------
from .sdk import (
    completion, acompletion, Router, parse_model, ModelResponse, StreamChunk,
    add_trace, add_traces, import_traces,
)
from .model_prices import model_cost, get_model_info, supported_models
from .distillation.client import Distiller, TrainingClient, DistillerError
from .models.llm_client import create_client, LLMResponse

__all__ = [
    "__version__",
    # Core SDK
    "completion",
    "acompletion",
    "Router",
    "ModelResponse",
    "StreamChunk",
    "parse_model",
    # Multi-provider client factory (13 providers via sdk.PROVIDERS)
    "create_client",
    "LLMResponse",
    # Pricing / model info
    "model_cost",
    "get_model_info",
    "supported_models",
    # Trace ingestion (how user traffic feeds the distillation loop)
    "add_trace",
    "add_traces",
    "import_traces",
    # Distillation — the wedge
    "Distiller",
    "TrainingClient",
    "DistillerError",
]

# ---------------------------------------------------------------------------
# Research surface — lazy. Still importable as `from lunar_router import X`
# for backward compatibility, but not advertised in __all__, autocomplete,
# or tab-completion. Loading sentence-transformers / sklearn / sub-frameworks
# only happens when one of these names is actually touched.
#
# Power users should prefer the explicit submodule path
# (e.g. `from lunar_router.evaluation import RouterEvaluator`) — these
# shims exist so existing notebooks and docs don't break.
# ---------------------------------------------------------------------------
_LAZY: dict[str, tuple[str, str | None]] = {
    # core
    "PromptEmbedder": ("lunar_router.core.embeddings", None),
    "SentenceTransformerProvider": ("lunar_router.core.embeddings", None),
    "EmbeddingProvider": ("lunar_router.core.embeddings", None),
    "ClusterAssigner": ("lunar_router.core.clustering", None),
    "ClusterResult": ("lunar_router.core.clustering", None),
    "KMeansClusterAssigner": ("lunar_router.core.clustering", None),
    "LearnedMapClusterAssigner": ("lunar_router.core.clustering", None),
    "load_cluster_assigner": ("lunar_router.core.clustering", None),
    # models
    "LLMProfile": ("lunar_router.models.llm_profile", None),
    "LLMRegistry": ("lunar_router.models.llm_registry", None),
    "LLMClient": ("lunar_router.models.llm_client", None),
    "OpenAIClient": ("lunar_router.models.llm_client", None),
    "AnthropicClient": ("lunar_router.models.llm_client", None),
    "MistralClient": ("lunar_router.models.llm_client", None),
    "GoogleClient": ("lunar_router.models.llm_client", None),
    "GroqClient": ("lunar_router.models.llm_client", None),
    "VLLMClient": ("lunar_router.models.llm_client", None),
    "MockLLMClient": ("lunar_router.models.llm_client", None),
    "UnifiedClient": ("lunar_router.models.llm_client", None),
    # research router (paper impl)
    "UniRouteRouter": ("lunar_router.router.uniroute", None),
    "RoutingDecision": ("lunar_router.router.uniroute", None),
    "RoutingStats": ("lunar_router.router.uniroute", None),
    "load_router": ("lunar_router.loader", None),
    "create_router": ("lunar_router.loader", None),
    "load_router_from_state": ("lunar_router.loader", None),
    # storage / data
    "StateManager": ("lunar_router.storage.state_manager", None),
    "PromptDataset": ("lunar_router.data.dataset", None),
    "PromptSample": ("lunar_router.data.dataset", None),
    # weights
    "download_weights": ("lunar_router.weights", None),
    "download_from_url": ("lunar_router.weights", None),
    "download_from_s3": ("lunar_router.weights", None),
    "download_from_huggingface": ("lunar_router.weights", None),
    "get_weights_path": ("lunar_router.weights", None),
    "list_available_weights": ("lunar_router.weights", None),
    "WeightsConfig": ("lunar_router.weights", None),
    "WEIGHTS_REGISTRY": ("lunar_router.weights", None),
    # augmentation / feedback
    "LLMJudge": ("lunar_router.augmentation", None),
    "JudgeVerdict": ("lunar_router.augmentation", None),
    "PreferencePair": ("lunar_router.augmentation", None),
    "PreferenceDataset": ("lunar_router.augmentation", None),
    "GoldenAugmenter": ("lunar_router.augmentation", None),
    "AugmentedSample": ("lunar_router.augmentation", None),
    "TraceToTraining": ("lunar_router.feedback", None),
    "ProductionPsiUpdate": ("lunar_router.feedback", None),
    "DriftDetector": ("lunar_router.feedback", None),
    "DriftReport": ("lunar_router.feedback", None),
    "IncrementalUpdater": ("lunar_router.feedback", None),
    "UpdateResult": ("lunar_router.feedback", None),
    # research evaluation
    "RouterEvaluator": ("lunar_router.evaluation", None),
    "EvaluationResult": ("lunar_router.evaluation", None),
    "ParetoPoint": ("lunar_router.evaluation", None),
    "ResponseCache": ("lunar_router.evaluation", None),
    "CachedResponse": ("lunar_router.evaluation", None),
    "compute_auroc": ("lunar_router.evaluation", None),
    "compute_apgr": ("lunar_router.evaluation", None),
    "compute_cpt": ("lunar_router.evaluation", None),
    "compute_pgr_at_savings": ("lunar_router.evaluation", None),
    "compute_win_rate": ("lunar_router.evaluation", None),
    "RoutingMetrics": ("lunar_router.evaluation", None),
    "RandomBaseline": ("lunar_router.evaluation", None),
    "OracleBaseline": ("lunar_router.evaluation", None),
    "AlwaysStrongBaseline": ("lunar_router.evaluation", None),
    "AlwaysWeakBaseline": ("lunar_router.evaluation", None),
    # training (sklearn-heavy)
    "KMeansTrainer": ("lunar_router.training.kmeans_trainer", None),
    "analyze_clusters": ("lunar_router.training.kmeans_trainer", None),
    "LearnedMapTrainer": ("lunar_router.training.learned_map_trainer", None),
    "ModelProfiler": ("lunar_router.profiler.model_profiler", None),
    "full_training_pipeline": ("lunar_router.training.pipeline", None),
    "train_clusters": ("lunar_router.training.pipeline", None),
    "profile_models": ("lunar_router.training.pipeline", None),
    "export_weights": ("lunar_router.training.pipeline", None),
    "quick_train": ("lunar_router.training.pipeline", None),
    "TrainingConfig": ("lunar_router.training.pipeline", None),
    "TrainingResult": ("lunar_router.training.pipeline", None),
    "AutoTrainer": ("lunar_router.training.auto_trainer", None),
    "AutoTrainConfig": ("lunar_router.training.auto_trainer", None),
    "AutoTrainResult": ("lunar_router.training.auto_trainer", None),
    # hub (download manager)
    "download": ("lunar_router.hub", None),
    "list_packages": ("lunar_router.hub", None),
    "package_info": ("lunar_router.hub", "info"),
    "remove": ("lunar_router.hub", None),
    "path": ("lunar_router.hub", None),
    "Hub": ("lunar_router.hub", None),
    "LUNAR_DATA_HOME": ("lunar_router.hub", None),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access for legacy / research-surface names."""
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module 'lunar_router' has no attribute {name!r}")
    module_path, attr = entry
    import importlib
    mod = importlib.import_module(module_path)
    value = getattr(mod, attr or name)
    globals()[name] = value  # cache so subsequent lookups skip the import dance
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY.keys()))
