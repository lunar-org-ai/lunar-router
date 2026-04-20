"""OpenTracy: the auto-distillation layer for your LLM calls.

Drop-in OpenAI-compatible SDK. Every request becomes a trace; traces become
datasets; datasets become distilled custom models; the routing layer swaps
those models in under your app via aliases — so your cost curve goes down
over time without code changes.

Quick start:
    >>> import opentracy as lr
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
    # All 13 providers routed through the OpenTracy engine; every request is a trace.

Distillation — what makes OpenTracy different from a plain LLM gateway:
    >>> from opentracy import Distiller
    >>> d = Distiller()
    # Submit a dataset, pick teacher + student, and OpenTracy trains the distilled
    # model and serves it behind a routing alias you can point traffic at.

Providers (13):
    OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, Together, Fireworks,
    Cerebras, Sambanova, Perplexity, Cohere, Bedrock.
"""

__version__ = "0.3.0"

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
# Research surface — lazy. Still importable as `from opentracy import X`
# for backward compatibility, but not advertised in __all__, autocomplete,
# or tab-completion. Loading sentence-transformers / sklearn / sub-frameworks
# only happens when one of these names is actually touched.
#
# Power users should prefer the explicit submodule path
# (e.g. `from opentracy.evaluation import RouterEvaluator`) — these
# shims exist so existing notebooks and docs don't break.
# ---------------------------------------------------------------------------
_LAZY: dict[str, tuple[str, str | None]] = {
    # student (local inference on trained adapters / GGUF)
    "Student": ("opentracy.student", None),
    "load_student": ("opentracy.student", None),
    "StudentError": ("opentracy.student", None),
    # distill (one-call training → Student)
    "distill": ("opentracy._distill", None),
    "DistillError": ("opentracy._distill", None),
    # aliases (file-based registry powering the alias swap)
    "list_aliases": ("opentracy.aliases", None),
    "get_alias": ("opentracy.aliases", None),
    "set_alias": ("opentracy.aliases", None),
    "unset_alias": ("opentracy.aliases", None),
    "AliasError": ("opentracy.aliases", None),
    # core
    "PromptEmbedder": ("opentracy.core.embeddings", None),
    "SentenceTransformerProvider": ("opentracy.core.embeddings", None),
    "EmbeddingProvider": ("opentracy.core.embeddings", None),
    "ClusterAssigner": ("opentracy.core.clustering", None),
    "ClusterResult": ("opentracy.core.clustering", None),
    "KMeansClusterAssigner": ("opentracy.core.clustering", None),
    "LearnedMapClusterAssigner": ("opentracy.core.clustering", None),
    "load_cluster_assigner": ("opentracy.core.clustering", None),
    # models
    "LLMProfile": ("opentracy.models.llm_profile", None),
    "LLMRegistry": ("opentracy.models.llm_registry", None),
    "LLMClient": ("opentracy.models.llm_client", None),
    "OpenAIClient": ("opentracy.models.llm_client", None),
    "AnthropicClient": ("opentracy.models.llm_client", None),
    "MistralClient": ("opentracy.models.llm_client", None),
    "GoogleClient": ("opentracy.models.llm_client", None),
    "GroqClient": ("opentracy.models.llm_client", None),
    "VLLMClient": ("opentracy.models.llm_client", None),
    "MockLLMClient": ("opentracy.models.llm_client", None),
    "UnifiedClient": ("opentracy.models.llm_client", None),
    # research router (paper impl)
    "UniRouteRouter": ("opentracy.router.uniroute", None),
    "RoutingDecision": ("opentracy.router.uniroute", None),
    "RoutingStats": ("opentracy.router.uniroute", None),
    "load_router": ("opentracy.loader", None),
    "create_router": ("opentracy.loader", None),
    "load_router_from_state": ("opentracy.loader", None),
    # storage / data
    "StateManager": ("opentracy.storage.state_manager", None),
    "PromptDataset": ("opentracy.data.dataset", None),
    "PromptSample": ("opentracy.data.dataset", None),
    # weights
    "download_weights": ("opentracy.weights", None),
    "download_from_url": ("opentracy.weights", None),
    "download_from_s3": ("opentracy.weights", None),
    "download_from_huggingface": ("opentracy.weights", None),
    "get_weights_path": ("opentracy.weights", None),
    "list_available_weights": ("opentracy.weights", None),
    "WeightsConfig": ("opentracy.weights", None),
    "WEIGHTS_REGISTRY": ("opentracy.weights", None),
    # augmentation / feedback
    "LLMJudge": ("opentracy.augmentation", None),
    "JudgeVerdict": ("opentracy.augmentation", None),
    "PreferencePair": ("opentracy.augmentation", None),
    "PreferenceDataset": ("opentracy.augmentation", None),
    "GoldenAugmenter": ("opentracy.augmentation", None),
    "AugmentedSample": ("opentracy.augmentation", None),
    "TraceToTraining": ("opentracy.feedback", None),
    "ProductionPsiUpdate": ("opentracy.feedback", None),
    "DriftDetector": ("opentracy.feedback", None),
    "DriftReport": ("opentracy.feedback", None),
    "IncrementalUpdater": ("opentracy.feedback", None),
    "UpdateResult": ("opentracy.feedback", None),
    # research evaluation
    "RouterEvaluator": ("opentracy.evaluation", None),
    "EvaluationResult": ("opentracy.evaluation", None),
    "ParetoPoint": ("opentracy.evaluation", None),
    "ResponseCache": ("opentracy.evaluation", None),
    "CachedResponse": ("opentracy.evaluation", None),
    "compute_auroc": ("opentracy.evaluation", None),
    "compute_apgr": ("opentracy.evaluation", None),
    "compute_cpt": ("opentracy.evaluation", None),
    "compute_pgr_at_savings": ("opentracy.evaluation", None),
    "compute_win_rate": ("opentracy.evaluation", None),
    "RoutingMetrics": ("opentracy.evaluation", None),
    "RandomBaseline": ("opentracy.evaluation", None),
    "OracleBaseline": ("opentracy.evaluation", None),
    "AlwaysStrongBaseline": ("opentracy.evaluation", None),
    "AlwaysWeakBaseline": ("opentracy.evaluation", None),
    # training (sklearn-heavy)
    "KMeansTrainer": ("opentracy.training.kmeans_trainer", None),
    "analyze_clusters": ("opentracy.training.kmeans_trainer", None),
    "LearnedMapTrainer": ("opentracy.training.learned_map_trainer", None),
    "ModelProfiler": ("opentracy.profiler.model_profiler", None),
    "full_training_pipeline": ("opentracy.training.pipeline", None),
    "train_clusters": ("opentracy.training.pipeline", None),
    "profile_models": ("opentracy.training.pipeline", None),
    "export_weights": ("opentracy.training.pipeline", None),
    "quick_train": ("opentracy.training.pipeline", None),
    "TrainingConfig": ("opentracy.training.pipeline", None),
    "TrainingResult": ("opentracy.training.pipeline", None),
    "AutoTrainer": ("opentracy.training.auto_trainer", None),
    "AutoTrainConfig": ("opentracy.training.auto_trainer", None),
    "AutoTrainResult": ("opentracy.training.auto_trainer", None),
    # hub (download manager)
    "download": ("opentracy.hub", None),
    "list_packages": ("opentracy.hub", None),
    "package_info": ("opentracy.hub", "info"),
    "remove": ("opentracy.hub", None),
    "path": ("opentracy.hub", None),
    "Hub": ("opentracy.hub", None),
    "OPENTRACY_DATA_HOME": ("opentracy.hub", None),
    # Legacy alias kept for backwards compatibility with pre-rebrand code that
    # imported `LUNAR_DATA_HOME` from the top-level package.
    "LUNAR_DATA_HOME": ("opentracy.hub", "OPENTRACY_DATA_HOME"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access for legacy / research-surface names."""
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module 'opentracy' has no attribute {name!r}")
    module_path, attr = entry
    import importlib
    mod = importlib.import_module(module_path)
    value = getattr(mod, attr or name)
    globals()[name] = value  # cache so subsequent lookups skip the import dance
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY.keys()))
