# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-29

### Added

- **Core routing engine** based on the UniRoute algorithm from
  [Universal Model Routing for Efficient LLM Inference](https://arxiv.org/abs/2502.08773)
- **7 LLM provider clients**: OpenAI, Anthropic, Google Gemini, Groq, Mistral, vLLM, and Mock
- **44+ pre-configured models** with pricing information
- **K-Means semantic routing**: cluster-based prompt embedding with SentenceTransformers
- **Cost-quality trade-off**: adjustable `cost_weight` parameter (0 = quality, 1 = cost)
- **Hub system**: download and manage pre-trained weights (inspired by NLTK/spaCy/HuggingFace)
  - CLI: `lunar-router download`, `list`, `info`, `remove`, `path`, `verify`
  - Python API: `lunar_router.download()`, `list_packages()`, `package_info()`
- **Pre-trained weights** on MMLU benchmark (`weights-mmlu-v1`) hosted on HuggingFace Hub
- **Training pipeline**: train custom routers with `KMeansTrainer` and `full_training_pipeline()`
- **OpenAI-compatible API server** with health-first routing and fallback support
- **Weights management module** (`lunar_router.weights`) for downloading from HuggingFace, URL, and S3
- **State management** for persisting router configurations and profiles

### Fixed

- **Cache eviction bug**: `PromptEmbedder` cache eviction failed silently when `cache_max_size < 10`
  because `max_size // 10` evaluated to 0, causing unbounded cache growth. Now evicts at least 1 entry.

### Changed

- **Unified branding**: standardized references from "PureAI" to "Lunar Router" in documentation,
  docstrings, API titles, and comments across the codebase

[0.1.0]: https://github.com/pureai-ecosystem/lunar-router/releases/tag/v0.1.0
