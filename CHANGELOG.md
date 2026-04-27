# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Azure OpenAI provider — Python SDK.** `ot.completion(model="azure/<deployment>", ...)`
  and `ot.distill(teacher="azure/<deployment>", ...)` now route natively
  through `openai.AzureOpenAI` / `openai.AsyncAzureOpenAI`. The endpoint is
  read from `AZURE_OPENAI_ENDPOINT` (per-resource), the key from
  `AZURE_OPENAI_API_KEY`, and the API version from
  `AZURE_OPENAI_API_VERSION` (default `2024-10-21`). On Azure the model
  string after `azure/` is the *deployment name* (whatever the user named
  it in their resource), not the underlying model.
- **Azure OpenAI provider — Go engine.** `go/internal/provider/azure.go`
  implements `AzureProvider`: builds the per-deployment URL
  (`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}`),
  uses the `api-key` header instead of `Authorization: Bearer`, and reuses
  `OpenAIProvider.buildBody` since Azure accepts the same request schema.
  Registered in `DefaultProviders()` and dispatched via `Format: "azure"`
  in the registry. This is what makes `ot.distill` work end-to-end with
  Azure (the teacher calls during distillation flow through the Go engine).
- **Azure OpenAI in the integrations UI.** New entry in
  `ui/src/constants/integrationModels.ts` so users can register an Azure
  resource alongside the other providers (icon: `Azure` from
  `@lobehub/icons`).

## [0.3.2] - 2026-04-22

### Fixed

- **`ot.load_router()` no longer hits HuggingFace for the default pack.**
  `opentracy/weights.py` now resolves the wheel-bundled
  `weights-mmlu-v1/` directory before falling back to the hub, closing the
  `401 Unauthorized` / `Repository Not Found` loop users hit when the
  private `diogovieira/opentracy-weights` repo wasn't reachable.
- **`GoBackedRouter.registry` is no longer missing.** Added a
  `_GoRegistryView` shim over the engine's `/v1/models` endpoint so
  snippets written for the Python backend (e.g.
  `router.registry.get_model_ids()`) work unchanged on the Go default.
- **`GoEngine.url` alias** added — notebooks were calling
  `engine.url` but the class only exposed `base_url`, 500'ing at
  notebook-02's inline-engine cell.
- **`ot.distill()` works inside Jupyter.** `_run_coro_sync()` helper spins
  a worker thread with a fresh event loop when called from a running loop
  (IPython kernel) and uses `asyncio.run()` otherwise. The old direct
  `asyncio.run(...)` call inside `distill()` raised
  `RuntimeError: asyncio.run() cannot be called from a running event loop`
  in any notebook.
- **`_InMemoryRepo` covers the pipeline's full surface** — `insert_candidates`,
  `insert_candidate` (upsert), `get_candidates`, `record_training_metric`,
  `get_latest_metric`, `list_metrics`, `list_candidates`. The shim used to
  be no-op-only; curation read back zero candidates and phase 2 crashed.
- **`trainer.py` / `export.py` `env` shadowing fixed.** A local
  `env = {...}` dict (subprocess env) hid the module-level
  `env()` lookup, raising
  `UnboundLocalError: cannot access local variable 'env'` before the
  training subprocess ever started. Renamed to `proc_env`.
- **`DATA_DIR` is resolved at call time, not import time** in `trainer.py`,
  `curation.py`, and `export.py`. Previously `ot.distill()` overrode
  `OPENTRACY_DATA_DIR` via a context manager *after* these modules were
  imported, so `DATA_DIR` stayed frozen at its default `Path("data")`
  and the training subprocess tried to read
  `data/distillation/<job>/train_config.json` that didn't exist.
- **Graceful GGUF-export fallback.** If phase 4 (`llama.cpp` conversion)
  fails but the adapter was trained successfully, `ot.distill()` returns
  a PEFT-backed `Student` instead of raising `DistillError`. Adapter
  lives on disk at `{out_dir}/distillation/{job_id}/adapter/` and is
  recorded in `artifacts["adapter_path"]` before export is attempted.
- **`Student._load_peft()` preflight for `jinja2>=3.1`.** A stale
  `jinja2==3.0.x` in the active interpreter (common when a `uvicorn` on
  PATH picks up a different Python than the one that has `opentracy`
  installed) used to surface as an unreadable stacktrace deep inside
  `transformers.apply_chat_template`. Now raises a `StudentError` with
  the exact `pip install` command for the correct interpreter.

### Added

- **`ot.distill()` preflight** — verifies `torch` + CUDA visibility
  before running data-generation / curation, so a misconfigured env
  fails fast instead of burning teacher / judge API spend and then
  dying in phase 3.
- **`torch>=2.1.0` pinned explicitly in `[distill]` extras** (was a
  transitive dep via unsloth/trl/peft). Silent install failures in any
  of those used to leave users with a half-installed env; now
  `pip install opentracy[distill]` fails loudly if torch can't install.

### Changed

- **Notebook `03_semantic_routing.ipynb` cell-4**: engine URL example
  corrected from `:3000` (dashboard nginx, returns 405 on POST
  `/v1/chat/completions`) to `:8080` (engine API), with an explicit
  comment about the distinction.
- **Notebooks 01 / 02 / 06** — consistent `:8080` vs `:3000`
  explanatory comments; `01_quickstart`'s
  `OPENTRACY_ENGINE_URL=http://localhost:8080` re-commented to match
  the notebook's "no server, no Docker" narrative.
- **Notebook `05_distillation.ipynb` rewritten** — replaces the old
  `Distiller` HTTP-client flow (which called non-existent methods like
  `upload_dataset()` and `register_model()` and spawned `uvicorn
  opentracy.api.server:app`) with the one-call `ot.distill()` + 
  `student.deploy()` path. ~19 cells vs. the previous 26. Progress
  callback emits phase transitions + log lines.
- **Notebook `06_distilled_inference.ipynb` rewritten** around
  `opentracy.Student`, `ot.set_alias` / `student.deploy`,
  `ot.completion(model=alias)` dispatch. Removed references to the
  non-existent `OpenTracy/demo-ticket-classifier-v1` HF repo. Added
  a **serve-via-FastAPI** section that writes a minimal
  OpenAI-compatible endpoint to `serve.py` and prints a copy-paste-safe
  launch + curl command (single-line, uses `sys.executable -m uvicorn`
  to avoid Python-version drift).

## [0.3.1] - 2026-03-XX

### Added

- **Default routing weights bundled in the wheel** (`weights-mmlu-v1`, 288 KB,
  100 clusters, 10 profiled models). `pip install opentracy` now carries the
  weights directly — `engine.start()` and `ot.load_router()` work offline on
  first run, no HuggingFace download, no API keys, no auth. `_find_weights()`
  checks `opentracy/_bundled_weights/weights-mmlu-v1/` before falling back to
  user data dirs / hub download, so a pre-installed newer weights pack still
  wins over the bundled default.

### Changed

- **`huggingface_hub` is now a core dependency.** Previously lived in the
  `[research]` extra, which meant `pip install opentracy` → `ot.load_router()`
  failed with `ModuleNotFoundError` the first time it reached for weights.
  Moving it to `dependencies` makes the default semantic-routing path work
  out of the box (the hub code still carries a stdlib `urllib` fallback as
  belt-and-suspenders for stripped-down installs).
- **Engine always emits `X-OpenTracy-Selected-Model` + `X-OpenTracy-Routing-Ms`
  response headers** on `/v1/chat/completions`, not only when `model="auto"`
  routes the request. Explicit `model="openai/gpt-4o-mini"` calls now also
  expose the effective model (useful for the SDK / notebook inspection flow);
  `X-OpenTracy-Routing-Ms` is `0.00` when no routing happened. `Cluster-ID` and
  `Expected-Error` stay scoped to actual routing decisions.
- **`engine.start()` auto-downloads weights on first run.** Previously, calling
  `GoEngine().start()` with no local weights raised `FileNotFoundError` and
  asked the user to run `opentracy download weights-mmlu-v1` manually. Now the
  default pack is fetched transparently from the hub on first start, so
  `pip install opentracy` → notebook works end-to-end with no extra step. Opt
  out with `OPENTRACY_NO_AUTO_DOWNLOAD=1` for CI / offline environments that
  pre-stage weights.

### Removed

- **`lunar_router` backwards-compat shim**: the PEP 451 `MetaPathFinder` that
  redirected `lunar_router.*` imports to `opentracy.*` is gone. Users still
  pinned to the old name must update their imports to `opentracy` — there is
  no longer a transparent alias. The Docker image, wheel, and CI no longer
  ship or test the shim, and `clickhouse/init.sql` creates only the
  `opentracy` database (pre-rebrand deployments still needing the
  `lunar_router` database should follow the 0.3.0 migration SQL below before
  upgrading).

## [0.3.0] — Rebrand: `lunar_router` → `opentracy`

### Changed

- **Package renamed** from `lunar-router` to `opentracy` on PyPI. Python
  import root is now `opentracy` (`import opentracy as ot`).
- **CLI entrypoint** renamed from `lunar-router` to `opentracy`.
- **Go binary** renamed from `lunar-engine` to `opentracy-engine`. Module path
  moved to `github.com/OpenTracy/opentracy/go`.
- **Environment variables** migrated to the `OPENTRACY_*` prefix (e.g.
  `OPENTRACY_ENGINE_URL`, `OPENTRACY_CH_DATABASE`). Legacy `LUNAR_*` vars are
  still read with a one-time `DeprecationWarning` — existing `.env` files keep
  working without changes.
- **ClickHouse database** default renamed from `lunar_router` to `opentracy`.
- **Secrets directory** moved from `~/.lunar/` to `~/.opentracy/`. The CLI
  auto-copies the directory on first run if the new path does not yet exist.
- **Docker services** renamed: `lunar-engine` → `opentracy-engine`,
  `lunar-api` → `opentracy-api`, `lunar-ui` → `opentracy-ui`.
- **HTTP headers** on the engine: `X-Lunar-*` → `X-OpenTracy-*` (the Python
  SDK and Go engine are shipped together in the wheel, so clients never see
  mixed versions).
- **MCP tool names**: `lunar_route` / `lunar_generate` / `lunar_smart_generate`
  / `lunar_list_models` / `lunar_compare` → `opentracy_*` equivalents.

### Added

- **Backwards-compat shim**: installing `opentracy` also installs a tiny
  `lunar_router` package whose `__init__.py` emits a `DeprecationWarning`
  and transparently redirects all `lunar_router.*` imports to `opentracy.*`
  via a `MetaPathFinder`. User code still doing `import lunar_router` keeps
  working unchanged.
- `opentracy._env.env()` helper: reads `OPENTRACY_<NAME>` first, falls back
  to `LUNAR_<NAME>` with a deprecation warning. Go equivalent lives at
  `internal/envfallback.Get()`.

### Migration notes

- **ClickHouse data**: existing deployments have live data in a `lunar_router`
  database. The included `clickhouse/init.sql` now creates *both* `opentracy`
  and `lunar_router` on first start so old data is not lost. For a fresh
  container against an existing volume, run this migration exactly once — it
  renames the database and rewires the two materialized views (their DDL
  hardcodes the source/target database name, which `RENAME DATABASE` does not
  rewrite, so the MVs must be dropped and recreated):

  ```sql
  RENAME DATABASE lunar_router TO opentracy;

  DROP VIEW IF EXISTS opentracy.mv_cluster_daily;
  DROP VIEW IF EXISTS opentracy.mv_model_hourly;

  CREATE MATERIALIZED VIEW opentracy.mv_cluster_daily
  TO opentracy.cluster_daily_stats AS
  SELECT toDate(timestamp) AS day, cluster_id,
         countState(is_error) AS request_count,
         sumState(is_error) AS error_count,
         quantilesState(0.5, 0.95)(latency_ms) AS latency_quantiles,
         sumState(total_tokens) AS total_tokens,
         sumState(total_cost_usd) AS total_cost_usd,
         uniqState(selected_model) AS unique_models
  FROM opentracy.llm_traces GROUP BY day, cluster_id;

  CREATE MATERIALIZED VIEW opentracy.mv_model_hourly
  TO opentracy.model_hourly_stats AS
  SELECT toStartOfHour(timestamp) AS hour, selected_model,
         countState(is_error) AS request_count,
         sumState(is_error) AS error_count,
         quantilesState(0.5, 0.95, 0.99)(latency_ms) AS latency_quantiles,
         quantilesState(0.5, 0.95)(ttft_ms) AS ttft_quantiles,
         sumState(tokens_in) AS total_tokens_in,
         sumState(tokens_out) AS total_tokens_out,
         sumState(total_cost_usd) AS total_cost_usd,
         uniqState(provider) AS unique_providers,
         uniqState(cluster_id) AS unique_clusters
  FROM opentracy.llm_traces GROUP BY hour, selected_model;
  ```

  After running, restart `opentracy-engine` and `opentracy-api` so they pick
  up the renamed database.
- **PyPI name**: first release under the new name must reserve `opentracy` on
  PyPI. Publishing a final `lunar-router` version that re-exports from
  `opentracy` is recommended to ease the transition for downstream pinning.

## [0.1.0] - 2025-01-29

### Added

- **Core routing engine** based on the UniRoute algorithm from
  [Universal Model Routing for Efficient LLM Inference](https://arxiv.org/abs/2502.08773)
- **7 LLM provider clients**: OpenAI, Anthropic, Google Gemini, Groq, Mistral, vLLM, and Mock
- **44+ pre-configured models** with pricing information
- **K-Means semantic routing**: cluster-based prompt embedding with SentenceTransformers
- **Cost-quality trade-off**: adjustable `cost_weight` parameter (0 = quality, 1 = cost)
- **Hub system**: download and manage pre-trained weights (inspired by NLTK/spaCy/HuggingFace)
  - CLI: `opentracy download`, `list`, `info`, `remove`, `path`, `verify`
  - Python API: `opentracy.download()`, `list_packages()`, `package_info()`
- **Pre-trained weights** on MMLU benchmark (`weights-mmlu-v1`) hosted on HuggingFace Hub
- **Training pipeline**: train custom routers with `KMeansTrainer` and `full_training_pipeline()`
- **OpenAI-compatible API server** with health-first routing and fallback support
- **Weights management module** (`opentracy.weights`) for downloading from HuggingFace, URL, and S3
- **State management** for persisting router configurations and profiles

### Fixed

- **Cache eviction bug**: `PromptEmbedder` cache eviction failed silently when `cache_max_size < 10`
  because `max_size // 10` evaluated to 0, causing unbounded cache growth. Now evicts at least 1 entry.

### Changed

- **Unified branding**: standardized references from "PureAI" to "OpenTracy" in documentation,
  docstrings, API titles, and comments across the codebase

[0.1.0]: https://github.com/pureai-ecosystem/opentracy/releases/tag/v0.1.0
