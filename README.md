<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/opentracy-logo-dark.png">
    <img src="assets/opentracy-logo-light.png" alt="OpenTracy" height="110">
  </picture>
</p>

<p align="center"><strong>The auto-distillation layer for your LLM calls.</strong></p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pypi.org/project/opentracy/"><img src="https://img.shields.io/pypi/v/opentracy.svg" alt="PyPI"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/01_quickstart.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28">
  </a>
  &nbsp;
  <a href="https://discord.gg/a8G5pQEN">
    <img src="https://img.shields.io/badge/Join%20our%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord" height="28">
  </a>
  &nbsp;
  <a href="https://opentracy.mintlify.app">
    <img src="https://img.shields.io/badge/Documentation-10B981?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation" height="28">
  </a>
</p>

Drop-in OpenAI-compatible SDK. Every request becomes a trace; traces become datasets; datasets become distilled custom models; the routing layer swaps those models in under your app via aliases — so your cost curve goes down over time **without code changes**.

## Try it in Colab (no install)

Each notebook runs end-to-end on a free Colab runtime — bring your own OpenAI key, optionally Anthropic / Groq.

| # | Notebook | One-line pitch | Colab |
|---|---|---|---|
| 01 | [Quickstart](notebooks/01_quickstart.ipynb) | First `completion()` call, see `_cost` + `_latency_ms`, swap providers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/01_quickstart.ipynb) |
| 02 | [Drop in over the OpenAI SDK](notebooks/02_drop_in_openai.ipynb) | Keep `from openai import OpenAI`, change only `base_url` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/02_drop_in_openai.ipynb) |
| 03 | [Semantic auto-routing](notebooks/03_semantic_routing.ipynb) | One prompt, the right model of 13 — learned, not rule-based | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/03_semantic_routing.ipynb) |
| 04 | [Ticket classifier (real app)](notebooks/04_ticket_classifier.ipynb) | End-to-end support-ticket classifier with cost breakdown | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/04_ticket_classifier.ipynb) |
| 05 | [Distillation — train your student](notebooks/05_distillation.ipynb) | Turn trace history into a distilled tiny model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/05_distillation.ipynb) |
| 06 | [Serve your distilled model](notebooks/06_distilled_inference.ipynb) | Four serving paths from load-the-adapter to alias swap | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenTracy/OpenTracy/blob/main/notebooks/06_distilled_inference.ipynb) |

> **Colab heads-up** — traces only show up in the dashboard if you set `OPENTRACY_ENGINE_URL` before `import opentracy`. Every notebook has a commented-out cell at the top with the two lines you need.

## Install

```bash
pip install opentracy
```

## Quick start

```python
import opentracy as lr

resp = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
print(f"cost: ${resp._cost:.6f}  latency: {resp._latency_ms:.0f}ms")
```

Works with 13 providers out of the box: OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, Together, Fireworks, Cerebras, Sambanova, Perplexity, Cohere, Bedrock.

### Connecting to the OpenTracy platform (traces, dashboards, distillation)

By default `lr.completion()` goes **direct to the provider**, so calls do *not*
appear in the OpenTracy dashboard. To route every call through a running
engine — the only way traces, metrics, and the distillation loop get data —
set `OPENTRACY_ENGINE_URL` **before** importing the SDK:

```python
import os
os.environ["OPENTRACY_ENGINE_URL"] = "http://<your-opentracy-host>:8080"  # engine port
import opentracy as lr

resp = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
# trace now visible in the dashboard on the UI host at :3000
```

Alternatives:
- Per-call: pass `force_engine=True, api_base="http://<host>:8080/v1"` to
  `lr.completion(...)`.
- Drop-in OpenAI SDK (no code change beyond `base_url` — see below).

API keys for your providers should be saved once via the UI
(**Settings → API Keys**) or the API (`POST /v1/secrets/<provider>`); the
engine picks them up immediately from `~/.opentracy/secrets.json`.

## Routing with fallbacks

```python
router = lr.Router(
    model_list=[
        {"model_name": "smart", "model": "openai/gpt-4o"},
        {"model_name": "smart", "model": "anthropic/claude-sonnet-4-6"},
    ],
    fallbacks=[{"smart": ["deepseek/deepseek-chat"]}],
)
resp = router.completion(model="smart", messages=[{"role": "user", "content": "Hi"}])
```

## Drop-in replacement for the OpenAI SDK

Point any existing OpenAI app at the OpenTracy engine — zero code changes beyond `base_url`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
# All 13 providers routed through the OpenTracy engine; every request is a trace.
```

## Distillation — what makes OpenTracy different from a plain gateway

```python
from opentracy import Distiller

d = Distiller()
# Submit a dataset built from your own traces, pick a teacher + student model,
# and OpenTracy trains the distilled model and serves it behind a routing alias
# you can point traffic at. Your app code never changes.
```

Install the training extras for the distillation pipeline:

```bash
pip install opentracy[distill]
```

## Self-host the full platform (traces + UI + REST API)

```bash
git clone https://github.com/OpenTracy/opentracy.git
cd opentracy
make start-full   # Gateway + ClickHouse analytics + Python API + UI
```

Engine at `http://localhost:8080`, Python API at `http://localhost:8000`, UI at `http://localhost:3000`.

## What OpenTracy Does

```
Requests ──► Gateway (13 providers) ──► Traces (ClickHouse)
                                            │
                                    ┌───────┴───────┐
                                    ▼               ▼
                              Clustering        Analytics
                              (domains)        (cost/latency)
                                    │
                              ┌─────┴─────┐
                              ▼           ▼
                         Evaluations   Distillation
                        (AI metrics)  (training data)
```

1. **Route** — proxy to 13 LLM providers with fallbacks, retries, and cost tracking
2. **Observe** — every request/response stored in ClickHouse with full content
3. **Cluster** — auto-group prompts by domain using embeddings + LLM labeling
4. **Evaluate** — run models against domain datasets with built-in and AI-suggested metrics
5. **Distill** — export input/output pairs per domain for fine-tuning smaller models

## Features

### Gateway

- **13 LLM Providers** through one OpenAI-compatible API
- **Python SDK** — `lr.completion()` one-liner
- **Router Class** — load balancing, fallbacks, retries, 4 strategies
- **Streaming** — all providers including Anthropic & Bedrock SSE translation
- **Cost Tracking** — 70+ models with per-token pricing on every response
- **Vision / Multimodal** — images via base64 or URL
- **Tool Calling** — function calls with cross-provider translation
- **Semantic Routing** — auto-select the best model per prompt (with weights)

### Observability

- **ClickHouse Analytics** — traces, cost, latency, model-level stats
- **Full Content Capture** — input/output text stored for every request
- **Trace Scanning** — AI agent detects hallucinations, refusals, quality regressions
- **Real-time Dashboard** — UI with filters, search, trace detail drawer

### Domain Clustering

- **Auto-clustering** — groups prompts by semantic similarity (KMeans + MiniLM embeddings)
- **LLM Labeling** — AI agent names each cluster (e.g., "JavaScript Concepts", "Business Strategy")
- **Quality Gates** — coherence scoring, outlier detection, merge suggestions
- **Input + Output Storage** — full pairs stored per cluster for distillation

### Evaluations

- **Run Evaluations** — send dataset samples through models, score and compare
- **6 Built-in Metrics** — exact match, contains, similarity, LLM-as-judge, latency, cost
- **AI Metric Suggestion** — harness agent analyzes dataset domain and creates tailored metrics
- **Background Execution** — evaluations run async with progress tracking
- **Model Comparison** — side-by-side results with winner determination

### Distillation

- **BOND Pipeline** — teacher → LLM-as-Judge curation → LoRA training (Unsloth) → GGUF export
- **Dataset Support** — use domain clusters or custom datasets as training source
- **UI + API** — create and monitor jobs via dashboard or REST endpoints

### Harness (AI Agent System)

- **Agent Runner** — loads `.md` agent configs, calls LLM, parses structured output
- **7 Agents** — cluster labeler, coherence scorer, outlier detector, merge checker, trace scanner, eval generator, metrics suggester
- **Memory Layer** — persistent agent memory with query/summary
- **Tool Access** — agents can call tools (list traces, query datasets, etc.)

## Supported Providers

| Provider        | Syntax                                                        | Env Var                                       |
| --------------- | ------------------------------------------------------------- | --------------------------------------------- |
| **OpenAI**      | `openai/gpt-4o-mini`                                          | `OPENAI_API_KEY`                              |
| **Anthropic**   | `anthropic/claude-haiku-4-5-20251001`                         | `ANTHROPIC_API_KEY`                           |
| **Gemini**      | `gemini/gemini-2.0-flash`                                     | `GEMINI_API_KEY`                              |
| **Mistral**     | `mistral/mistral-small-latest`                                | `MISTRAL_API_KEY`                             |
| **Groq**        | `groq/llama-3.3-70b-versatile`                                | `GROQ_API_KEY`                                |
| **DeepSeek**    | `deepseek/deepseek-chat`                                      | `DEEPSEEK_API_KEY`                            |
| **Perplexity**  | `perplexity/sonar`                                            | `PERPLEXITY_API_KEY`                          |
| **Cerebras**    | `cerebras/llama3.1-70b`                                       | `CEREBRAS_API_KEY`                            |
| **SambaNova**   | `sambanova/Meta-Llama-3.1-70B-Instruct`                       | `SAMBANOVA_API_KEY`                           |
| **Together**    | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo`            | `TOGETHER_API_KEY`                            |
| **Fireworks**   | `fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct` | `FIREWORKS_API_KEY`                           |
| **Cohere**      | `cohere/command-r-plus`                                       | `COHERE_API_KEY`                              |
| **AWS Bedrock** | `bedrock/amazon.titan-text-express-v1`                        | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |

## Installation

```bash
pip install -e ".[openai,anthropic,api]"   # SDK + common providers
pip install -e ".[all]"                     # everything
pip install -e ".[train]"                   # training/distillation deps (CUDA)
```

## Python SDK

### Completion

```python
import opentracy as lr

response = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
for chunk in lr.completion(model="openai/gpt-4o-mini", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# Fallbacks
response = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[...],
    fallbacks=["anthropic/claude-haiku-4-5-20251001", "groq/llama-3.3-70b-versatile"],
    num_retries=2,
)
```

### Router (Load Balancing)

```python
router = lr.Router(
    model_list=[
        {"model_name": "smart", "model": "openai/gpt-4o"},
        {"model_name": "smart", "model": "anthropic/claude-sonnet-4-20250514"},
        {"model_name": "fast",  "model": "groq/llama-3.3-70b-versatile"},
    ],
    fallbacks=[{"smart": ["deepseek/deepseek-chat"]}],
    strategy="round-robin",  # or: least-cost, lowest-latency, weighted-random
)

response = router.completion(model="smart", messages=[...])
```

### Drop-in OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
client.chat.completions.create(model="openai/gpt-4o-mini", messages=[...])
client.chat.completions.create(model="anthropic/claude-haiku-4-5-20251001", messages=[...])
client.chat.completions.create(model="mistral/mistral-small-latest", messages=[...])
```

## Running

| Command             | What                                   | Requires     |
| ------------------- | -------------------------------------- | ------------ |
| `make start`        | Gateway proxy (no weights needed)      | Go           |
| `make start-full`   | Gateway + ClickHouse + Python API      | Go + Docker  |
| `make start-router` | Full semantic routing (`model="auto"`) | Go + weights |
| `make dev-python`   | Python API only (uvicorn --reload)     | Python       |

### API Keys

Configure via the UI, environment variables, or `~/.opentracy/secrets.json`:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
make start-full
```

## API Endpoints

### Gateway (Go Engine — port 8080)

| Method | Endpoint               | Description                       |
| ------ | ---------------------- | --------------------------------- |
| `POST` | `/v1/chat/completions` | Chat completion (any provider)    |
| `POST` | `/v1/route`            | Route a prompt without generating |
| `GET`  | `/v1/models`           | List registered models            |
| `GET`  | `/health`              | Health check                      |

### Python API (port 8000)

| Method           | Endpoint                                  | Description                                           |
| ---------------- | ----------------------------------------- | ----------------------------------------------------- |
| **Analytics**    |                                           |                                                       |
| `GET`            | `/v1/stats/{tenant}/analytics`            | Full analytics (traces, cost, latency, distributions) |
| **Clustering**   |                                           |                                                       |
| `POST`           | `/v1/clustering/run`                      | Run clustering pipeline (embed, cluster, label)       |
| `GET`            | `/v1/clustering/datasets`                 | List domain datasets from latest run                  |
| `GET`            | `/v1/clustering/datasets/{run}/{cluster}` | Get traces for a cluster                              |
| **Datasets**     |                                           |                                                       |
| `GET`            | `/v1/datasets`                            | List all datasets (eval + domain clusters)            |
| `POST`           | `/v1/datasets`                            | Create evaluation dataset                             |
| `POST`           | `/v1/datasets/{id}/samples`               | Add samples to dataset                                |
| **Evaluations**  |                                           |                                                       |
| `POST`           | `/v1/evaluations`                         | Create and run evaluation (async)                     |
| `GET`            | `/v1/evaluations`                         | List evaluations                                      |
| `GET`            | `/v1/evaluations/{id}/status`             | Evaluation progress                                   |
| `GET`            | `/v1/evaluations/{id}/results`            | Evaluation results with scores                        |
| **Distillation** |                                           |                                                       |
| `POST`           | `/v1/distillation/{tenant}/jobs`          | Create distillation job                               |
| `GET`            | `/v1/distillation/{tenant}/jobs`          | List distillation jobs                                |
| `GET`            | `/v1/distillation/{tenant}/jobs/{id}`     | Get job status and results                            |
| **Metrics**      |                                           |                                                       |
| `GET`            | `/v1/metrics`                             | List built-in + custom metrics                        |
| `POST`           | `/v1/metrics`                             | Create custom metric                                  |
| `POST`           | `/v1/auto-eval/suggest-metrics`           | AI-powered metric suggestion                          |
| **Models**       |                                           |                                                       |
| `GET`            | `/v1/models/available`                    | Models available from configured providers            |
| **Harness**      |                                           |                                                       |
| `GET`            | `/v1/harness/agents`                      | List AI agents                                        |
| `POST`           | `/v1/harness/run/{name}`                  | Run an agent with input                               |
| `GET`            | `/v1/harness/memory`                      | Query agent memory                                    |
| **Secrets**      |                                           |                                                       |
| `GET`            | `/v1/secrets`                             | List configured providers                             |
| `POST`           | `/v1/secrets/{provider}`                  | Save API key                                         
| `GET` | `/v1/harness/agents` | List AI agents |
| `POST` | `/v1/harness/run/{name}` | Run an agent with input |
| `GET` | `/v1/harness/memory` | Query agent memory |
| **Secrets** | | |
| `GET` | `/v1/secrets` | List configured providers |
| `POST` | `/v1/secrets/{provider}` | Save API key |

## Architecture

```
go/                              # Go engine (high-performance gateway)
├── cmd/opentracy-engine/            # Entry point
├── internal/
│   ├── provider/                # 13 providers
│   ├── server/                  # HTTP handlers + session management
│   ├── clickhouse/              # Trace writer + 8 migrations
│   ├── router/                  # UniRoute algorithm + LRU cache
│   └── embeddings/              # ONNX MiniLM embedder

opentracy/                    # Python layer (analytics, clustering, evals)
├── api/server.py                # FastAPI — analytics, clustering, evaluations, metrics
├── sdk.py                       # completion(), acompletion(), Router class
├── clustering/
│   ├── pipeline.py              # Extract → embed → cluster → label → store
│   ├── labeler.py               # LLM-powered cluster labeling via harness
│   └── quality.py               # Coherence, diversity, noise quality gates
├── harness/
│   ├── runner.py                # Agent executor (JSON parsing, retry, tools)
│   ├── tools.py                 # Agent tools (query traces, datasets, etc.)
│   ├── memory_store.py          # Persistent agent memory
│   └── agents/                  # 7 agent configs (.md files)
│       ├── cluster_labeler.md
│       ├── coherence_scorer.md
│       ├── outlier_detector.md
│       ├── merge_checker.md
│       ├── trace_scanner.md
│       ├── eval_generator.md
│       └── metrics_suggester.md
├── distillation/
│   ├── pipeline.py              # 4-phase orchestrator (data gen → curation → train → export)
│   ├── data_gen.py              # Teacher model candidate generation
│   ├── curation.py              # LLM-as-Judge scoring & selection
│   ├── trainer.py               # SFT/BOND fine-tuning (Unsloth + LoRA)
│   ├── export.py                # LoRA merge + GGUF conversion
│   ├── repository.py            # ClickHouse persistence
│   ├── router.py                # API endpoints
│   └── schemas.py               # Pydantic models & model catalog
├── evaluations/                 # Evaluation runs & results
├── datasets/                    # Dataset CRUD, from-traces, auto-collect
├── metrics/                     # Metric definitions & validation
├── experiments/                 # A/B experiments & comparison
├── annotations/                 # Human annotation queues
├── auto_eval/                   # Automated evaluation configs & triggers
├── eval_agent/                  # AI-powered eval setup assistant
├── proposals/                   # Decision engine proposals
├── trace_issues/                # Issue scanning & detection
├── training/                    # Custom router training (UniRoute)
├── storage/
│   ├── clickhouse_client.py     # Analytics queries
│   ├── secrets.py               # API key management
│   └── state_manager.py         # File-based state persistence
├── model_prices.py              # 70+ models with pricing
└── mcp/                         # Claude Code MCP server

ui/                              # React dashboard
├── src/features/
│   ├── traces/                  # Trace explorer with drawer, filters, timeline
│   ├── evaluations/             # Run evaluations, metrics, experiments
│   └── distill-dataset/         # Dataset management, clustering, export
```

## Evaluation Workflow

```bash
# 1. Send traffic through the gateway (traces auto-captured)
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Explain closures in JS"}]}'

# 2. Run clustering to group prompts by domain
curl -X POST http://localhost:8000/v1/clustering/run?days=30&min_traces=5

# 3. AI suggests metrics for a domain dataset
curl -X POST http://localhost:8000/v1/auto-eval/suggest-metrics \
  -d '{"dataset_id": "cluster:run-id:2"}'

# 4. Run evaluation comparing models on that dataset
curl -X POST http://localhost:8000/v1/evaluations \
  -d '{"name": "JS eval", "dataset_id": "cluster:run-id:2",
       "models": ["openai/gpt-4o-mini", "mistral/mistral-small-latest"],
       "metrics": ["similarity", "latency", "cost", "llm_judge"]}'

# 5. Check results
curl http://localhost:8000/v1/evaluations/{id}/results
```

## Distillation

BOND-style distillation pipeline: generate candidates with a teacher model, score them with LLM-as-Judge, fine-tune a student model with LoRA, and export to GGUF.

```bash
make install-train   # install training deps (requires CUDA)
```

Via UI at `http://localhost:3000` → Distillation, or via API:

```bash
curl -X POST http://localhost:8000/v1/distillation/default/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "teacher_model": "openai/gpt-4o-mini",
    "student_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "num_candidates": 5,
    "dataset_id": "my-dataset"
  }'
```

## Semantic Routing

With pre-trained weights, the router picks the best model per prompt:

```bash
make download-weights   # download from HuggingFace
make start-router       # start with semantic routing enabled
```

```python
router = load_router()
decision = router.route("Explain quantum computing")
print(f"Best model: {decision.selected_model}")
print(f"Expected error: {decision.expected_error:.4f}")
```

### Training Custom Routers

```python
from opentracy import full_training_pipeline, TrainingConfig, PromptDataset, create_client

train_data = PromptDataset.load("train.json")
val_data = PromptDataset.load("val.json")

clients = [
    create_client("openai", "gpt-4o"),
    create_client("openai", "gpt-4o-mini"),
    create_client("groq", "llama-3.1-8b-instant"),
]

result = full_training_pipeline(
    train_data, val_data, clients,
    TrainingConfig(num_clusters=100, output_dir="./weights"),
)
```

## MCP Integration (Claude Code)

```bash
pip install opentracy[mcp]
```

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "opentracy": {
      "command": "python",
      "args": ["-m", "opentracy.mcp"]
    }
  }
}
```

Tools: `opentracy_route`, `opentracy_generate`, `opentracy_smart_generate`, `opentracy_list_models`, `opentracy_compare`.

## Development

```bash
make help               # show all commands
make install            # install Python SDK + Go deps
make install-all        # install everything (Python + Go + UI)
make install-train      # install training/distillation deps (CUDA)
make dev-all            # start full local stack (ClickHouse + Go + API + UI)
make stop-all           # stop all local services
make test               # run all tests
make lint               # lint all code
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [GitHub Repository](https://github.com/OpenTracy/opentracy)
- [HuggingFace Weights](https://huggingface.co/diogovieira/opentracy-weights)
