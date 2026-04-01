# Lunar Router

**Unified LLM Gateway + Evaluation Engine — route, observe, evaluate, distill.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/diogovieira/lunar-router-weights)

Lunar Router is an LLM infrastructure platform. It routes requests to **13 providers** through a single OpenAI-compatible API, collects traces with full input/output content, clusters prompts into domain datasets, runs evaluations with AI-suggested metrics, and prepares training data for model distillation.

## Quick Start

```bash
git clone https://github.com/lunar-org-ai/lunar-router.git
cd lunar-router
make start-full   # Gateway + ClickHouse analytics + Python API
```

Engine at `http://localhost:8080`, Python API at `http://localhost:8000`, UI at `http://localhost:3000`.

```python
import lunar_router as lr

response = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
print(f"Cost: ${response._cost:.6f}")
```

## What Lunar Router Does

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

### Harness (AI Agent System)
- **Agent Runner** — loads `.md` agent configs, calls LLM, parses structured output
- **7 Agents** — cluster labeler, coherence scorer, outlier detector, merge checker, trace scanner, eval generator, metrics suggester
- **Memory Layer** — persistent agent memory with query/summary
- **Tool Access** — agents can call tools (list traces, query datasets, etc.)

## Supported Providers

| Provider | Syntax | Env Var |
|----------|--------|---------|
| **OpenAI** | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic/claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| **Gemini** | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| **Mistral** | `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| **Groq** | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **DeepSeek** | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| **Perplexity** | `perplexity/sonar` | `PERPLEXITY_API_KEY` |
| **Cerebras** | `cerebras/llama3.1-70b` | `CEREBRAS_API_KEY` |
| **SambaNova** | `sambanova/Meta-Llama-3.1-70B-Instruct` | `SAMBANOVA_API_KEY` |
| **Together** | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| **Fireworks** | `fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct` | `FIREWORKS_API_KEY` |
| **Cohere** | `cohere/command-r-plus` | `COHERE_API_KEY` |
| **AWS Bedrock** | `bedrock/amazon.titan-text-express-v1` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |

## Installation

```bash
pip install -e ".[openai,anthropic,api]"   # SDK + common providers
pip install -e ".[all]"                     # everything
```

## Python SDK

### Completion

```python
import lunar_router as lr

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

| Command | What | Requires |
|---------|------|----------|
| `make start` | Gateway proxy (no weights needed) | Go |
| `make start-full` | Gateway + ClickHouse + Python API | Go + Docker |
| `make start-router` | Full semantic routing (`model="auto"`) | Go + weights |
| `make dev-python` | Python API only (uvicorn --reload) | Python |

### API Keys

Configure via the UI, environment variables, or `~/.lunar/secrets.json`:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
make start-full
```

## API Endpoints

### Gateway (Go Engine — port 8080)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (any provider) |
| `POST` | `/v1/route` | Route a prompt without generating |
| `GET`  | `/v1/models` | List registered models |
| `GET`  | `/health` | Health check |

### Python API (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Analytics** | | |
| `GET` | `/v1/stats/{tenant}/analytics` | Full analytics (traces, cost, latency, distributions) |
| **Clustering** | | |
| `POST` | `/v1/clustering/run` | Run clustering pipeline (embed, cluster, label) |
| `GET` | `/v1/clustering/datasets` | List domain datasets from latest run |
| `GET` | `/v1/clustering/datasets/{run}/{cluster}` | Get traces for a cluster |
| **Datasets** | | |
| `GET` | `/v1/datasets` | List all datasets (eval + domain clusters) |
| `POST` | `/v1/datasets` | Create evaluation dataset |
| `POST` | `/v1/datasets/{id}/samples` | Add samples to dataset |
| **Evaluations** | | |
| `POST` | `/v1/evaluations` | Create and run evaluation (async) |
| `GET` | `/v1/evaluations` | List evaluations |
| `GET` | `/v1/evaluations/{id}/status` | Evaluation progress |
| `GET` | `/v1/evaluations/{id}/results` | Evaluation results with scores |
| **Metrics** | | |
| `GET` | `/v1/metrics` | List built-in + custom metrics |
| `POST` | `/v1/metrics` | Create custom metric |
| `POST` | `/v1/auto-eval/suggest-metrics` | AI-powered metric suggestion |
| **Models** | | |
| `GET` | `/v1/models/available` | Models available from configured providers |
| **Harness** | | |
| `GET` | `/v1/harness/agents` | List AI agents |
| `POST` | `/v1/harness/run/{name}` | Run an agent with input |
| `GET` | `/v1/harness/memory` | Query agent memory |
| **Secrets** | | |
| `GET` | `/v1/secrets` | List configured providers |
| `POST` | `/v1/secrets/{provider}` | Save API key |

## Architecture

```
go/                              # Go engine (high-performance gateway)
├── cmd/lunar-engine/            # Entry point
├── internal/
│   ├── provider/                # 13 providers
│   ├── server/                  # HTTP handlers + session management
│   ├── clickhouse/              # Trace writer + 8 migrations
│   ├── router/                  # UniRoute algorithm + LRU cache
│   └── embeddings/              # ONNX MiniLM embedder

lunar_router/                    # Python layer (analytics, clustering, evals)
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

## MCP Integration (Claude Code)

```bash
pip install lunar-router[mcp]
```

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "lunar-router": {
      "command": "python",
      "args": ["-m", "lunar_router.mcp"]
    }
  }
}
```

Tools: `lunar_route`, `lunar_generate`, `lunar_smart_generate`, `lunar_list_models`, `lunar_compare`.

## Development

```bash
make help               # show all commands
make install            # install Python SDK + Go deps
make test               # run all tests
make lint               # lint all code
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [GitHub Repository](https://github.com/lunar-org-ai/lunar-router)
- [HuggingFace Weights](https://huggingface.co/diogovieira/lunar-router-weights)
