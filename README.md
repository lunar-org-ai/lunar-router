# Lunar Router

**Unified LLM Gateway — 13 providers, one API.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/diogovieira/lunar-router-weights)

Lunar Router is an LLM gateway and routing engine. It proxies requests to **13 providers** through a single OpenAI-compatible API, with optional semantic routing that picks the best model for each prompt.

## Quick Start

```bash
git clone https://github.com/lunar-org-ai/lunar-router.git
cd lunar-router
make start
```

That's it. The engine is running at `http://localhost:8080`.

```python
import lunar_router as lr

response = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
print(f"Cost: ${response._cost:.6f}")
```

## Features

- **13 LLM Providers** through one unified API
- **Python SDK** — `lr.completion()` one-liner (like LiteLLM / OpenRouter)
- **Router Class** — load balancing, fallbacks, retries, 4 strategies
- **Streaming** — all providers including Anthropic & Bedrock SSE translation
- **Cost Tracking** — 70+ models with per-token pricing on every response
- **OpenAI-Compatible Gateway** — drop-in `OpenAI(base_url=...)` replacement
- **Semantic Routing** — auto-select the best model per prompt (with weights)
- **Vision / Multimodal** — images via base64 or URL
- **Tool Calling** — function calls with cross-provider translation
- **Computer Use** — `computer_use_preview` with OpenAI/Anthropic translation
- **ClickHouse Analytics** — traces, cost, latency, model-level stats
- **MCP Integration** — Claude Code / Claw via Model Context Protocol

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

### Completion (like LiteLLM)

```python
import lunar_router as lr

# Any provider, same interface
response = lr.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Auto-detect provider from model name
response = lr.completion(model="gpt-4o-mini", messages=[...])

# Async
response = await lr.acompletion(model="openai/gpt-4o-mini", messages=[...])
```

### Streaming

```python
for chunk in lr.completion(model="openai/gpt-4o-mini", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

### Fallbacks & Retries

```python
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

### Cost Tracking

```python
from lunar_router import model_cost, get_model_info

# Every response includes cost
print(f"${response._cost:.6f}")

# Calculate cost manually
cost = model_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)

# 70+ models with pricing data
info = get_model_info("gpt-4o-mini")
# {'input_cost_per_token': 1.5e-07, 'output_cost_per_token': 6e-07, 'max_input_tokens': 128000, ...}
```

### Drop-in OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")

# All 13 providers through one client
client.chat.completions.create(model="openai/gpt-4o-mini", messages=[...])
client.chat.completions.create(model="anthropic/claude-haiku-4-5-20251001", messages=[...])
client.chat.completions.create(model="mistral/mistral-small-latest", messages=[...])
```

## Running

| Command | What | Requires |
|---------|------|----------|
| `make start` | Gateway proxy (no weights needed) | Go |
| `make start-full` | Gateway + ClickHouse analytics | Go + Docker |
| `make start-router` | Full semantic routing (`model="auto"`) | Go + weights |

### API Keys

Configure via the UI, environment variables, or `~/.lunar/secrets.json`:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
make start
```

Keys are auto-loaded from `~/.lunar/secrets.json` on startup and hot-reloaded when the SDK connects.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (`model="auto"` for semantic routing) |
| `POST` | `/v1/route` | Route a prompt without generating |
| `GET`  | `/v1/models` | List available models |
| `POST` | `/v1/config/keys` | Set provider API key at runtime |
| `POST` | `/v1/config/reload` | Reload keys from secrets file |
| `GET`  | `/v1/metrics` | Aggregated request metrics |
| `GET`  | `/v1/cache` | Cache statistics |
| `GET`  | `/health` | Health check |

### Examples

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hello!"}]}'

# Semantic routing (requires weights)
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Explain ML"}]}'

# Vision
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "openai/gpt-4o", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}]}'
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

The routing formula: `h* = argmin_h [γ(x,h) + λ·c(h)]` where `γ(x,h) = Φ(x)ᵀ · Ψ(h)`.

## Architecture

```
go/                          # Go engine (high-performance runtime)
├── cmd/lunar-engine/        # Entry point (--gateway, --weights)
├── internal/
│   ├── provider/            # 13 providers (OpenAI, Anthropic, Bedrock, etc.)
│   │   ├── openai.go        # OpenAI-compatible provider (11 providers)
│   │   ├── anthropic.go     # Anthropic Messages API translation
│   │   ├── bedrock.go       # AWS Bedrock Converse API + SigV4 signing
│   │   ├── stream_adapter.go    # Anthropic SSE → OpenAI SSE
│   │   └── bedrock_stream.go    # Bedrock event-stream → OpenAI SSE
│   ├── router/              # UniRoute algorithm + LRU cache
│   ├── embeddings/          # ONNX MiniLM embedder
│   ├── clickhouse/          # Trace writer + migrations
│   └── server/              # HTTP handlers

lunar_router/                # Python SDK
├── sdk.py                   # completion(), acompletion(), Router class
├── model_prices.py          # 70+ models with pricing
├── loader.py                # load_router() with Go engine auto-detect
├── storage/secrets.py       # API key management (~/.lunar/secrets.json)
├── training/                # Train custom routers
├── mcp/                     # Claude Code MCP server
└── cli.py                   # CLI (download, route, mcp)
```

## Training Custom Routers

```python
from lunar_router import full_training_pipeline, TrainingConfig, PromptDataset, create_client

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
- [SDK Examples Notebook](notebooks/sdk_examples.ipynb)
