# Lunar Router

**Intelligent LLM Routing for Efficient Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/diogovieira/lunar-router-weights)

Lunar Router dynamically selects the best LLM for each prompt, optimizing the trade-off between quality and cost. Based on the papers:
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)
- [Universal Model Routing for Efficient LLM Inference](https://arxiv.org/abs/2502.08773)

## Features

- **7 LLM Providers**: OpenAI, Anthropic, Google Gemini, Groq, Mistral, vLLM, Mock
- **44+ Pre-configured Models**: With pricing information
- **Semantic Routing**: K-Means clustering on prompt embeddings
- **Cost-Quality Trade-off**: Adjustable cost weight parameter
- **Hub System**: Download pre-trained weights like NLTK/spaCy/HuggingFace
- **Training Pipeline**: Train custom routers on your data
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API

## Installation

```bash
pip install lunar-router
```

With all dependencies:

```bash
pip install lunar-router[all]
```

Or install from source:

```bash
git clone https://github.com/pureai-ecosystem/lunar-router.git
cd lunar-router
pip install -e ".[all]"
```

## Quick Start

### 1. Download Pre-trained Weights

```python
import lunar_router

# Download weights from HuggingFace Hub
lunar_router.download("weights-mmlu-v1")
```

Or via CLI:

```bash
lunar-router download weights-mmlu-v1
```

### 2. Load Router and Route Prompts

```python
from lunar_router import load_router

# Load the router
router = load_router()

# Route a prompt
decision = router.route("Explain quantum computing in simple terms")
print(f"Selected model: {decision.selected_model}")
print(f"Expected error: {decision.expected_error:.4f}")
print(f"Cluster: {decision.cluster_id}")
```

### 3. Use LLM Clients Directly

```python
from lunar_router import create_client

# Create clients for different providers
openai = create_client("openai", "gpt-4o-mini")
anthropic = create_client("anthropic", "claude-3-5-haiku-20241022")
groq = create_client("groq", "llama-3.1-8b-instant")

# Generate responses
response = openai.generate("What is the capital of France?")
print(response.text)
print(f"Latency: {response.latency_ms}ms, Tokens: {response.tokens_used}")
```

## Supported Providers

| Provider | Models | Example |
|----------|--------|---------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3-mini | `create_client("openai", "gpt-4o")` |
| **Anthropic** | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus | `create_client("anthropic", "claude-3-5-sonnet-20241022")` |
| **Google** | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash | `create_client("google", "gemini-1.5-flash")` |
| **Groq** | llama-3.3-70b, llama-3.1-8b, mixtral-8x7b | `create_client("groq", "llama-3.1-8b-instant")` |
| **Mistral** | mistral-large, mistral-small, codestral | `create_client("mistral", "mistral-large-latest")` |
| **vLLM** | Any local model | `create_client("vllm", "meta-llama/Llama-2-7b")` |
| **Mock** | For testing | `create_client("mock", "test-model")` |

## Hub System

Download and manage pre-trained weights like NLTK, spaCy, and HuggingFace:

### CLI Commands

```bash
lunar-router list                      # List available packages
lunar-router download weights-mmlu-v1  # Download weights
lunar-router info weights-mmlu-v1      # Show package info
lunar-router path weights-mmlu-v1      # Show installation path
lunar-router remove weights-mmlu-v1    # Remove package
lunar-router verify weights-mmlu-v1    # Verify integrity
```

### Python API

```python
import lunar_router

# List available packages
packages = lunar_router.list_packages()
for pkg in packages:
    print(f"{pkg.id}: {pkg.description}")

# Download a package
lunar_router.download("weights-mmlu-v1")

# Get package info
info = lunar_router.package_info("weights-mmlu-v1")
print(f"Installed: {info['installed']}")

# Get package path
path = lunar_router.path("weights-mmlu-v1")
```

## Cost-Quality Trade-off

Adjust the `cost_weight` parameter to balance quality vs cost:

```python
router = load_router()

# cost_weight = 0.0: Prioritize quality (select best model)
# cost_weight = 1.0: Prioritize cost (select cheapest model)

router.cost_weight = 0.0  # Best quality
decision = router.route("Complex reasoning task")
print(decision.selected_model)  # -> gpt-4o

router.cost_weight = 0.7  # Prefer cheaper
decision = router.route("Complex reasoning task")
print(decision.selected_model)  # -> gpt-4o-mini
```

## Training Custom Routers

Train your own router with custom data:

```python
from lunar_router import (
    PromptDataset,
    KMeansTrainer,
    PromptEmbedder,
    SentenceTransformerProvider,
)

# 1. Prepare your dataset
data = [
    {"prompt": "What is 2+2?", "ground_truth": "4"},
    {"prompt": "Explain relativity", "ground_truth": "..."},
    # ... more samples
]
dataset = PromptDataset(data)

# 2. Create embedder
provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
embedder = PromptEmbedder(provider)

# 3. Train clusters
trainer = KMeansTrainer(embedder, num_clusters=100)
cluster_assigner = trainer.train(dataset)

# 4. Save weights
cluster_assigner.save("./my_weights/clusters")
```

### Full Training Pipeline

```python
from lunar_router import (
    full_training_pipeline,
    TrainingConfig,
    PromptDataset,
    create_client,
)

# Load data
train_data = PromptDataset.load("train.json")
val_data = PromptDataset.load("val.json")

# Create LLM clients for profiling
clients = [
    create_client("openai", "gpt-4o"),
    create_client("openai", "gpt-4o-mini"),
    create_client("groq", "llama-3.1-8b-instant"),
]

# Run full pipeline
config = TrainingConfig(
    num_clusters=100,
    output_dir="./weights",
    embedding_model="all-MiniLM-L6-v2",
)

result = full_training_pipeline(train_data, val_data, clients, config)
```

## MCP Integration (Claude Code / Claw)

Lunar Router includes an MCP (Model Context Protocol) server for integration with Claude Code, Claw, and other MCP-compatible tools.

### Install

```bash
pip install lunar-router[mcp]
```

### Setup for Claude Code

Add the following to your `~/.claude/settings.json` (create if it doesn't exist):

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

> **Conda/venv users**: Use the full path to your Python interpreter instead of `python`:
> ```json
> {
>   "mcpServers": {
>     "lunar-router": {
>       "command": "/path/to/your/env/bin/python",
>       "args": ["-m", "lunar_router.mcp"]
>     }
>   }
> }
> ```

Or run the server manually:

```bash
lunar-router mcp
# or
python -m lunar_router.mcp
```

### Available MCP Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `lunar_route` | Route a prompt to the best model based on semantic understanding | `prompt` |
| `lunar_generate` | Generate a response using a specific provider/model | `prompt`, `provider`, `model` |
| `lunar_smart_generate` | Auto-route and generate in one step | `prompt` |
| `lunar_list_models` | List available models and costs per provider | _(none)_ |
| `lunar_compare` | Compare responses from multiple models side by side | `prompt`, `models` |

### Tool Details

**`lunar_route`** — Semantic routing without generation
```
prompt: "Explain quantum computing"    # The prompt to analyze
cost_weight: 0.3                       # 0.0 = best quality, 1.0 = lowest cost
```
Returns: selected model, expected error, cluster ID, top-5 ranked models with scores.

**`lunar_generate`** — Direct generation with a specific model
```
prompt: "Hello!"
provider: "openai"                     # openai, anthropic, google, groq, mistral
model: "gpt-4o-mini"
max_tokens: 1000                       # optional
temperature: 0.7                       # optional
```

**`lunar_smart_generate`** — Route + generate in one call
```
prompt: "Write a sorting algorithm"
cost_weight: 0.3                       # optional
max_tokens: 1000                       # optional
```

**`lunar_compare`** — Side-by-side comparison
```
prompt: "Explain gravity"
models: [
  {"provider": "openai", "model": "gpt-4o-mini"},
  {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"}
]
```

### Example Usage in Claude Code

Once configured, you can use natural language:

- "Use lunar_route to find the best model for this coding task"
- "Use lunar_smart_generate to answer this question cost-effectively"
- "Use lunar_compare to test GPT-4 vs Claude on this prompt"
- "Use lunar_list_models to show me OpenAI pricing"

## API Server

Run an OpenAI-compatible API server:

```bash
cd lunar
./run.sh
```

Server runs at `http://localhost:8000` with:
- `/docs` - Swagger UI
- `/v1/chat/completions` - OpenAI-compatible endpoint
- `/semantic/route` - Semantic routing endpoint

### OpenAI-Compatible Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Semantic Routing

```bash
curl http://localhost:8000/semantic/route \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "messages": [{"role": "user", "content": "Explain machine learning"}],
    "models": ["gpt-4o", "gpt-4o-mini", "llama-3.1-70b"],
    "cost_weight": 0.3,
    "execute": true
  }'
```

## Architecture

```
lunar_router/
├── core/
│   ├── embeddings.py      # Prompt embeddings (SentenceTransformers, OpenAI)
│   └── clustering.py      # K-Means cluster assignment
├── models/
│   ├── llm_client.py      # LLM provider clients (7 providers)
│   ├── llm_profile.py     # Model performance profiles
│   └── llm_registry.py    # Model registry
├── router/
│   └── uniroute.py        # Main routing logic
├── training/
│   ├── kmeans_trainer.py  # K-Means training
│   └── pipeline.py        # Full training pipeline
├── hub/
│   ├── manager.py         # Download manager (like NLTK/spaCy)
│   └── index.json         # Package registry
├── mcp/
│   └── server.py          # MCP server (Claude Code integration)
└── cli.py                 # Command-line interface
```

## How It Works

The routing formula from the paper:

```
h* = argmin_h [γ(x, h) + λ·c(h)]
```

Where:
- `γ(x, h) = Φ(x)ᵀ · Ψ(h)` - Expected error for prompt x with model h
- `Φ(x)` - Soft cluster assignment of prompt x (SentenceTransformers embeddings)
- `Ψ(h)` - Model h's error rate per cluster
- `λ` - Cost weight (0 = quality only, 1 = cost only)
- `c(h)` - Normalized cost of model h

## Environment Variables

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...

# Optional
LUNAR_DATA_HOME=/custom/path  # Override default data directory
```

## Pre-trained Weights

| Package | Description | Clusters | Models |
|---------|-------------|----------|--------|
| `weights-mmlu-v1` | Trained on MMLU benchmark | 100 | 10 models |
| `weights-default` | Alias for weights-mmlu-v1 | 100 | 10 models |

Download from [HuggingFace Hub](https://huggingface.co/diogovieira/lunar-router-weights).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black lunar_router/
ruff check lunar_router/

# Type check
mypy lunar_router/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use Lunar Router in your research, please cite:

```bibtex
@software{lunar_router,
  title = {Lunar Router: Intelligent LLM Routing for Efficient Inference},
  author = {Diogo Vieira},
  year = {2025},
  url = {https://github.com/pureai-ecosystem/lunar-router}
}

@article{lu2025universal,
  title={Universal Model Routing for Efficient LLM Inference},
  author={Lu, Shan and others},
  journal={arXiv preprint arXiv:2502.08773},
  year={2025}
}
```

## Links

- [GitHub Repository](https://github.com/pureai-ecosystem/lunar-router)
- [HuggingFace Weights](https://huggingface.co/diogovieira/lunar-router-weights)
- [PyPI Package](https://pypi.org/project/lunar-router/)
