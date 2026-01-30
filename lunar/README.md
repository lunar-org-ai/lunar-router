# Lunar - Open Source LLM Router

Lunar is an intelligent LLM routing system that automatically selects the best provider for your requests based on health metrics, cost, and semantic understanding.

## Features

- **Health-First Routing**: Automatically routes requests to the healthiest provider based on error rates, latency, and TTFT (Time to First Token)
- **Multi-Provider Support**: OpenAI, Anthropic, DeepSeek, Gemini, Mistral, Groq, Cohere, and more
- **UniRoute Semantic Routing**: AI-powered routing that understands prompt content and routes to the best model for the task
- **Cost Tracking**: Real-time cost calculation and tracking per request
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Fallback Support**: Automatic failover to backup providers

## Quick Start

### Using Docker

```bash
# Clone the repository
cd lunar

# Copy environment file
cp router/.env.example router/.env

# Add your API keys to router/.env
# At minimum, add OPENAI_API_KEY

# Start with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Running Locally

```bash
cd lunar/router

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn app.main_local:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: lunar-dev-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Force a Specific Provider

```bash
# Use provider/model format to force a specific provider
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: lunar-dev-key" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Semantic Routing (UniRoute)

```bash
# Get optimal model recommendation based on prompt content
curl -X POST http://localhost:8000/v1/semantic/route \
  -H "Content-Type: application/json" \
  -H "x-api-key: lunar-dev-key" \
  -d '{
    "prompt": "Explain quantum computing",
    "cost_weight": 0.1
  }'

# Route and execute in one call
curl -X POST http://localhost:8000/v1/semantic/route-and-execute \
  -H "Content-Type: application/json" \
  -H "x-api-key: lunar-dev-key" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "cost_weight": 0.1
  }'
```

### List Available Models

```bash
curl http://localhost:8000/v1/pricing/models
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `GOOGLE_API_KEY` | Google AI API key | - |
| `MISTRAL_API_KEY` | Mistral API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `LUNAR_DEV_MODE` | Enable development mode | `true` |
| `LUNAR_API_KEYS` | API keys (format: key:tenant) | - |
| `LUNAR_CORS_ORIGINS` | Allowed CORS origins | `localhost:3000` |

### Adding Models and Pricing

Edit `router/data/pricing.json` to add or modify model pricing:

```json
{
  "Provider": "openai",
  "Model": "gpt-4o-mini",
  "ModelId": "gpt-4o-mini-2024-07-18",
  "input_per_million": 0.15,
  "output_per_million": 0.60,
  "cache_input_per_million": 0.075
}
```

## Architecture

```
                    ┌─────────────────┐
                    │   Client App    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Lunar Router   │
                    │                 │
                    │  ┌───────────┐  │
                    │  │ UniRoute  │  │  ← Semantic Analysis
                    │  └───────────┘  │
                    │        │        │
                    │  ┌───────────┐  │
                    │  │  Planner  │  │  ← Health-First Ranking
                    │  └───────────┘  │
                    │        │        │
                    └────────┼────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ OpenAI  │         │Anthropic│         │   ...   │
    └─────────┘         └─────────┘         └─────────┘
```

## UniRoute - Semantic Routing

UniRoute uses embeddings and clustering to understand prompt content and route to the optimal model:

1. **Prompt Embedding**: Converts prompts to 384-dimensional vectors
2. **Cluster Assignment**: Maps embeddings to semantic clusters (code, math, creative, etc.)
3. **Model Scoring**: Calculates expected error for each model based on cluster
4. **Cost-Adjusted Selection**: Balances quality with cost using configurable weight

Formula: `score(h) = expected_error(h) + λ * cost(h)`

## Project Structure

```
lunar/
├── router/
│   ├── app/
│   │   ├── main_local.py      # FastAPI application
│   │   ├── router.py          # Health-first planner
│   │   ├── adapter_*.py       # Provider adapters
│   │   ├── uniroute/          # Semantic routing engine
│   │   │   ├── core/          # Embeddings, clustering
│   │   │   ├── models/        # LLM profiles
│   │   │   └── router/        # UniRoute router
│   │   └── database/
│   │       └── local/         # JSON-based storage
│   ├── data/
│   │   └── pricing.json       # Model pricing
│   ├── requirements.txt
│   └── Dockerfile
├── router-core/               # Shared routing abstractions
├── docker-compose.yml
└── README.md
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
