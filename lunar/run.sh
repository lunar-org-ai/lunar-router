#!/bin/bash
# Lunar Router - Quick Start Script

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR/router"

# Check if .env exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "Please edit router/.env and add your API keys"
    else
        echo "Creating default .env..."
        cat > .env << 'EOF'
# Lunar Router Configuration

# Local API Key (for authentication)
LOCAL_API_KEY=lunar-dev-key

# LLM Provider API Keys (add yours)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=

# Semantic Router Settings
UNIROUTE_STATE_PATH=./app/data/uniroute_state
UNIROUTE_EMBEDDING_MODEL=all-MiniLM-L6-v2
UNIROUTE_DEFAULT_COST_WEIGHT=0.0
EOF
        echo "Please edit .env and add your API keys"
    fi
fi

# Check if virtual environment exists
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install lunar_router package from root
echo "Installing lunar_router package..."
pip install -e "$ROOT_DIR" --quiet

# Install API dependencies
echo "Installing API dependencies..."
pip install -r requirements.txt --quiet

# Create data directories
mkdir -p data app/data/uniroute_state/clusters app/data/uniroute_state/profiles

# Run the server
echo ""
echo "============================================"
echo "  Lunar Router - Open Source LLM Routing"
echo "============================================"
echo ""
echo "  Server:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "  Endpoints:"
echo "    POST /v1/chat/completions  - OpenAI-compatible"
echo "    POST /semantic/route       - Semantic routing"
echo "    GET  /pricing/models       - Available models"
echo ""
echo "============================================"
echo ""

uvicorn app.main_local:app --reload --host 0.0.0.0 --port 8000
