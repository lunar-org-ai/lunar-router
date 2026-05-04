# OpenTracy — Developer Makefile
#
#   make up             ← full local dev stack (auto-installs deps on first run)
#   make gateway        ← gateway only, no analytics (fastest)
#   make gateway-db     ← gateway + ClickHouse, no UI/API
#   make stop           ← stop all services
#   make test           ← run all tests
#
# Prerequisites:
#   - Go 1.22+     (engine build)
#   - Python 3.10+ (SDK / API)
#   - Docker       (ClickHouse)
#   - Node.js 18+  (UI)
# ============================================================================

.PHONY: help install install-train download-weights \
	    build build-docker \
	    gateway gateway-db gateway-router \
	    up engine api ui stop \
	    test test-go test-python test-clickhouse \
	    lint lint-go lint-python \
	    db-up db-down db-shell \
	    clean

help: ## Show this help
	@echo ""
	@echo "  \033[1mQuick start:\033[0m"
	@echo "    make up             Install deps + start all services (UI, API, engine, ClickHouse)"
	@echo "    make gateway        Gateway only — no ClickHouse, no UI (fastest)"
	@echo "    make gateway-db     Gateway + ClickHouse (foreground, no UI/Python API)"
	@echo "    make stop           Stop all running services"
	@echo "    make reset          Stop → clear DB → seed 300 traces → start all services"
	@echo "    make seed           Seed 300 fake traces (ClickHouse must be running)"
	@echo ""
	@echo "  \033[1mAll commands:\033[0m"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Install
install: ## Install all dependencies: Python SDK, Go modules, UI npm packages
	pip install -e ".[openai,anthropic,api]"
	cd go && go mod download
	@if [ -d ui ]; then cd ui && npm install; fi

install-train: ## Install training/distillation dependencies (requires CUDA GPU)
	pip install -r opentracy/requirements-train.txt
	@python3 -c "import torch; print('  PyTorch', torch.__version__, '— CUDA:', torch.cuda.is_available())" 2>/dev/null || \
	    echo "  ⚠ PyTorch not found — install: pip install torch --index-url https://download.pytorch.org/whl/cu126"
	@for pkg in unsloth trl peft bitsandbytes datasets; do \
	    python3 -c "import $$pkg; print('  ✓ $$pkg')" 2>/dev/null || echo "  ✗ $$pkg failed"; \
	done

download-weights: ## Download pre-trained routing weights from HuggingFace
	opentracy download weights-mmlu-v1

# Build
build: ## Build the Go engine binary → go/bin/opentracy-engine
	@cd go && go build -ldflags "-X main.version=$$(git describe --tags --always 2>/dev/null || echo dev)" \
	    -o bin/opentracy-engine ./cmd/opentracy-engine
	@echo "  \033[32m✓ Built go/bin/opentracy-engine\033[0m"

build-docker: ## Build all Docker images
	docker compose build

# start (foreground)

gateway: build ## Gateway only — proxy to all providers, no analytics (foreground)
	@echo ""
	@echo "  Engine:  http://localhost:8080"
	@echo "  Usage:   openai/gpt-4o-mini, anthropic/claude-3-5-sonnet, ..."
	@echo ""
	./go/bin/opentracy-engine --gateway

gateway-db: build ## Gateway + ClickHouse analytics (foreground, no UI/Python API)
	@docker compose up clickhouse -d
	@until docker compose exec -T clickhouse clickhouse-client --password opentracy -q "SELECT 1" > /dev/null 2>&1; do sleep 1; done
	@echo "  \033[32m✓ ClickHouse  localhost:8123\033[0m"
	@echo "  \033[32m✓ Engine      http://localhost:8080\033[0m"
	@echo ""
	OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost \
	OPENTRACY_CH_PASSWORD=opentracy OPENTRACY_CH_DATABASE=opentracy \
	./go/bin/opentracy-engine --gateway

gateway-router: build ## Gateway with semantic routing (requires pre-downloaded weights)
	@WEIGHTS_PATH="$$(opentracy path weights-mmlu-v1 2>/dev/null || echo ./weights)"; \
	OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost \
	OPENTRACY_CH_PASSWORD=opentracy OPENTRACY_CH_DATABASE=opentracy \
	./go/bin/opentracy-engine --weights "$$WEIGHTS_PATH" --no-embedder

# UP (background, full stack)
# ClickHouse env vars shared by engine and Python API
export OPENTRACY_CH_ENABLED  := true
export OPENTRACY_CH_HOST     := localhost
export OPENTRACY_CH_PASSWORD := opentracy
export OPENTRACY_CH_DATABASE := opentracy

up: build ## Auto-install deps if needed + start all services in background
	@# Install missing dependencies on first run
	@python3 -c "import opentracy" 2>/dev/null || { echo "  Installing Python packages …"; pip install -e ".[openai,anthropic,api]" -q; }
	@cd go && go mod download
	@if [ ! -d ui/node_modules ]; then echo "  Installing UI dependencies …"; cd ui && npm install --silent; fi
	@# Kill any stale processes on dev ports
	@-lsof -ti :8080 -ti :8000 -ti :3000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@-rm -f /tmp/opentracy-*.pid
	@sleep 1
	@echo ""
	@echo "  \033[1mStarting services …\033[0m"
	@# ClickHouse
	@docker compose up clickhouse -d
	@until docker compose exec -T clickhouse clickhouse-client --password opentracy -q "SELECT 1" > /dev/null 2>&1; do sleep 1; done
	@echo "  \033[32m✓ ClickHouse    localhost:8123\033[0m"
	@# Go engine
	@OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost OPENTRACY_CH_PASSWORD=opentracy OPENTRACY_CH_DATABASE=opentracy \
	    nohup ./go/bin/opentracy-engine --gateway > /tmp/opentracy-engine.log 2>&1 & echo $$! > /tmp/opentracy-engine.pid
	@sleep 2
	@kill -0 $$(cat /tmp/opentracy-engine.pid) 2>/dev/null \
	    && echo "  \033[32m✓ Engine        localhost:8080  (pid $$(cat /tmp/opentracy-engine.pid))\033[0m" \
	    || { echo "  \033[31m✗ Engine failed — check /tmp/opentracy-engine.log\033[0m"; exit 1; }
	@# Python API
	@OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost OPENTRACY_CH_PASSWORD=opentracy OPENTRACY_CH_DATABASE=opentracy \
	    nohup uvicorn opentracy.api.server:app --host 0.0.0.0 --port 8000 > /tmp/opentracy-api.log 2>&1 & echo $$! > /tmp/opentracy-api.pid
	@sleep 3
	@kill -0 $$(cat /tmp/opentracy-api.pid) 2>/dev/null \
	    && echo "  \033[32m✓ Python API    localhost:8000  (pid $$(cat /tmp/opentracy-api.pid))\033[0m" \
	    || { echo "  \033[31m✗ Python API failed — check /tmp/opentracy-api.log\033[0m"; exit 1; }
	@# UI dev server (vite)
	@cd ui && nohup npm run dev -- --port 3000 > /tmp/opentracy-ui.log 2>&1 & echo $$! > /tmp/opentracy-ui.pid
	@sleep 3
	@kill -0 $$(cat /tmp/opentracy-ui.pid) 2>/dev/null \
	    && echo "  \033[32m✓ UI            localhost:3000  (pid $$(cat /tmp/opentracy-ui.pid))\033[0m" \
	    || { echo "  \033[31m✗ UI failed — check /tmp/opentracy-ui.log\033[0m"; exit 1; }
	@echo ""
	@echo "  UI:      http://localhost:3000"
	@echo "  API:     http://localhost:8000"
	@echo "  Engine:  http://localhost:8080"
	@echo "  DB:      localhost:8123"
	@echo ""
	@echo "  \033[2mStop: make stop\033[0m"
	@echo ""

engine: build ## Run only the Go engine (with ClickHouse env)
	OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost \
	OPENTRACY_CH_PASSWORD=opentracy OPENTRACY_CH_DATABASE=opentracy \
	./go/bin/opentracy-engine --gateway

api: ## Run only the Python API server (with hot reload)
	uvicorn opentracy.api.server:app --reload --host 0.0.0.0 --port 8000

ui: ## Run only the UI dev server (vite)
	cd ui && npm run dev -- --port 3000

stop: ## Stop all running services (engine, API, UI, ClickHouse)
	@if [ -f /tmp/opentracy-ui.pid ] && kill $$(cat /tmp/opentracy-ui.pid) 2>/dev/null; then \
		echo "  \033[32m✓ UI stopped\033[0m"; rm -f /tmp/opentracy-ui.pid; \
	else echo "  \033[2m– UI was not running\033[0m"; fi
	@if [ -f /tmp/opentracy-api.pid ] && kill $$(cat /tmp/opentracy-api.pid) 2>/dev/null; then \
		echo "  \033[32m✓ Python API stopped\033[0m"; rm -f /tmp/opentracy-api.pid; \
	else echo "  \033[2m– Python API was not running\033[0m"; fi
	@if [ -f /tmp/opentracy-engine.pid ] && kill $$(cat /tmp/opentracy-engine.pid) 2>/dev/null; then \
		echo "  \033[32m✓ Engine stopped\033[0m"; rm -f /tmp/opentracy-engine.pid; \
	else echo "  \033[2m– Engine was not running\033[0m"; fi
	@-lsof -ti :8080 -ti :8000 -ti :3000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@if docker compose down 2>/dev/null; then \
		echo "  \033[32m✓ ClickHouse stopped\033[0m"; \
	else echo "  \033[2m– ClickHouse was not running\033[0m"; fi

# Test
test: test-go test-python ## Run all tests

test-go: ## Run Go tests
	cd go && go test ./... -count=1

test-python: ## Run Python tests
	pytest tests/ -v

test-clickhouse: ## Run ClickHouse integration tests (requires running ClickHouse)
	cd go && OPENTRACY_CH_ENABLED=true OPENTRACY_CH_PASSWORD=opentracy \
	    go test -v -run TestIntegration ./internal/clickhouse/

# lint

lint: lint-go lint-python ## Lint all code

lint-go: ## Lint Go code
	cd go && go vet ./...

lint-python: ## Lint Python code
	ruff check opentracy/ tests/

# DB
db-up: ## Start ClickHouse container
	docker compose up clickhouse -d

db-down: ## Stop ClickHouse container
	docker compose down clickhouse

db-shell: ## Open ClickHouse SQL shell
	docker compose exec clickhouse clickhouse-client --database opentracy --password opentracy

db-clear: ## Truncate all trace data from ClickHouse (keeps schema)
	docker compose exec -T clickhouse clickhouse-client --database opentracy --password opentracy \
	    -q "TRUNCATE TABLE llm_traces"
	@echo "  \033[32m✓ llm_traces truncated\033[0m"

seed: ## Seed 300 fake traces into ClickHouse (clears existing data first)
	@docker compose up clickhouse -d
	@until docker compose exec -T clickhouse clickhouse-client --password opentracy -q "SELECT 1" > /dev/null 2>&1; do sleep 1; done
	OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost OPENTRACY_CH_PASSWORD=opentracy \
	    OPENTRACY_CH_DATABASE=opentracy python3 scripts/seed_traces.py --count 300

reset: stop ## Full reset: stop services, clear DB, seed 300 traces, start all services
	@echo ""
	@echo "  \033[1mResetting stack …\033[0m"
	@docker compose up clickhouse -d
	@until docker compose exec -T clickhouse clickhouse-client --password opentracy -q "SELECT 1" > /dev/null 2>&1; do sleep 1; done
	@echo "  \033[32m✓ ClickHouse ready\033[0m"
	OPENTRACY_CH_ENABLED=true OPENTRACY_CH_HOST=localhost OPENTRACY_CH_PASSWORD=opentracy \
	    OPENTRACY_CH_DATABASE=opentracy python3 scripts/seed_traces.py --count 300
	@echo ""
	@$(MAKE) up

# Cleanup
clean: ## Remove build artifacts (binaries, caches, egg-info)
	rm -rf go/bin/ outputs/ unsloth_compiled_cache/ .vite/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

.DEFAULT_GOAL := help

