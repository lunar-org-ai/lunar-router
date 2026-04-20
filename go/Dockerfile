# ============================================================================
# OpenTracy Engine — multi-stage Docker build
# ============================================================================
# Usage:
#   docker build -t opentracy-engine ./go
#   docker run -p 8080:8080 opentracy-engine --weights /weights --no-embedder
# ============================================================================

# --- Stage 1: Build ---
# Debian-based so the built binary is glibc — matches the debian-slim runtime
# stage below. An alpine (musl) binary fails with "no such file or directory"
# when exec'd on glibc at runtime.
FROM golang:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build \
    -ldflags "-s -w -X main.version=$(git describe --tags --always 2>/dev/null || echo docker)" \
    -o /opentracy-engine ./cmd/opentracy-engine

# --- Stage 2: Runtime ---
# glibc-based so we can dlopen the official onnxruntime binary (needed for
# MiniLM embeddings / semantic routing). Alpine/musl is incompatible with
# upstream libonnxruntime.so — _chk symbols aren't exported even via gcompat.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opentracy-engine /usr/local/bin/opentracy-engine

# Default weights path (mount or COPY your weights here)
RUN mkdir -p /weights
VOLUME /weights

EXPOSE 8080

ENTRYPOINT ["opentracy-engine"]
# Default: gateway mode (no weights needed). Override with --weights /weights for semantic routing.
CMD ["--gateway", "--host", "0.0.0.0", "--port", "8080"]
