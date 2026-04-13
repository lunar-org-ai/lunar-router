# ============================================================================
# Lunar Engine — multi-stage Docker build
# ============================================================================
# Usage:
#   docker build -t lunar-engine ./go
#   docker run -p 8080:8080 lunar-engine --weights /weights --no-embedder
# ============================================================================

# --- Stage 1: Build ---
FROM golang:1-alpine AS builder

RUN apk add --no-cache git gcc musl-dev

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build \
    -ldflags "-s -w -X main.version=$(git describe --tags --always 2>/dev/null || echo docker)" \
    -o /lunar-engine ./cmd/lunar-engine

# --- Stage 2: Runtime ---
FROM alpine:3.20

RUN apk add --no-cache ca-certificates libstdc++

COPY --from=builder /lunar-engine /usr/local/bin/lunar-engine

# Default weights path (mount or COPY your weights here)
RUN mkdir -p /weights
VOLUME /weights

EXPOSE 8080

ENTRYPOINT ["lunar-engine"]
# Default: gateway mode (no weights needed). Override with --weights /weights for semantic routing.
CMD ["--gateway", "--host", "0.0.0.0", "--port", "8080"]
