# backend/

Request-serving layer. The thing channels talk to.

- `api/` — HTTP/gRPC endpoints.
- `channels/` — adapters: webhook, Slack, widget, etc.
- `orchestrator/` — routes incoming request to `runtime/executor/`.
- `auth/` — keys, rate limits, tenant resolution.

TypeScript. Not mutated by the harness.
