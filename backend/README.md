# backend/

Request-serving layer. The thing channels talk to.

- `api/` — Hono entry (`server.ts`). Mounts /health, /v1/* with auth.
- `channels/` — inbound adapters. `webhook/` is the simplest; Slack/widget come later.
- `orchestrator/` — `runtime_client.ts`: typed fetch to the Python runtime service.
- `auth/` — `api_key.ts`: Bearer-key middleware.

TypeScript (Hono + zod + tsx). Not mutated by the harness.

## Architecture

```
client → backend (TS, :8002) → runtime (Python, :8001) → pipeline → traces/
                ↑ auth                  ↑ FastAPI               → ledger/
                ↑ channel routing       ↑ compiles agent.yaml
```

The split is deliberate: TS handles channel-specific SDKs (Slack Bolt, edge
functions, etc.); Python owns the agent itself.

## Running

Two processes. Both must be up.

**Runtime (Python, :8001):**
```bash
uv run python -m runtime.server
```

**Backend (TS, :8002):**
```bash
cd backend
npm install     # first time only
npm run start
```

Optionally enforce auth:
```bash
BACKEND_API_KEYS="key1,key2" npm run start
```
Without `BACKEND_API_KEYS`, the middleware lets every request through (dev mode,
logged via `x-auth-mode: dev-no-keys` header).

## Try it

```bash
curl -X POST http://127.0.0.1:8002/v1/webhook \
  -H 'authorization: Bearer key1' \
  -H 'content-type: application/json' \
  -d '{"request":"Where is order #999?"}'
```

Returns `{ response, trace_id, duration_ms, success }`. The trace is appended
to `traces/raw/<YYYY-MM-DD>.jsonl`.
