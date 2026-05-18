# infra/restate — daily compactor scheduler

Runs `runtime/store/compactor.py` once a day, durably, via [Restate](https://www.restate.dev).

## What's in here

| File | Role |
|---|---|
| `services.py` | The `compactor` service. Two handlers: `tick` (the daily cron, self-rescheduling) and `run_now` (ad-hoc compaction for a specific date). |
| `serve.py` | ASGI entrypoint. `uv run --extra infra python -m infra.restate.serve` hosts the service on `:9080`. |
| `docker-compose.yaml` | Single-node Restate server. Ports `8080` (ingress), `9070` (admin). |

## Run it

```bash
# 1. Start the Restate server.
docker compose -f infra/restate/docker-compose.yaml up -d

# 2. Start the Python service that holds the compactor handler.
uv sync --extra infra
uv run --extra infra python -m infra.restate.serve

# 3. Register the service with Restate. host.docker.internal lets the
#    server (in Docker) reach the service (on the host).
curl http://localhost:9070/deployments \
  --json '{"uri": "http://host.docker.internal:9080"}'

# 4. Bootstrap the daily tick. After this, the service reschedules itself
#    every 24h aligned to 00:05 UTC. Restart-safe, server-restart-safe.
curl http://localhost:8080/compactor/tick

# Optional: ad-hoc backfill of a single day.
curl http://localhost:8080/compactor/run_now --json '"2026-05-07"'
```

## Inspect

```bash
# List registered services.
curl http://localhost:9070/services

# List recent invocations of the compactor.
curl http://localhost:9070/query/invocations | jq '.[] | select(.target | startswith("compactor"))'

# Optional UI (if you'd rather click).
open http://localhost:9070
```

## Tear down

```bash
docker compose -f infra/restate/docker-compose.yaml down --volumes
```

## Why Restate (and when it's overkill)

**Honest take.** For a single daily compactor that takes <1s, this is meaningfully more
infrastructure than a one-line cron. You're getting:

- **Durable schedule.** The next-day delayed call is journaled. Restart the
  Restate server, restart the worker, change machines — the next tick still
  fires on time.
- **Exactly-once side effects.** `ctx.run("compact", …)` is journaled. If a
  tick is replayed, the compaction isn't re-executed; the journal value is
  replayed instead. Date-keyed `idempotency_key` on the rescheduling adds a
  second guarantee at the invocation layer.
- **Audit trail.** Every tick (success or failure) is queryable from the admin
  API. Cron has no such record without you building one.
- **Retries with policy.** `RunOptions(max_attempts=5, max_retry_duration=…)`
  is one line; cron's retry story is "set up monitoring + on-call."

**It earns its keep when** Restate also runs the harness loop (durable
proposer → critics → approver → executor) or other restart-safe workflows.
Right now it runs one cron, so the cost-to-value ratio is high. Treat this
as evaluation infrastructure.

**If you decide it's not worth it,** delete this folder and replace with a
`@daily` cron line; the compactor CLI (`python -m runtime.store.compactor`)
already handles everything Restate is wrapping.

## What's NOT here

- TLS / production hardening — single-node, dev-grade. Add `--identity-keys`
  to `restate.app(...)` and put it behind your own gateway before exposing
  beyond localhost.
- Service container — `serve.py` runs on the host. A `Dockerfile` and a
  `service` entry in `docker-compose.yaml` would make it self-contained;
  deferred until the harness also runs as a Restate service.
- Metrics scrape — Restate exposes Prometheus at `:9070/metrics`. Wire it up
  if you ever stand up Grafana for ops telemetry (separate from the
  training-data telemetry that lives in DuckDB+Parquet).
