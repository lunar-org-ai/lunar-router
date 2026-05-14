# Multi-tenant mode (P16.1)

OpenTracy ships in **two modes**:

- **OSS local** (default) — single-tenant. You clone the repo, run it,
  point Claude Code at it. Storage lives at the project root: `agents/`,
  `ledger/`, `traces/`, `corpora/`. No admin tokens, no tenant routing,
  no migration on boot. **Nothing in this doc applies to you.**

- **Hosted/infra** — multi-tenant. Multiple customer orgs share one
  process. Each tenant has its own agents, ledger, traces, corpora,
  audit trail. Enabled by a single env var.

Switching modes is one env var:

```bash
export OPENTRACY_MULTI_TENANT=1
```

Read fresh on every call, so flipping it doesn't require a restart
beyond the obvious "the active process won't run the bootstrap
migration mid-flight."

## On-disk layout

```
tenants/
  _registry.json                  # list of tenants + metadata
  _tokens_index.json              # sha256(token) → tenant_id (lookup)
  _default/                       # bootstrap tenant, owns all pre-migration data
    tokens.json
    agents/<agent_id>/...
    ledger/<agent_id>/{entries,lessons,decisions}/
    traces/<agent_id>/{raw,parquet}/
    corpora/indexed/
  <tenant_id>/
    tokens.json
    agents/...
    ledger/...
    traces/...
    corpora/...
```

Reserved IDs: `_default`, `_deleted`, `_registry`, `_tokens_index`.
Slug rule: `^[a-z0-9][a-z0-9-]{1,40}$` (kebab-case, 2-41 chars).

## Migration from single-tenant

When the runtime boots with `OPENTRACY_MULTI_TENANT=1` for the first
time, it:

1. Detects the legacy layout (`agents/`, `ledger/`, `traces/`, `corpora/`
   at the project root).
2. Moves them under `tenants/_default/<dir>/`.
3. Drops back-compat **symlinks** at the project root pointing to the
   new location. Code that hasn't been refactored to the tenant-aware
   resolvers keeps working through the symlinks.
4. Writes a `tenants/_default/migration.log.json` recording what moved.
5. Idempotent on every subsequent boot — the log presence is the gate.

The migration is process-locked at `tenants/.migration.lock` so two
simultaneous boots don't race.

## Auth model

Two layers of Bearer tokens:

### Operator admin token (`BACKEND_API_KEYS`)

Same env var as before P16.1. Comma-separated list of admin keys.
Gates **only** `/v1/admin/*` in multi-tenant mode — the surface for
managing tenants and minting per-tenant tokens.

### Per-tenant token (`otrcy_live_<…>`)

Format: `otrcy_live_<32 char base32>` (160 bits entropy).
Stored as `sha256(token)` on disk — the plaintext is shown once at
mint time and never persisted. Gates **everything else** on `/v1/*`.

```bash
ADMIN=$BACKEND_API_KEYS

# Create a tenant
curl -H "Authorization: Bearer $ADMIN" \
     -H "Content-Type: application/json" \
     -d '{"name":"Acme Corp"}' \
     -X POST https://api.opentracy.cloud/v1/admin/tenants
# → 201 { id: "acme-corp", ... }

# Mint a token for that tenant
curl -H "Authorization: Bearer $ADMIN" \
     -H "Content-Type: application/json" \
     -d '{"label":"production CLI"}' \
     -X POST https://api.opentracy.cloud/v1/admin/tenants/acme-corp/tokens
# → 201 { token: "otrcy_live_aBcD…",
#         record: { hash_prefix, label, created_at, last_used_at },
#         display: "show_once" }
#
# *** The plaintext token is in the response ONCE. Capture it now. ***

# Tenant uses it
curl -H "Authorization: Bearer otrcy_live_aBcD…" \
     https://api.opentracy.cloud/v1/agents
# → only this tenant's agents

# Revoke
curl -H "Authorization: Bearer $ADMIN" \
     -X DELETE https://api.opentracy.cloud/v1/admin/tenants/acme-corp/tokens/<hash_prefix>
# → 204
```

## Request flow

```
Customer
  │ POST /v1/agents
  │ Authorization: Bearer otrcy_live_…
  ▼
backend (Hono, port 8002)
  │ tenantAuth middleware:
  │   POST runtime/admin/tokens/resolve {token}     (with admin key)
  │   → { tenant_id }
  │   c.set('tenant_id', tenant_id)
  │
  │ channel proxy:
  │   fetch(runtime + path, {
  │     headers: { ..., ...proxyHeaders(c) }   ← adds x-tenant-id
  │   })
  ▼
runtime (FastAPI, port 8001)
  │ @app.middleware("http") tenant_middleware:
  │   read x-tenant-id header
  │   tenant_context.set_active(tenant_id)
  │   try: await next(request)
  │   finally: tenant_context.set_active(previous)
  │
  │ endpoint:
  │   agents_root() resolves through tenant_context →
  │     tenants/<tenant_id>/agents/...
```

## Operator API reference

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET` | `/v1/admin/tenants` | — | list tenants |
| `POST` | `/v1/admin/tenants` | `{name, slug?, description?}` | create. slug auto-derived from name on collision |
| `DELETE` | `/v1/admin/tenants/{tid}` | — | soft-delete to `tenants/_deleted/`. `_default` refused (400). |
| `GET` | `/v1/admin/tenants/{tid}/tokens` | — | list (hash_prefix + label + timestamps; never plaintext) |
| `POST` | `/v1/admin/tenants/{tid}/tokens` | `{label}` | mint; plaintext returned ONCE |
| `DELETE` | `/v1/admin/tenants/{tid}/tokens/{hash_prefix}` | — | revoke |

Internal (not on public surface):

| Method | Path | Notes |
|---|---|---|
| `POST` | `/admin/tokens/resolve` | called by the backend's tenantAuth; not exposed via the gateway. |

## Known limitations (P16.1)

- **Singleton state**: the active tenant lives on a module global, not
  a contextvar. Two concurrent requests in the same Python process can
  race. P16.1 explicitly assumes single-tenant-per-process; P16.2 fixes
  via request-scoped executors.
- **Channel proxies**: only the `agents` channel handler currently
  threads `x-tenant-id` to the runtime. The rest fall back to `_default`
  on the runtime side. They get migrated in P16.2 when real multi-tenant
  traffic ramps up.
- **No KMS**: per-tenant Anthropic / OpenAI keys still live in plain
  `secrets.env` files under `tenants/<tid>/agents/<aid>/secrets.env`
  (mode 0600). KMS envelope encryption lands in P16.3.
- **Shared infra dirs**: `evals/`, `experiments/`, `policies/`,
  `config/`, `techniques/` stay at the project root and are shared
  across tenants. Splitting them is a separate, deliberately-deferred
  scope.
- **No UI**: all tenant CRUD is curl-only. P16.4 adds the operator UI.

## Operator runbook

### "I want to start fresh"

```bash
rm -rf tenants/  ledger/  traces/  agents/  corpora/  # nuke the lot
# Restart the runtime — the migration sees a fresh layout, creates an
# empty tenants/_default/.
```

### "I want to roll back to OSS layout"

```bash
unset OPENTRACY_MULTI_TENANT
# Move data back from tenants/_default/ to the project root:
rm  agents  ledger  traces  corpora             # delete the symlinks
mv  tenants/_default/agents   agents
mv  tenants/_default/ledger   ledger
mv  tenants/_default/traces   traces
mv  tenants/_default/corpora  corpora
rm -rf tenants/
# Restart the runtime — back to single-tenant.
```

### "A token got accidentally logged somewhere — revoke it"

```bash
# Find the hash_prefix from the audit log (look for the mint event)
curl -H "Authorization: Bearer $ADMIN" \
     -X DELETE https://api.opentracy.cloud/v1/admin/tenants/<tid>/tokens/<hash_prefix>
```

The token is unusable as soon as the index is updated (same request).
