# PLAN — P16.1 · Multi-tenant foundation

| Field | Value |
|---|---|
| Phase | P16.1 |
| Parent | P16 (Remote MCP, multi-tenant, BYOK, GCP deploy) |
| Status | Not started |
| Depends on | P2.0 (per-agent registry), P3.1 (per-agent BYOK keys) |
| Unblocks | P16.2 (MCP HTTP/SSE), P16.3 (BYOK + KMS), P16.4 (UI admin), P16.5 (deploy) |
| Reference | mirrors `runtime/agent_context.py` + `runtime/agents/registry.py` patterns from P2.0 |

## Goal

Introduce a **tenant** layer above the agent layer so a single process can serve
multiple customer orgs without leaking data between them. Every persistent path
gets prefixed by `tenants/<tenant_id>/`. Every request resolves to a tenant via
its Bearer token (`otrcy_live_…`) before any agent-scoped work runs.

This phase ships **foundation only**: tenant context, path namespacing, registry
CRUD, auth middleware extracting tenant from token, automatic migration of the
existing single-tenant layout to a `_default` tenant. KMS envelope-encryption,
MCP HTTP transport, UI admin tab, request-scoped singletons all stay out — they
have their own phases (P16.2 – P16.4).

## Locked decisions (operator confirmed 2026-05-13)

- **Storage layout**: nested `tenants/<tid>/{agents,ledger,traces,corpora}/...`. Matches the planned `gs://opentracy-prod-data/tenants/<tid>/…` bucket layout. Wins over flat-with-metadata because GCS fuse mounts a single bucket and per-tenant prefix isolation is cleanest at path level.
- **Auth model**: Bearer `otrcy_live_<32-char-base32>` per tenant. Stored hashed (SHA-256) in `tenants/<tid>/tokens.json`. Reverse lookup via a single `tenants/_tokens_index.json` (hash → tenant_id). Operator admin token (current `BACKEND_API_KEYS` env) stays separate, scoped to `/v1/admin/*` only.
- **Migration**: automatic on runtime startup, idempotent. Detects legacy `agents/`, `ledger/<aid>/`, `traces/<aid>/`, `corpora/indexed/` at the project root → moves them under `tenants/_default/…`. Same pattern P2.0 used to migrate `agent/` → `agents/_default/`.
- **Scope cut**: foundation only. Singletons (embedder pool, trace bus) are deferred — single process still implies single active tenant at a time in P16.1. Request-scoped instances land in P16.2 alongside the MCP server work.
- **Tenant ID format**: kebab-case slug, same regex as agent IDs (`^[a-z0-9][a-z0-9-]{2,40}$`). `_default` reserved. `_deleted` reserved.
- **_default tenant**: created on bootstrap, owns all pre-migration data. Cannot be deleted. Local dev keeps working with zero changes.

## Architecture

```
Request flow:

  Customer → Bearer otrcy_live_aBcDeF...
                  │
                  ▼
  backend/api/server.ts ──── apiKeyAuth (admin) for /v1/admin/*
                          ── tenantAuth         for /v1/* (default)
                                  │
                                  ▼ proxy to runtime, x-tenant-id header
  runtime/server.py ─── tenant_middleware
                              │
                              ▼ tenant_context.set_active(tenant_id)
                              ▼
                       agent_context.set_active(agent_id)
                              │
                              ▼
       all storage helpers compute paths under tenants/<tid>/...
```

## File layout (target)

```
tenants/
  _registry.json                          [new — list + metadata]
  _tokens_index.json                      [new — sha256(token) → tenant_id]
  _default/
    tokens.json                           [new — hashes + metadata]
    agents/                               [migrated from project-root agents/]
      <agent_id>/
        prompts/
        pipeline/
        integrations/
        ...
    ledger/                               [migrated from project-root ledger/]
      <agent_id>/entries/
      <agent_id>/lessons/
      <agent_id>/decisions/
      versions/                           [moved from ledger/versions/]
    traces/                               [migrated from project-root traces/]
      <agent_id>/raw/
      <agent_id>/parquet/
      distilled/                          [moved from traces/distilled/]
    corpora/
      indexed/                            [moved from corpora/indexed/]
      ingested/                           [stays gitignored, moved as-is]
  <tenant_id>/
    tokens.json
    agents/
    ledger/
    traces/
    corpora/

# These stay at project root (operator-shared, never per-tenant):
agent/                                    [live agent symlink/dir; resolves per active tenant+agent]
evals/                                    [shared eval surface for now — P16 doesn't split it]
experiments/                              [shared — same reason]
policies/                                 [shared — operator-set rules]
config/                                   [shared — claude_code allowlist]
techniques/                               [shared — read-only catalog]
agents/                                   [stays as a SYMLINK → tenants/_default/agents for back-compat with any caller that didn't get migrated yet; removed in P16.2 once everything is tenant-aware]
```

### Why some dirs stay shared

- `evals/`, `experiments/` — golden suites and experiment runs are operator-level today. Splitting them per-tenant means re-curating goldens per customer; out of scope. P16.4 (UI admin) decides whether to expose a per-tenant eval surface.
- `policies/`, `config/`, `techniques/` — operator-set rules and read-only catalog. Same for everyone.
- `agent/` (singular) — the **live agent dir** that the executor mounts. The runtime mutates this in place as the active agent's source-of-truth. Per-tenant live-agent dirs are deferred to P16.2 (request-scoped executors).

## Module changes

### New modules

**`runtime/tenant_context.py`** — mirrors `runtime/agent_context.py`.

```python
_DEFAULT_TENANT_ID = "_default"
_active: Optional[str] = None
_lock = threading.Lock()

def set_active(tenant_id: str) -> None: ...
def get_active(default: str = _DEFAULT_TENANT_ID) -> str: ...
def reset() -> None:  # tests only
    ...
```

Resolution order (same as agent):
1. Process-global `_active`
2. `OPENTRACY_TENANT_ID` env var
3. `default` argument (`"_default"`)

**`runtime/tenants/__init__.py`**, **`runtime/tenants/registry.py`** — full CRUD.

```python
@dataclass
class TenantSummary:
    id: str
    name: str
    created_at: str
    is_active: bool

@dataclass
class TenantListResponse:
    tenants: list[TenantSummary]
    active: str | None

# slugging + reserved-name validation
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")
_RESERVED = {"_default", "_deleted", "_tokens_index", "_registry"}

def list_tenants() -> TenantListResponse: ...
def create_tenant(name: str, slug: str | None = None) -> TenantSummary: ...
def delete_tenant(tenant_id: str) -> None: ...  # soft-delete to tenants/_deleted/
def get_tenant_dir(tenant_id: str) -> Path: ...   # tenants/<tid>/
def get_active_tenant_dir() -> Path: ...          # tenants/<get_active()>/
```

**`runtime/tenants/tokens.py`** — token mint + lookup.

```python
_TOKEN_PREFIX = "otrcy_live_"
_HASH_ALGO = "sha256"

@dataclass
class TokenRecord:
    hash: str          # sha256(token), no plaintext stored
    label: str
    created_at: str
    last_used_at: str | None

def mint_token(tenant_id: str, label: str) -> tuple[str, TokenRecord]:
    """Returns (plaintext, record). Plaintext shown ONCE."""
    ...

def resolve_token(token: str) -> str | None:
    """Returns tenant_id or None. Touches last_used_at."""
    ...

def revoke_token(tenant_id: str, hash_prefix: str) -> bool: ...
```

`tenants/_tokens_index.json` is a flat map for O(1) lookup. Per-tenant `tokens.json` is the canonical store; the index is rebuildable from per-tenant files via `runtime/tenants/tokens.rebuild_index()`.

### Modified modules

**`runtime/agents/registry.py`** — `_DEFAULT_ROOT` becomes tenant-aware.

```python
# Before:
_DEFAULT_ROOT = Path("agents")

# After:
def _agents_root() -> Path:
    from runtime.tenant_context import get_active
    from runtime.tenants.registry import get_tenant_dir
    return get_tenant_dir(get_active()) / "agents"
```

All call sites in `registry.py` that touched `_DEFAULT_ROOT` switch to `_agents_root()`. Same for `_LIVE_AGENT_DIR` if needed; the live `agent/` dir stays at project root for P16.1 (request-scoped in P16.2).

**`ledger/writer.py`**, **`ledger/versioning.py`** — `_LEDGER_ROOT` becomes tenant-aware via the same `get_tenant_dir(...) / "ledger"` helper.

**`runtime/executor/tracing.py`**, **`runtime/store/traces.py`** — `traces/` path becomes `tenants/<tid>/traces/`.

**`runtime/agents/secrets.py`**, **`runtime/agents/improvement.py`** — already accept an explicit `root` parameter; defaulted to `Path("agents")`. Change default to lazy resolver via tenant context.

**`corpora/store.py`** — `_DEFAULT_ROOT = Path("corpora") / "indexed"` becomes tenant-aware.

**`backend/api/server.ts`** — split the middleware chain:

```ts
// Before:
app.use('/v1/*', apiKeyAuth)

// After:
app.use('/v1/admin/*', adminAuth)         // operator admin token (BACKEND_API_KEYS)
app.use('/v1/*', tenantAuth)              // tenant token (otrcy_live_…)
                                           //   → sets c.set('tenant_id', resolved)
                                           //   → adds x-tenant-id header on the proxy fetch
```

**`backend/auth/tenant.ts`** (new) — middleware that resolves the Bearer to a tenant_id by calling `GET /v1/admin/tokens/resolve` on the runtime (or a local lookup if the backend co-locates the registry — TBD, prefer round-tripping for clarity).

**`runtime/server.py`** — startup migration + tenant middleware.

- Lifespan startup: call `runtime.tenants.bootstrap.migrate_legacy_to_default()` BEFORE loading the agent config. Idempotent.
- ASGI middleware: extract `x-tenant-id` header from incoming requests, call `tenant_context.set_active(...)` for the duration of the request. Falls back to `_default` for legacy callers (local dev, smoke tests).
- New routes (under `/v1/admin/tenants/...` — operator-only):
  - `GET /v1/admin/tenants` → list
  - `POST /v1/admin/tenants` `{name, slug?}` → create
  - `DELETE /v1/admin/tenants/{tid}` → soft-delete
  - `GET /v1/admin/tenants/{tid}/tokens` → list
  - `POST /v1/admin/tenants/{tid}/tokens` `{label}` → mint, returns plaintext ONCE
  - `DELETE /v1/admin/tenants/{tid}/tokens/{hash_prefix}` → revoke
- New internal route:
  - `POST /v1/admin/tokens/resolve` `{token}` → `{tenant_id}` or 401 (used by backend's tenantAuth middleware)

### Public webhook surfaces

Slack, WhatsApp, Widget public endpoints need to resolve tenant BEFORE resolving agent. The current code already resolves agent from widget_id / slack team_id / api token. Approach:

- Each agent's `integrations/<channel>.json` carries the `tenant_id` (back-filled by the migration).
- Webhook handler looks up the integration file by external identifier (widget_id, team_id, etc.) → reads `tenant_id` and `agent_id` → calls `tenant_context.set_active(tenant_id)` then `agent_context.set_active(agent_id)`.
- For `_default` tenant: same flow, no observable change for single-tenant local dev.

## Migration

`runtime/tenants/bootstrap.py:migrate_legacy_to_default()` runs idempotently on every startup.

Detection rules — if the project root has ALL of:
- `agents/` (dir, non-symlink, non-empty)
- no `tenants/_default/agents/` yet

then perform the migration:

1. Create `tenants/_default/` dir.
2. Move (`shutil.move`) project-root → `tenants/_default/`:
   - `agents/` → `tenants/_default/agents/`
   - `ledger/<agent_id>/` (any non-flat subdirs) → `tenants/_default/ledger/<agent_id>/`
   - `ledger/versions/` → `tenants/_default/ledger/versions/`
   - `ledger/entries/`, `ledger/lessons/`, `ledger/decisions/` (legacy flat dirs from pre-P2.1) → `tenants/_default/ledger/_legacy_flat/`
   - `traces/<agent_id>/` → `tenants/_default/traces/<agent_id>/`
   - `traces/distilled/` → `tenants/_default/traces/distilled/`
   - `corpora/indexed/` → `tenants/_default/corpora/indexed/`
3. Leave a SYMLINK `agents -> tenants/_default/agents` at the project root so old code that hasn't been refactored yet keeps working through P16.1.
4. Write `tenants/_registry.json` with `{"_default": {...}}`.
5. Write `tenants/_default/tokens.json` empty (operator can mint via `/v1/admin/tenants/_default/tokens`).
6. Append a `migration.log.json` entry in `tenants/_default/` describing what moved.

Rollback: documented in `docs/runbook.md` (manual `git mv` reverse with the log).

Tests: `runtime/tests/test_tenant_migration.py` — set up a tmp project with the legacy layout, call the migration, assert (a) files moved, (b) symlinks created, (c) idempotent on re-run, (d) writes new files at the new location, (e) reads back consistently.

## API surface (operator-facing)

```
GET  /v1/admin/tenants
     → 200 { tenants: [...], active: "_default" }

POST /v1/admin/tenants
     body: { name: "Acme Corp", slug?: "acme-corp" }
     → 201 { id, name, created_at, is_active }

DELETE /v1/admin/tenants/{tid}
     → 204 (soft-delete; cannot delete _default)

POST /v1/admin/tenants/{tid}/tokens
     body: { label: "production CLI" }
     → 201 { token: "otrcy_live_aBcD…", record: { hash, label, created_at } }
     (the plaintext is the only place this is ever returned)

DELETE /v1/admin/tenants/{tid}/tokens/{hash_prefix}
     → 204

POST /v1/admin/tokens/resolve  [INTERNAL — not exposed via public LB]
     body: { token: "otrcy_live_…" }
     → 200 { tenant_id: "acme-corp" }  |  401 if unknown
```

## Tests / smoke

- **Unit (runtime)**:
  - `runtime/tests/test_tenant_context.py` — set/get/reset, env-var fallback, default fallback.
  - `runtime/tests/test_tenant_registry.py` — create/list/delete, slug validation, reserved-name rejection.
  - `runtime/tests/test_tenant_tokens.py` — mint (plaintext returned once, hash stored), resolve (touches last_used), revoke.
  - `runtime/tests/test_tenant_migration.py` — full legacy → multi-tenant migration, idempotent, symlink correctness.
  - `runtime/tests/test_agent_context_under_tenant.py` — agent context still works when tenant is set; data lands under the right `tenants/<tid>/...` path.

- **Integration**:
  - Extend `/tmp/smoke_all_channels.sh` with `T0.create_tenant`, `T1.mint_token`, `T2.use_token_with_existing_endpoints`. Cover: same agent ID under two tenants writes to separate paths, neither sees the other's ledger or traces.

- **Smoke E2E** (`scripts/smoke_p16.1.sh`, new):
  ```
  # As operator admin:
  curl -H "Authorization: Bearer $ADMIN" -XPOST .../v1/admin/tenants -d '{"name":"acme"}'
  curl -H "Authorization: Bearer $ADMIN" -XPOST .../v1/admin/tenants/acme/tokens -d '{"label":"test"}'
       → captures otrcy_live_…
  # As tenant:
  curl -H "Authorization: Bearer $T" -XGET .../v1/agents     # only acme's agents
  curl -H "Authorization: Bearer $T" -XPOST .../v1/agents/$AID/chat -d '{...}'
  # Verify tenants/acme/agents/$AID/ exists, tenants/_default/ untouched.
  ```

## Risks

| Risk | Mitigation |
|---|---|
| Migration races with concurrent runtime processes (two startups racing on `shutil.move`). | Take a file lock at `tenants/.migration.lock` for the duration of the migration; second process waits then sees the migration is already done and exits the migration step. |
| Symlink `agents -> tenants/_default/agents` confuses git, IDE indexers, or backend storage layer. | (a) gitignore `tenants/` AND keep `agents/` symlink unchecked; (b) backend's `agentsRoot()` resolver explicitly handles symlink (TS lib does this transparently); (c) remove the symlink in P16.2 once all callers are refactored. |
| Singleton state (embedder pool, trace bus) is process-global → first tenant's request wins; second tenant gets first tenant's bus events. | Acceptable for P16.1: one active tenant per process at a time. Document as known limitation. Fix in P16.2 with request-scoped executors. |
| Operator forgets `_default` is reserved and tries to create it via API. | `runtime/tenants/registry.py` rejects creation with 400; tests cover this. |
| Token plaintext leaks because logging accidentally captures the response body. | (a) `mint_token` response body has a one-line warning string + `display: "show_once"` field that clients can key off; (b) lint rule on backend logs to strip `token` field; (c) document in runbook. |
| Legacy CI scripts hardcode `ledger/`, `traces/` paths. | Symlinks `ledger -> tenants/_default/ledger` and `traces -> tenants/_default/traces` at project root, gitignored, created by the migration. Documented in runbook. Removed in P16.2. |
| Tests written before this phase assume `agents/` at project root. | The symlink keeps them green. Tests for the new tenant-aware path land in this phase. |

## What's NOT in this phase

- **MCP HTTP/SSE transport** (P16.2): `/mcp/sse` endpoint, Bearer-token routing, request-scoped executor pools.
- **KMS envelope encryption** for per-tenant Anthropic keys (P16.3). For P16.1, per-tenant BYOK keys stay in `tenants/<tid>/agents/<aid>/secrets.json` (mode 0600, gitignored), same as today's per-agent layout — just under the tenant prefix.
- **UI admin tab** for Tenants (P16.4). For P16.1, all CRUD is curl-only. The UI keeps showing the active tenant's data implicitly; no Tenant switcher in the topbar yet.
- **Singleton refactor** — embedder pool, trace bus, config state stay process-global. One active tenant per process. Multi-tenant concurrent requests in a single process is P16.2.
- **Per-tenant evals / experiments / policies** — those dirs stay shared at project root. Splitting them is a separate scope.
- **Tenant-aware billing or quotas** — out of scope for this phase entirely.

## Order of work

1. Land `runtime/tenant_context.py` + `runtime/tenants/{registry,tokens,bootstrap}.py` with unit tests. No call sites changed yet — just the new module surface.
2. Wire the startup migration in `runtime/server.py` lifespan. Smoke: start the server, assert `tenants/_default/` got created from the legacy layout, restart and assert idempotent.
3. Add admin API routes under `/v1/admin/tenants/*` + integration tests.
4. Refactor path resolvers in `runtime/agents/registry.py`, `ledger/writer.py`, `runtime/executor/tracing.py`, `runtime/store/traces.py`, `corpora/store.py` to use the tenant-aware paths. Keep the project-root symlinks for back-compat. All existing tests stay green.
5. Add the ASGI tenant middleware + `OPENTRACY_TENANT_ID` env fallback. Don't enforce it on `/v1/admin/*`.
6. Backend: split `apiKeyAuth` into `adminAuth` + `tenantAuth`, add `backend/auth/tenant.ts`, thread `x-tenant-id` through the proxy to the runtime.
7. Extend `/tmp/smoke_all_channels.sh` with the multi-tenant cases; write `scripts/smoke_p16.1.sh`.
8. Documentation: `docs/multi-tenant.md` quick reference + entry in `README.md`.

Each step is its own commit. Steps 1-3 can land before any path refactor; the symlink lets the system run in the legacy layout for as long as we need.

## Done when

- `tenants/_default/` exists on every fresh boot and contains everything the project used to keep at the root.
- `curl -H "Authorization: Bearer otrcy_live_<token>" .../v1/agents` returns only that tenant's agents.
- `tenants/A/.../entries.jsonl` and `tenants/B/.../entries.jsonl` for the same logical agent_id never collide.
- The smoke suite (existing 39 cases + new tenant cases) passes.
- A fresh clone, after first runtime boot, has a working `_default` tenant with all legacy data migrated.
