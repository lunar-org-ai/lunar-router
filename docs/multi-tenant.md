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

## MCP HTTP/SSE transport (P16.2)

Customer Claude Code CLIs connect to the hosted runtime via the
official MCP HTTP transports. The exact same 10 introspection tools
that OSS local users see over stdio are available here, but gated by
the per-tenant Bearer.

### Endpoints

| Transport | Endpoint | Notes |
|---|---|---|
| Streamable HTTP (modern) | `POST /mcp/` | One endpoint; client POSTs JSON-RPC, server responds with JSON or an SSE stream depending on `Accept`. |
| SSE (legacy) | `GET /mcp/sse` + `POST /mcp/messages/<id>` | Older Claude Code builds may need this. |

### Auth

Every MCP request must carry the per-tenant Bearer header — the same
`otrcy_live_<…>` token used for the REST surface:

```
Authorization: Bearer otrcy_live_<32-char base32>
```

The runtime resolves the token before any MCP handshake runs. Missing
or wrong-shape tokens return **401 unauthorized** with a JSON body;
the SDK transport is never reached.

### Client setup (Claude Code)

```bash
claude mcp add opentracy \
  --transport http \
  --header "Authorization: Bearer otrcy_live_aBcD…" \
  https://api.opentracy.cloud/mcp/
```

Or for an older Claude Code build that needs SSE:

```bash
claude mcp add opentracy \
  --transport sse \
  --header "Authorization: Bearer otrcy_live_aBcD…" \
  https://api.opentracy.cloud/mcp/sse
```

After `claude` restarts, the 10 tools (`list_recent_promotions`,
`get_lesson`, `router_health_check`, etc) appear in Claude Code's
tool palette and read **only** that tenant's data.

### Tools exposed

Same set as the OSS local stdio server (`harness/introspection/tools.py`):

- `list_recent_promotions`, `list_recent_rollbacks`, `get_lesson`
- `get_day_epoch`, `list_predictions`, `list_available_epochs`
- `router_health_check`, `propose_router_retrain`
- `dataset_health_check`, `propose_dataset_curation`

### What if I'm running OSS?

Then you don't need any of this. `.mcp.json` at the repo root already
points `claude mcp` at the stdio server — same tools, no Bearer needed,
no infra needed. The HTTP transport is only mounted when
`OPENTRACY_MULTI_TENANT=1` and returns **503 mcp_disabled** otherwise.

## At-rest encryption via KMS (P16.3)

Per-tenant BYOK keys (Anthropic, OpenAI) can be **envelope-encrypted
at rest** using Google Cloud KMS. With KMS on, a filesystem snapshot
or gcsfuse mount leak gives an attacker only ciphertext — they'd
need both the disk image AND IAM access to the KEK to recover keys.

The KMS feature is **independent of multi-tenant mode**: an operator
can enable it for a single-tenant deploy too if they want at-rest
encryption.

### Enabling KMS

1. Provision the KEK once (idempotent — done by `opentracy-infra/bootstrap/00-bootstrap-project.sh`):

   ```bash
   gcloud kms keyrings create opentracy-byok --location=us-east4
   gcloud kms keys create byok-master \
     --keyring=opentracy-byok --location=us-east4 \
     --purpose=encryption
   ```

2. Grant the Cloud Run service account encrypt + decrypt on the key:

   ```bash
   gcloud kms keys add-iam-policy-binding byok-master \
     --keyring=opentracy-byok --location=us-east4 \
     --member="serviceAccount:opentracy-runtime@${PROJECT}.iam.gserviceaccount.com" \
     --role=roles/cloudkms.cryptoKeyEncrypterDecrypter
   ```

3. Set the env var on the runtime:

   ```bash
   export OPENTRACY_KMS_KEY_NAME=projects/${PROJECT}/locations/us-east4/keyRings/opentracy-byok/cryptoKeys/byok-master
   ```

4. Install the optional dep:

   ```bash
   pip install opentracy-new-mode[kms]
   ```

5. Migrate any existing plaintext secrets:

   ```bash
   # Dry-run first
   uv run python -m tools.migrate_secrets_to_kms --dry-run

   # Then for real, deleting plaintext after a successful encrypt
   uv run python -m tools.migrate_secrets_to_kms --delete-plaintext
   ```

### What gets encrypted

Only **per-agent secrets files** — `tenants/<tid>/agents/<aid>/secrets.env`
becomes `tenants/<tid>/agents/<aid>/secrets.enc.json` after migration.

Each `.enc.json` file is a self-contained envelope:

```json
{
  "v": 1,
  "kek": "projects/<…>/cryptoKeys/byok-master",
  "kek_version": 3,
  "nonce_b64": "<12 bytes>",
  "wrapped_dek_b64": "<KMS-wrapped DEK>",
  "ciphertext_b64": "<AES-256-GCM ciphertext>"
}
```

A fresh 32-byte DEK is generated locally per file; only the wrapped
DEK ever leaves the runtime process. AES-256-GCM (via the audited
`cryptography.hazmat` library) protects the dotenv body.

Other per-agent state (`onboarding.json`, `integrations/<channel>.json`,
`improvement.yaml`, etc) stays plaintext — they hold non-secret
configuration. If a channel integration carries its own signing
secret (Slack, Twilio), encrypting THAT is a follow-up scope.

### Threat model

✅ At-rest leak (disk image, gcsfuse mount, bucket snapshot): ciphertext only.
❌ In-memory leak: decrypted keys are in process memory before the LLM SDK uses them. Anyone with process memory access already has full attack surface.
❌ KMS misconfiguration: wrong key version, IAM removed mid-flight, KMS quota exceeded → the runtime fails closed (decrypt error raises, not silent empty dict).

### Reverse migration (rollback)

`tools/migrate_secrets_to_kms.py` deliberately has no reverse mode —
moving back to plaintext is a security regression that should be a
conscious operator decision with an audit log entry. The procedure
lives in the `opentracy-infra` runbook.

## Known limitations

- ~~**Singleton state**: the active tenant lives on a module global, not
  a contextvar. Two concurrent requests in the same Python process can
  race. P16.1 explicitly assumes single-tenant-per-process; P16.2 fixes
  via request-scoped executors.~~ Fixed in P16.2: `tenant_context`
  now uses `contextvars.ContextVar` for proper per-request isolation.
- **No KMS key rotation** yet. The runtime stamps `kek_version` on
  every envelope but doesn't re-encrypt when KMS rotates the primary
  version. Handling rotation cleanly is a P16.3.1 follow-up.
- **DEK caching**: every read of a per-tenant secret = one
  `kms.decrypt` call. At reasonable traffic this is fine; under heavy
  load we'll want an in-process LRU keyed by `(kek, kek_version, wrapped_dek_hash)`.
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
