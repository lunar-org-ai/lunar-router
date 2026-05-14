# PLAN — P16.2 · MCP HTTP/SSE transport

| Field | Value |
|---|---|
| Phase | P16.2 |
| Parent | P16 (Remote MCP, multi-tenant, BYOK, GCP deploy) |
| Status | Not started |
| Depends on | P16.1 (multi-tenant foundation) |
| Unblocks | P16.3 (KMS), P16.4 (UI admin), P16.5 (deploy) |
| Reference | `harness/introspection/mcp_server.py` (existing stdio MCP server with 10 tools) |

## Goal

Expose the existing `harness/introspection/mcp_server.py` tools over **HTTP** so customer Claude Code CLIs can connect to a hosted OpenTracy via:

```bash
claude mcp add opentracy --transport http \
       --header "Authorization: Bearer otrcy_live_…" \
       https://api.opentracy.cloud/mcp/
```

Per-tenant token gates access; tools see only the calling tenant's data.

## Locked decisions

- **Transport**: Streamable HTTP (modern, single endpoint, `POST /mcp/`). Plus SSE (`GET /mcp/sse` + `POST /mcp/messages/`) for backward compat with older Claude Code builds.
- **Auth**: per-tenant Bearer (`otrcy_live_<…>`), same tokens as REST surface from P16.1. Reuses the runtime's existing token resolution; the MCP mount adds its own minimal middleware so unauthenticated requests get 401 BEFORE any MCP handshake.
- **Tenant context**: upgrade `runtime/tenant_context._active` from a `threading.Lock` + module global to `contextvars.ContextVar`. P16.1 documented this as a known limitation; concurrent MCP sessions in one process REQUIRE proper isolation.
- **Tool reuse**: the 10 introspection tools from the stdio server move into a shared module (`harness/introspection/tools.py`) used by both stdio (local OSS) and HTTP (hosted) transports.
- **OSS vs infra**: HTTP MCP is **infra-only** — gated by `OPENTRACY_MULTI_TENANT=1`. OSS users keep using stdio via `.mcp.json`.
- **Scope cut**: foundation + 10 existing tools. No new tools, no per-tenant tool customization, no rate limiting, no audit logging (that's P16.3+).
- **No session pinning**: Streamable HTTP's stateful session model (session_id) IS supported by the SDK; we use it. SSE keeps its own session id.

## Architecture

```
Claude Code CLI                                  
   │ POST /mcp/  Authorization: Bearer otrcy_live_…
   │ (or GET /mcp/sse + POST /mcp/messages/)     
   ▼                                             
backend (Hono, port 8002)                        
   │ tenantAuth resolves Bearer → tenant_id      
   │ rewrites path /mcp/* to runtime             
   │ forwards x-tenant-id header                 
   ▼                                             
runtime (FastAPI, port 8001)                     
   │ ASGI tenant middleware reads x-tenant-id    
   │ → ContextVar set for this request scope     
   │                                             
   ├── Streamable HTTP at /mcp/  (POST + optional SSE response)
   │   StreamableHTTPSessionManager
   │     ↓                                       
   │   server = mcp.server.Server("opentracy-introspection")
   │   tools: list_recent_promotions, get_lesson, …
   │   handlers read tenant via ContextVar       
   │     ↓                                       
   │   harness.introspection.lib.* reads ledger via
   │   ledger.writer (tenant-aware after P16.1.S4)
   │                                             
   └── SSE at /mcp/sse, /mcp/messages/  (legacy transport, same server)
```

## File changes

### New

- **`runtime/mcp/http.py`** — mounts Streamable HTTP + SSE transports on a Starlette `Router`. Owns auth middleware that resolves the Bearer before any MCP handshake. Exports a `build_mcp_router()` factory the main FastAPI app mounts at `/mcp`.
- **`harness/introspection/tools.py`** — extracted tool definitions + handlers from `mcp_server.py`. Shared between stdio + HTTP. The stdio server becomes a 10-line shim that registers the same tools.
- **`runtime/tests/test_mcp_http.py`** — integration tests via TestClient: handshake, list_tools, call_tool with a real `otrcy_live_` token, isolation between two tenants.

### Modified

- **`runtime/tenant_context.py`** — swap module global + threading.Lock for `contextvars.ContextVar`. Public API stays the same (`set_active`, `get_active`). Multi-tenant tests get cleaner per-test isolation.
- **`runtime/server.py`** — mount `build_mcp_router()` at `/mcp` when `OPENTRACY_MULTI_TENANT=1`. ASGI tenant middleware already in place from P16.1.S5 handles tenant scoping.
- **`harness/introspection/mcp_server.py`** — slim down to a stdio shim that imports + registers tools from `harness/introspection/tools.py`. No behavior change for OSS local users running `.mcp.json`.
- **`backend/api/server.ts`** — proxy `/mcp/*` to the runtime, threading `x-tenant-id` via `proxyHeaders(c)`. Same pattern as `/v1/*`, but mounted OUTSIDE the `/v1/` namespace so the path matches Claude Code's MCP URL convention.

## Tests

| Layer | What | How |
|---|---|---|
| Unit | ContextVar migration preserves API | `runtime/tests/test_tenant_context.py` — same tests, swap impl |
| Unit | Concurrent contexts isolate | new test: spawn 2 asyncio tasks setting + reading different tenants, assert no leak |
| Integration | MCP handshake over HTTP | TestClient POST /mcp/ with handshake JSON, expect proper response |
| Integration | list_tools returns the 10 tools | TestClient call after handshake |
| Integration | call_tool isolation | tenant A's `list_recent_promotions` doesn't return tenant B's promotions |
| Integration | 401 without Bearer | TestClient without auth header → 401 before handshake |
| Integration | Reject non-`otrcy_live_` Bearer | wrong shape → 401 |

## Risks

| Risk | Mitigation |
|---|---|
| Streamable HTTP requires careful ASGI mounting — getting routes wrong returns 404 to Claude Code with a cryptic message. | Pin the route layout in tests; document in `docs/multi-tenant.md`. |
| ContextVar migration might break existing tests that monkeypatch `tenant_context._active` directly. | Search-and-replace those tests to use `set_active(...)` instead, which is the public API anyway. |
| SSE keep-alive over Cloud Run cold-starts: first request times out if the runtime cold-starts. | `min-instances=1` on `runtime-prod` per `opentracy-infra` PLAN_P16.5.md. Documented. |
| Tools currently use shared module state (proposer caches, embedder pool). Concurrent tenant calls might race. | Already documented in P16.1; tools are READ-ONLY against the ledger so the race is benign for now. Writes (out-of-scope for these 10 tools) would need extra discipline. |
| `harness/introspection/mcp_server.py` is referenced by `.mcp.json` (OSS local). Refactoring it could break local Claude Code MCP. | Keep the entry point + tool definitions byte-compatible; only restructure where tool *handlers* live. |

## What's NOT in this phase

- **New tools**: 10 existing tools port over. New tools are a separate scope.
- **Audit logging** of MCP calls: P16.3 adds this with the KMS work.
- **Rate limiting / quotas** per tenant: deferred.
- **Tool argument validation beyond MCP spec**: existing tools already have inputSchema; we don't tighten or expand.
- **Streaming tool responses**: the MCP SDK supports streaming results but the existing tools return small JSON. Not needed.
- **Custom transports**: only Streamable HTTP + SSE. WebSocket is not in MCP spec; HTTP/2 is implicit via uvicorn.

## Order of work

1. **S1** — Extract tool definitions/handlers into `harness/introspection/tools.py`. Stdio server becomes a thin shim. No behavior change for OSS. Existing tests still pass.
2. **S2** — Migrate `runtime/tenant_context.py` to `contextvars.ContextVar`. Update tests that touched the global directly (use `set_active(...)`).
3. **S3** — Build `runtime/mcp/http.py` with auth middleware + Streamable HTTP + SSE mounts. Unit tests for auth (401 paths).
4. **S4** — Mount the MCP router in `runtime/server.py` (flag-gated). Integration tests: handshake, list_tools, call_tool, isolation.
5. **S5** — Backend proxy: `/mcp/*` → runtime, threading tenant header. Type-check (tsc).
6. **S6** — Smoke E2E via `scripts/smoke_p16.2.sh` running a real handshake against a TestClient-spawned app. Docs: append MCP HTTP section to `docs/multi-tenant.md`.
7. **S7** — Open PR (stacked on P16.1).

Each step is its own commit. Steps 1-2 land before any HTTP transport code so the refactor risk is isolated from the transport risk.

## Done when

- `claude mcp add opentracy --transport http --header "Authorization: Bearer otrcy_live_..." http://localhost:8001/mcp/` succeeds against a locally-running runtime in multi-tenant mode.
- `claude` invocation from that CLI can `list_recent_promotions` and the tool sees only that tenant's data.
- OSS local `.mcp.json` keeps working — `claude mcp` against the stdio server returns the same tool list as before.
- 678 existing tests + new MCP tests stay green.
