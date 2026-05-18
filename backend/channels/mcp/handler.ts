/**
 * MCP HTTP/SSE proxy (P16.2.S5).
 *
 * Customer Claude Code CLIs hit the gateway at `/mcp/*`. We forward
 * to the runtime (port 8001) preserving:
 *
 *   - the full URL path (`/mcp/`, `/mcp/sse`, `/mcp/messages/<…>`)
 *   - the Authorization header (per-tenant otrcy_live_<…>) — the
 *     runtime's own middleware resolves it and sets the tenant
 *     ContextVar before the SDK transport runs
 *   - the request body (verbatim — works for both JSON and SSE)
 *   - the response stream (verbatim — preserves Streamable HTTP's
 *     optional SSE response and the SSE transport's keep-alive)
 *
 * The MCP routes are mounted OUTSIDE `/v1/*` because `claude mcp add`
 * expects a clean URL like `https://api.opentracy.cloud/mcp/`. They
 * are NOT gated by `apiKeyAuth` or `tenantAuth` at the gateway level
 * — both gates would interfere with the per-tenant Bearer that
 * Claude Code already sends. The runtime authenticates directly.
 *
 * OSS mode: this router is still mounted but the runtime's handler
 * returns 503 mcp_disabled, so Claude Code sees a clear error rather
 * than a confusing 404.
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'

export const mcpRouter = new Hono()

// All HTTP methods, all sub-paths. Hono's `.all('*', …)` matches every
// request that landed on this router.
mcpRouter.all('*', async (c) => {
  const incoming = c.req.raw
  const url = new URL(incoming.url)
  // Forward everything after the mount point. Hono mounts this router
  // at /mcp, so the upstream URL is `${RUNTIME_URL}/mcp${rest}`.
  const upstreamUrl = `${RUNTIME_URL}/mcp${url.pathname.replace(/^\/mcp/, '')}${url.search}`

  // Preserve method, body, and headers verbatim. Don't drop the
  // Authorization header — the runtime needs it to resolve tenant.
  const init: RequestInit = {
    method: incoming.method,
    headers: incoming.headers,
    body:
      incoming.method === 'GET' || incoming.method === 'HEAD'
        ? undefined
        : incoming.body,
    // `duplex: 'half'` is required by undici/Node when streaming a
    // request body. Without it, body streams fail with "duplex option
    // is required when sending a body".
    // @ts-expect-error — `duplex` is not in the standard RequestInit
    duplex: 'half',
    redirect: 'manual',
  }

  let upstream: Response
  try {
    upstream = await fetch(upstreamUrl, init)
  } catch (e) {
    const reason = e instanceof Error ? e.message : 'fetch failed'
    return c.json({ error: 'runtime_unavailable', detail: reason }, 502)
  }

  // Stream the response back. For SSE this keeps the connection
  // open and forwards events as they arrive; for one-shot JSON it
  // works the same way (just shorter).
  const headers = new Headers()
  upstream.headers.forEach((value, key) => {
    // Some headers shouldn't be forwarded — but for MCP the runtime
    // sets only application-level headers (mcp-session-id, content-
    // type) so we forward all of them.
    headers.set(key, value)
  })

  // Null-body statuses (204, 205, 304) can't carry a body in
  // Response constructor.
  const nullBody =
    upstream.status === 204 || upstream.status === 205 || upstream.status === 304
  return new Response(nullBody ? null : upstream.body, {
    status: upstream.status,
    headers,
  })
})
