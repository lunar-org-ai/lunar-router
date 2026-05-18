/**
 * Web widget channel (P3.5).
 *
 * Two routers:
 *  - widgetPublicRouter — mounted at `/widget/*` OUTSIDE the /v1/* apiKeyAuth
 *    chain, because the embed script on the operator's site fetches v1.js
 *    and POSTs visitor messages without any bearer token. Origin pinning
 *    in the runtime is the gate.
 *  - widgetConfigRouter — mounted under /v1/agents/:id/channels/web,
 *    proxies operator-facing CRUD (connect/rotate/update/disconnect) to
 *    the Python runtime where storage lives.
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 30_000

// ─── Public router — embed script + inbound messages ──────────────────────

export const widgetPublicRouter = new Hono()

const proxyPublic = async (
  c: import('hono').Context,
  path: string,
  init?: RequestInit,
): Promise<Response> => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    // Preserve the original Origin so the runtime can gate against
    // allowed_domains. We don't trust the operator-side env; if a visitor
    // hits /widget/X/message from foo.com, the runtime sees Origin: foo.com.
    const headers: Record<string, string> = {}
    const origin = c.req.header('origin')
    if (origin) headers['origin'] = origin
    if (init?.headers) Object.assign(headers, init.headers as Record<string, string>)

    const res = await fetch(RUNTIME_URL + path, {
      ...init,
      headers,
      signal: controller.signal,
    })
    const ct = res.headers.get('content-type') ?? 'application/octet-stream'
    const cors = res.headers.get('access-control-allow-origin')
    const out: Record<string, string> = { 'content-type': ct }
    if (cors) out['access-control-allow-origin'] = cors
    // Null-body statuses (204/205/304) can't carry a body in the Fetch
    // spec; passing one trips Undici's `Invalid response status code`.
    const nullBody = res.status === 204 || res.status === 205 || res.status === 304
    const body = nullBody ? null : await res.arrayBuffer()
    return new Response(body, { status: res.status, headers: out })
  } finally {
    clearTimeout(timer)
  }
}

widgetPublicRouter.get('/:widgetId/v1.js', (c) =>
  proxyPublic(c, `/widget/${encodeURIComponent(c.req.param('widgetId') ?? '')}/v1.js`),
)

widgetPublicRouter.options('/:widgetId/message', (c) =>
  proxyPublic(c, `/widget/${encodeURIComponent(c.req.param('widgetId') ?? '')}/message`, {
    method: 'OPTIONS',
  }),
)

widgetPublicRouter.post('/:widgetId/message', async (c) => {
  let body: unknown
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'invalid_json' }, 400)
  }
  return proxyPublic(
    c,
    `/widget/${encodeURIComponent(c.req.param('widgetId') ?? '')}/message`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
      body: JSON.stringify(body),
    },
  )
})
