/**
 * Admin channel — proxies operator-only routes to the runtime (P16.1).
 *
 *  GET    /v1/admin/tenants                              list tenants
 *  POST   /v1/admin/tenants                              create
 *  DELETE /v1/admin/tenants/:id                          soft-delete
 *  GET    /v1/admin/tenants/:id/tokens                   list tokens
 *  POST   /v1/admin/tenants/:id/tokens                   mint (show_once)
 *  DELETE /v1/admin/tenants/:id/tokens/:prefix           revoke
 *
 * All gated by `apiKeyAuth` (BACKEND_API_KEYS env), distinct from the
 * per-tenant `otrcy_live_<…>` tokens that gate everything else.
 *
 * `/v1/admin/tokens/resolve` is NOT exposed here — it's an internal
 * call that the `tenantAuth` middleware makes directly against the
 * runtime, bypassing the public surface so the route can't be used
 * to enumerate tenants.
 */

import { Hono, type Context } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 30_000

export const adminRouter = new Hono()

const proxy = (
  method: 'GET' | 'POST' | 'DELETE',
  pathBuilder: (c: Context) => string,
) => async (c: Context) => {
  let body: unknown = undefined
  if (method === 'POST') {
    try {
      body = await c.req.json()
    } catch {
      body = {}
    }
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const writeMethod = method === 'POST'
    const res = await fetch(RUNTIME_URL + pathBuilder(c), {
      method,
      headers: writeMethod ? { 'content-type': 'application/json' } : undefined,
      body: writeMethod ? JSON.stringify(body ?? {}) : undefined,
      signal: controller.signal,
    })
    if (res.status === 204) {
      return c.body(null, 204)
    }
    const text = await res.text()
    const ct = res.headers.get('content-type') ?? 'application/json'
    return new Response(text, {
      status: res.status,
      headers: { 'content-type': ct },
    })
  } catch (e) {
    const reason = e instanceof Error ? e.message : 'fetch failed'
    return c.json({ error: 'runtime_unavailable', detail: reason }, 502)
  } finally {
    clearTimeout(timer)
  }
}

// Hono types c.req.param('x') as `string | undefined`; in practice it's
// always defined for a matched route, but TS doesn't know that. Coerce
// to a typed helper so the call sites stay readable.
const p = (c: Context, key: string): string => c.req.param(key) ?? ''

adminRouter.get('/features', proxy('GET', () => '/admin/features'))
adminRouter.get('/tenants', proxy('GET', () => '/admin/tenants'))
adminRouter.post('/tenants', proxy('POST', () => '/admin/tenants'))
adminRouter.delete(
  '/tenants/:id',
  proxy('DELETE', (c) => `/admin/tenants/${encodeURIComponent(p(c, 'id'))}`),
)
adminRouter.get(
  '/tenants/:id/tokens',
  proxy(
    'GET',
    (c) => `/admin/tenants/${encodeURIComponent(p(c, 'id'))}/tokens`,
  ),
)
adminRouter.post(
  '/tenants/:id/tokens',
  proxy(
    'POST',
    (c) => `/admin/tenants/${encodeURIComponent(p(c, 'id'))}/tokens`,
  ),
)
adminRouter.delete(
  '/tenants/:id/tokens/:prefix',
  proxy(
    'DELETE',
    (c) =>
      `/admin/tenants/${encodeURIComponent(
        p(c, 'id'),
      )}/tokens/${encodeURIComponent(p(c, 'prefix'))}`,
  ),
)
