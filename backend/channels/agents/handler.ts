/**
 * Agents channel — proxy to runtime /agents/* endpoints (P2.0).
 *
 *  GET    /v1/agents              → list all agents in the registry
 *  GET    /v1/agents/:id          → one agent
 *  POST   /v1/agents              → create (used by onboarding launch flow)
 *  POST   /v1/agents/:id/activate → switch live agent + recompile pipeline
 *  DELETE /v1/agents/:id          → soft delete
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 30_000

export const agentsRouter = new Hono()

const proxy = (
  method: 'GET' | 'POST' | 'DELETE',
  pathBuilder: (c: import('hono').Context) => string,
) => async (c: import('hono').Context) => {
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
    const res = await fetch(RUNTIME_URL + pathBuilder(c), {
      method,
      headers: method === 'POST' ? { 'content-type': 'application/json' } : undefined,
      body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
      signal: controller.signal,
    })
    if (res.status === 204) {
      return c.body(null, 204)
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return c.json(
        { error: 'runtime error', detail: `${res.status}: ${text.slice(0, 200)}` },
        502,
      )
    }
    const data = await res.json()
    return c.json(data)
  } catch (e) {
    return c.json(
      { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
      502,
    )
  } finally {
    clearTimeout(timer)
  }
}

agentsRouter.get('/', proxy('GET', () => '/agents'))
agentsRouter.post('/', proxy('POST', () => '/agents'))
agentsRouter.get('/:id', proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id'))}`))
agentsRouter.post(
  '/:id/activate',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id'))}/activate`),
)
agentsRouter.delete(
  '/:id',
  proxy('DELETE', (c) => `/agents/${encodeURIComponent(c.req.param('id'))}`),
)
