/**
 * Traces channel — proxy to runtime /traces.
 *
 *  GET /v1/traces                 → paginated trace feed
 *  GET /v1/traces/stream          → SSE live feed (proxied 1:1 from runtime)
 *  GET /v1/traces/:trace_id       → single trace with stages
 *
 * Query params on the list endpoint pass through verbatim:
 *   date, limit, offset, success, agent_version, q
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

export const tracesRouter = new Hono()

// SSE proxy. Must be registered BEFORE the /:id route so the literal path
// wins. No client-side timeout — the connection is long-lived. The upstream
// fetch is cancelled when the downstream client disconnects.
tracesRouter.get('/stream', async (c) => {
  const upstream = await fetch(`${RUNTIME_URL}/traces/stream`, {
    headers: { Accept: 'text/event-stream', ...proxyHeaders(c) },
    signal: c.req.raw.signal,
  })
  if (!upstream.ok || !upstream.body) {
    const text = upstream.body ? await upstream.text().catch(() => '') : ''
    return c.json(
      { error: 'runtime error', detail: `${upstream.status}: ${text.slice(0, 200)}` },
      502,
    )
  }
  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  })
})

tracesRouter.get('/', async (c) => {
  const qs = c.req.url.split('?')[1] ?? ''
  const url = qs ? `${RUNTIME_URL}/traces?${qs}` : `${RUNTIME_URL}/traces`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(url, { headers: proxyHeaders(c), signal: controller.signal })
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
})

const traceLikeProxy = (path: (id: string) => string) =>
  async (c: import('hono').Context) => {
    const id = c.req.param('id')
    if (!id) return c.json({ error: 'missing id' }, 400)
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
    try {
      const res = await fetch(`${RUNTIME_URL}${path(id)}`, { headers: proxyHeaders(c), signal: controller.signal })
      if (res.status === 404) {
        const data = await res.json().catch(() => ({}))
        return c.json(data, 404)
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

tracesRouter.get('/:id', traceLikeProxy((id) => `/traces/${encodeURIComponent(id)}`))

// P16.2 — CSAT signal.
tracesRouter.get('/:id/feedback',
  traceLikeProxy((id) => `/traces/${encodeURIComponent(id)}/feedback`))

// P16.3 — flag side-table (persistent operator marks + auto-flag rule).
tracesRouter.get('/:id/flag',
  traceLikeProxy((id) => `/traces/${encodeURIComponent(id)}/flag`))

tracesRouter.post('/:id/flag', async (c) => {
  const id = c.req.param('id') ?? ''
  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(
      `${RUNTIME_URL}/traces/${encodeURIComponent(id)}/flag`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
        body: JSON.stringify(body),
        signal: controller.signal,
      },
    )
    if (res.status === 400 || res.status === 404 || res.status === 422) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 400 | 404 | 422)
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return c.json(
        { error: 'runtime error', detail: `${res.status}: ${text.slice(0, 200)}` },
        502,
      )
    }
    const data = await res.json()
    return c.json(data, res.status as 200 | 201)
  } catch (e) {
    return c.json(
      { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
      502,
    )
  } finally {
    clearTimeout(timer)
  }
})

tracesRouter.delete('/:id/flag', async (c) => {
  const id = c.req.param('id') ?? ''
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(
      `${RUNTIME_URL}/traces/${encodeURIComponent(id)}/flag`,
      { method: 'DELETE', signal: controller.signal },
    )
    if (res.status === 404) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, 404)
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
})

tracesRouter.post('/:id/feedback', async (c) => {
  const id = c.req.param('id') ?? ''
  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(
      `${RUNTIME_URL}/traces/${encodeURIComponent(id)}/feedback`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
        body: JSON.stringify(body),
        signal: controller.signal,
      },
    )
    if (res.status === 400 || res.status === 404 || res.status === 422) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 400 | 404 | 422)
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return c.json(
        { error: 'runtime error', detail: `${res.status}: ${text.slice(0, 200)}` },
        502,
      )
    }
    const data = await res.json()
    return c.json(data, res.status as 200 | 201)
  } catch (e) {
    return c.json(
      { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
      502,
    )
  } finally {
    clearTimeout(timer)
  }
})

export const sessionsRouter = new Hono()
sessionsRouter.get('/:id', traceLikeProxy((id) => `/sessions/${encodeURIComponent(id)}`))
