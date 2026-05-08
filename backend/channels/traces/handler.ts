/**
 * Traces channel — proxy to runtime /traces.
 *
 *  GET /v1/traces                 → paginated trace feed
 *  GET /v1/traces/:trace_id       → single trace with stages
 *
 * Query params on the list endpoint pass through verbatim:
 *   date, limit, offset, success, agent_version, q
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

export const tracesRouter = new Hono()

tracesRouter.get('/', async (c) => {
  const qs = c.req.url.split('?')[1] ?? ''
  const url = qs ? `${RUNTIME_URL}/traces?${qs}` : `${RUNTIME_URL}/traces`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(url, { signal: controller.signal })
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
      const res = await fetch(`${RUNTIME_URL}${path(id)}`, { signal: controller.signal })
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

export const sessionsRouter = new Hono()
sessionsRouter.get('/:id', traceLikeProxy((id) => `/sessions/${encodeURIComponent(id)}`))
