/**
 * Dataset channel — proxy to runtime /datasets/* endpoints (P15.4.2).
 *
 *  GET    /v1/datasets               → list (optional ?use=&owner=&sourceType=).
 *  GET    /v1/datasets/:name         → full detail + samples + history.
 *  GET    /v1/datasets/:name/health  → coverage report (cluster_distribution).
 *  POST   /v1/datasets               → manual create via AHE pipeline.
 *  PUT    /v1/datasets/:name         → manual meta edit via AHE pipeline.
 *  DELETE /v1/datasets/:name         → soft delete (registry flag).
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

export const datasetRouter = new Hono()

// GET proxy that forwards query string from the incoming request.
const getProxy = (runtimePath: string) => async (c: import('hono').Context) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  const qs = c.req.url.includes('?') ? c.req.url.slice(c.req.url.indexOf('?')) : ''
  try {
    const res = await fetch(RUNTIME_URL + runtimePath + qs, { signal: controller.signal })
    // Pass through 4xx that carry user-facing reasons.
    if (res.status === 400 || res.status === 404 || res.status === 409 || res.status === 422) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 400 | 404 | 409 | 422)
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

const bodyProxy = (method: 'POST' | 'PUT', pathBuilder: (c: import('hono').Context) => string) =>
  async (c: import('hono').Context) => {
    let body: unknown = {}
    try {
      body = await c.req.json()
    } catch {
      body = {}
    }
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
    try {
      const res = await fetch(RUNTIME_URL + pathBuilder(c), {
        method,
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      if (
        res.status === 400 ||
        res.status === 404 ||
        res.status === 409 ||
        res.status === 422 ||
        res.status === 500
      ) {
        const data = await res.json().catch(() => ({}))
        return c.json(data, res.status as 400 | 404 | 409 | 422 | 500)
      }
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        return c.json(
          { error: 'runtime error', detail: `${res.status}: ${text.slice(0, 200)}` },
          502,
        )
      }
      // 201 from POST stays 201; everything else 2xx → JSON pass-through.
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 200 | 201)
    } catch (e) {
      return c.json(
        { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
        502,
      )
    } finally {
      clearTimeout(timer)
    }
  }

datasetRouter.get('/', getProxy('/datasets'))
datasetRouter.get('/:name', (c) =>
  getProxy(`/datasets/${encodeURIComponent(c.req.param('name') ?? '')}`)(c),
)
datasetRouter.get('/:name/health', (c) =>
  getProxy(`/datasets/${encodeURIComponent(c.req.param('name') ?? '')}/health`)(c),
)

datasetRouter.post('/', bodyProxy('POST', () => '/datasets'))
datasetRouter.put('/:name', bodyProxy('PUT', (c) =>
  `/datasets/${encodeURIComponent(c.req.param('name') ?? '')}`,
))

datasetRouter.delete('/:name', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(
      RUNTIME_URL + `/datasets/${encodeURIComponent(c.req.param('name') ?? '')}`,
      { method: 'DELETE', signal: controller.signal },
    )
    if (res.status === 404) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, 404)
    }
    if (!res.ok && res.status !== 204) {
      const text = await res.text().catch(() => '')
      return c.json(
        { error: 'runtime error', detail: `${res.status}: ${text.slice(0, 200)}` },
        502,
      )
    }
    return c.body(null, 204)
  } catch (e) {
    return c.json(
      { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
      502,
    )
  } finally {
    clearTimeout(timer)
  }
})
