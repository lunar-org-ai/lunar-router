/**
 * Lessons channel — proxy to runtime /lessons endpoints.
 *
 *  GET  /v1/lessons              → flat lesson feed (Evolution timeline)
 *  GET  /v1/lessons/:id          → single lesson by id (LessonDetail)
 *  POST /v1/lessons/:id/approve  → approve a queued review lesson + promote
 *  POST /v1/lessons/:id/reject   → reject a queued review lesson
 *
 * The shape mirrors the runtime LessonSummary Pydantic model. AHE three-pillar
 * fields (mutations, delta, voice) flow through unchanged so the UI can render
 * component / experience / decision views without re-shaping.
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

export const lessonsRouter = new Hono()

lessonsRouter.get('/', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/lessons', { signal: controller.signal })
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

lessonsRouter.get('/:id', async (c) => {
  const id = c.req.param('id')
  if (!id) return c.json({ error: 'missing id' }, 400)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(`${RUNTIME_URL}/lessons/${encodeURIComponent(id)}`, {
      signal: controller.signal,
    })
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

const reviewAction = (action: 'approve' | 'reject') =>
  async (c: import('hono').Context) => {
    const id = c.req.param('id')
    if (!id) return c.json({ error: 'missing id' }, 400)
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
        `${RUNTIME_URL}/lessons/${encodeURIComponent(id)}/${action}`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        },
      )
      if (res.status === 404 || res.status === 409) {
        const data = await res.json().catch(() => ({}))
        return c.json(data, res.status as 404 | 409)
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

lessonsRouter.post('/:id/approve', reviewAction('approve'))
lessonsRouter.post('/:id/reject', reviewAction('reject'))
