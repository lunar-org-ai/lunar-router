/**
 * Versions channel — proxy to runtime /versions endpoints.
 *
 *  GET  /v1/versions                       → list versions + lessons
 *  POST /v1/versions/:version/rollback     → revert agent/ to that version
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'
import { z } from 'zod'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

const RollbackBodySchema = z.object({
  reason: z.string().optional(),
})

export const versionsRouter = new Hono()

versionsRouter.get('/', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/versions', { headers: proxyHeaders(c), signal: controller.signal })
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

versionsRouter.post('/:version/rollback', async (c) => {
  const version = c.req.param('version')
  if (!version) return c.json({ error: 'missing version' }, 400)

  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }
  const parsed = RollbackBodySchema.safeParse(body)
  if (!parsed.success) {
    return c.json({ error: 'invalid body', details: parsed.error.flatten() }, 400)
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(`${RUNTIME_URL}/versions/${encodeURIComponent(version)}/rollback`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
      body: JSON.stringify(parsed.data),
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
