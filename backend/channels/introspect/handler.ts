/**
 * Introspection channel — proxy to runtime /introspect.
 *
 * POST /v1/introspect with { request, history? }
 *   → { response, tool_calls, success, error, model, iterations }
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'
import { z } from 'zod'

const HistoryMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string(),
})

const BodySchema = z.object({
  request: z.string().min(1),
  history: z.array(HistoryMessageSchema).optional(),
})

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 60_000 // tool-use loops can be slow

export const introspectRouter = new Hono()

introspectRouter.post('/', async (c) => {
  let body: unknown
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'body must be valid JSON' }, 400)
  }

  const parsed = BodySchema.safeParse(body)
  if (!parsed.success) {
    return c.json({ error: 'invalid body', details: parsed.error.flatten() }, 400)
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const res = await fetch(RUNTIME_URL + '/introspect', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
      body: JSON.stringify(parsed.data),
      signal: controller.signal,
    })
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
    const msg = e instanceof Error ? e.message : String(e)
    return c.json({ error: 'runtime call failed', detail: msg }, 502)
  } finally {
    clearTimeout(timer)
  }
})
