/**
 * Webhook channel — the simplest channel.
 *
 * POST /v1/webhook with { request: string, history?: [{role, content}] }
 * → { response, trace_id, duration_ms, success, error }
 */

import { Hono } from 'hono'
import { z } from 'zod'
import { runAgent, RuntimeError } from '../../orchestrator/runtime_client'

const HistoryMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string(),
})

const WebhookBodySchema = z.object({
  request: z.string().min(1),
  history: z.array(HistoryMessageSchema).optional(),
})

export const webhookRouter = new Hono()

webhookRouter.post('/', async (c) => {
  let body: unknown
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'body must be valid JSON' }, 400)
  }

  const parsed = WebhookBodySchema.safeParse(body)
  if (!parsed.success) {
    return c.json({ error: 'invalid body', details: parsed.error.flatten() }, 400)
  }

  try {
    const result = await runAgent(parsed.data.request, parsed.data.history)
    return c.json({
      response: result.response,
      trace_id: result.trace_id,
      duration_ms: result.duration_ms,
      success: result.success,
      error: result.error,
    })
  } catch (e) {
    if (e instanceof RuntimeError) {
      return c.json({ error: 'runtime error', detail: e.message }, 502)
    }
    return c.json(
      { error: 'unexpected error', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})
