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
import { SlackConfigError } from '../slack/config'
import { disconnectAgentSlack, getAgentSlackStatus } from '../slack/handler'
import {
  connectAgentWhatsApp,
  disconnectAgentWhatsApp,
  getAgentWhatsAppStatus,
} from '../whatsapp/handler'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 30_000

export const agentsRouter = new Hono()

const proxy = (
  method: 'GET' | 'POST' | 'DELETE' | 'PATCH' | 'PUT',
  pathBuilder: (c: import('hono').Context) => string,
) => async (c: import('hono').Context) => {
  let body: unknown = undefined
  if (method === 'POST' || method === 'PATCH' || method === 'PUT') {
    try {
      body = await c.req.json()
    } catch {
      body = {}
    }
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const writeMethod = method === 'POST' || method === 'PATCH' || method === 'PUT'
    const res = await fetch(RUNTIME_URL + pathBuilder(c), {
      method,
      headers: writeMethod ? { 'content-type': 'application/json' } : undefined,
      body: writeMethod ? JSON.stringify(body ?? {}) : undefined,
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
agentsRouter.get('/:id', proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}`))
agentsRouter.patch(
  '/:id',
  proxy('PATCH', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}`),
)
agentsRouter.get(
  '/:id/secrets',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/secrets`),
)
agentsRouter.put(
  '/:id/secrets',
  proxy('PUT', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/secrets`),
)
agentsRouter.get(
  '/:id/improvement',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/improvement`),
)
agentsRouter.put(
  '/:id/improvement',
  proxy('PUT', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/improvement`),
)
agentsRouter.get(
  '/:id/channels',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels`),
)
agentsRouter.get(
  '/:id/channels/api',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/api`),
)
agentsRouter.post(
  '/:id/channels/api/connect',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/api/connect`),
)
agentsRouter.post(
  '/:id/channels/api/rotate',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/api/rotate`),
)
agentsRouter.delete(
  '/:id/channels/api',
  proxy('DELETE', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/api`),
)

// ─── Slack channel (handled locally — OAuth files live in TS) ──────────────
agentsRouter.get('/:id/channels/slack', async (c) => {
  const id = c.req.param('id')
  try {
    return c.json(await getAgentSlackStatus(id))
  } catch (e) {
    return c.json(
      { error: 'slack status failed', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})

agentsRouter.delete('/:id/channels/slack', async (c) => {
  const id = c.req.param('id')
  try {
    await disconnectAgentSlack(id)
    return c.body(null, 204)
  } catch (e) {
    if (e instanceof SlackConfigError) return c.json({ error: e.message }, 503)
    return c.json(
      { error: 'slack disconnect failed', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})

// ─── WhatsApp / Twilio ─────────────────────────────────────────────────────
agentsRouter.get('/:id/channels/whatsapp', async (c) => {
  const id = c.req.param('id') ?? ''
  return c.json(await getAgentWhatsAppStatus(id))
})

agentsRouter.put('/:id/channels/whatsapp', async (c) => {
  const id = c.req.param('id') ?? ''
  let body: { account_sid?: string; auth_token?: string; from_number?: string; installer_email?: string | null }
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'invalid json' }, 400)
  }
  if (!body.account_sid || !body.auth_token || !body.from_number) {
    return c.json(
      { error: 'missing fields: account_sid, auth_token, from_number all required' },
      400,
    )
  }
  try {
    await connectAgentWhatsApp(id, body as {
      account_sid: string; auth_token: string; from_number: string; installer_email?: string | null;
    })
    return c.json(await getAgentWhatsAppStatus(id))
  } catch (e) {
    return c.json(
      { error: 'whatsapp connect failed', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})

agentsRouter.delete('/:id/channels/whatsapp', async (c) => {
  const id = c.req.param('id') ?? ''
  try {
    await disconnectAgentWhatsApp(id)
    return c.body(null, 204)
  } catch (e) {
    return c.json(
      { error: 'whatsapp disconnect failed', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})
agentsRouter.post(
  '/:id/activate',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/activate`),
)
agentsRouter.delete(
  '/:id',
  proxy('DELETE', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}`),
)
