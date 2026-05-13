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
import {
  deleteAgentSlackCredentials,
  disconnectAgentSlack,
  getAgentSlackCredentials,
  getAgentSlackStatus,
  putAgentSlackCredentials,
} from '../slack/handler'
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
    // Pass the runtime's status through verbatim — including 4xx like
    // 404 (unknown agent), 409 (already_connected), 400 (validation).
    // The UI relies on these codes to choose its branch (e.g. "already
    // connected, prompt to rotate" vs "real error"). The gateway only
    // synthesizes its own 502 / 504 when fetch itself failed (below).
    const text = await res.text()
    const ct = res.headers.get('content-type') ?? 'application/json'
    return new Response(text, {
      status: res.status,
      headers: { 'content-type': ct },
    })
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

// MCP / Hands — per-agent tool servers (P3.4)
agentsRouter.get(
  '/:id/mcp',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/mcp`),
)
agentsRouter.post(
  '/:id/mcp',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/mcp`),
)
agentsRouter.patch(
  '/:id/mcp/:server',
  proxy('PATCH', (c) =>
    `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/mcp/${encodeURIComponent(c.req.param('server') ?? '')}`,
  ),
)
agentsRouter.delete(
  '/:id/mcp/:server',
  proxy('DELETE', (c) =>
    `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/mcp/${encodeURIComponent(c.req.param('server') ?? '')}`,
  ),
)
agentsRouter.get(
  '/:id/mcp/tools',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/mcp/tools`),
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

// ─── Web widget channel (P3.5) ─────────────────────────────────────────────
// Operator-facing CRUD proxies straight to the Python runtime; storage,
// secret minting, and origin matching all live there.
agentsRouter.get(
  '/:id/channels/web',
  proxy('GET', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/web`),
)
agentsRouter.post(
  '/:id/channels/web/connect',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/web/connect`),
)
agentsRouter.post(
  '/:id/channels/web/rotate-secret',
  proxy('POST', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/web/rotate-secret`),
)
agentsRouter.patch(
  '/:id/channels/web',
  proxy('PATCH', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/web`),
)
agentsRouter.delete(
  '/:id/channels/web',
  proxy('DELETE', (c) => `/agents/${encodeURIComponent(c.req.param('id') ?? '')}/channels/web`),
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

// ─── Per-agent Slack app credentials (P3.5 BYOK) ───────────────────────────
agentsRouter.get('/:id/channels/slack/credentials', async (c) => {
  const id = c.req.param('id') ?? ''
  try {
    return c.json(await getAgentSlackCredentials(id))
  } catch (e) {
    return c.json(
      { error: 'slack credentials read failed', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})

agentsRouter.put('/:id/channels/slack/credentials', async (c) => {
  const id = c.req.param('id') ?? ''
  let body: { client_id?: string; client_secret?: string; signing_secret?: string }
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'invalid json' }, 400)
  }
  if (!body.client_id || !body.client_secret || !body.signing_secret) {
    return c.json(
      { error: 'missing fields: client_id, client_secret, signing_secret all required' },
      400,
    )
  }
  try {
    return c.json(
      await putAgentSlackCredentials(id, body as {
        client_id: string
        client_secret: string
        signing_secret: string
      }),
    )
  } catch (e) {
    return c.json(
      { error: 'slack credentials write failed', detail: e instanceof Error ? e.message : String(e) },
      500,
    )
  }
})

agentsRouter.delete('/:id/channels/slack/credentials', async (c) => {
  const id = c.req.param('id') ?? ''
  try {
    await deleteAgentSlackCredentials(id)
    return c.body(null, 204)
  } catch (e) {
    return c.json(
      { error: 'slack credentials delete failed', detail: e instanceof Error ? e.message : String(e) },
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
