/**
 * Public API channel — operator-facing REST endpoint per agent (P3.3.1).
 *
 * Each agent gets a unique bearer token. External callers POST to
 * /v1/api/<agent_id>/chat with the token to invoke the agent
 * synchronously. The TS gateway authenticates against the token
 * stored in the agent's integrations/api.json, then proxies the
 * actual run through the Python runtime's /run endpoint.
 *
 * Connect/disconnect lifecycle:
 *   POST /v1/agents/<id>/channels/api/connect    → mint a new token
 *   GET  /v1/agents/<id>/channels/api             → status (mask + last used)
 *   POST /v1/agents/<id>/channels/api/rotate      → mint a fresh token (revokes old)
 *   DELETE /v1/agents/<id>/channels/api           → disconnect
 *
 * Tokens are persisted by the Python runtime in
 * agents/<id>/integrations/api.json (mode 0600). The TS gateway
 * proxies all mutation routes through the runtime so there's a single
 * source of truth on disk.
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 30_000

export const apiChannelRouter = new Hono()

// ─── Public chat endpoint ──────────────────────────────────────────────────
// POST /v1/api/<agent_id>/chat
//
// Auth: Authorization: Bearer <token>. The token is looked up in the
// runtime registry (which also bumps last_used_at).
apiChannelRouter.post('/:agent_id/chat', async (c) => {
  const agentId = c.req.param('agent_id')
  const auth = c.req.header('Authorization') ?? ''
  const m = /^Bearer\s+(.+)$/.exec(auth)
  if (!m) return c.json({ error: 'missing bearer token' }, 401)
  const token = m[1].trim()

  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    // Ask the runtime to authenticate + run. Runtime resolves the
    // token, activates the agent context if needed, and invokes the
    // pipeline.
    const res = await fetch(`${RUNTIME_URL}/api/${encodeURIComponent(agentId)}/chat`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: auth,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (res.status === 401 || res.status === 403) {
      return c.json({ error: 'unauthorized' }, 401)
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return c.json({ error: 'runtime error', detail: text.slice(0, 200) }, 502)
    }
    return c.json(await res.json())
  } catch (e) {
    return c.json(
      { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
      502,
    )
  } finally {
    clearTimeout(timer)
  }
})
