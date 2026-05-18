/**
 * Slack OAuth install flow (P3.3.2).
 *
 *   GET /v1/slack/install?agent_id=<id>      → redirect to Slack authorize
 *   GET /v1/slack/oauth/callback             → handle Slack's callback
 *
 * The ``state`` cookie carries the agent_id from install → callback so
 * the resulting installation lands in the right agent's
 * integrations/slack.json.
 */

import crypto from 'node:crypto'
import { Hono } from 'hono'
import { loadSlackOAuthConfig, SlackConfigError } from './config'
import { writeInstallation } from './storage'

const STATE_TTL_MS = 10 * 60 * 1000
// state → { issued_at, agent_id }
const stateStore = new Map<string, { issuedAt: number; agentId: string }>()

function issueState(agentId: string): string {
  const state = crypto.randomBytes(24).toString('hex')
  stateStore.set(state, { issuedAt: Date.now(), agentId })
  return state
}

function consumeState(state: string): string | null {
  const entry = stateStore.get(state)
  if (!entry) return null
  stateStore.delete(state)
  if (Date.now() - entry.issuedAt > STATE_TTL_MS) return null
  return entry.agentId
}

setInterval(() => {
  const now = Date.now()
  for (const [k, t] of stateStore.entries()) {
    if (now - t.issuedAt > STATE_TTL_MS) stateStore.delete(k)
  }
}, STATE_TTL_MS).unref()

interface SlackOAuthResponse {
  ok: boolean
  error?: string
  app_id?: string
  authed_user?: { id?: string }
  access_token?: string
  bot_user_id?: string
  team?: { id?: string; name?: string }
}

export const oauthRouter = new Hono()

oauthRouter.get('/install', (c) => {
  const agentId = c.req.query('agent_id')
  if (!agentId) {
    return c.json({ error: 'missing agent_id query param' }, 400)
  }

  let cfg
  try {
    cfg = loadSlackOAuthConfig(agentId)
  } catch (e) {
    if (e instanceof SlackConfigError) return c.json({ error: e.message }, 503)
    throw e
  }

  const state = issueState(agentId)
  const redirect = `${cfg.publicBaseUrl}/slack/oauth/callback`
  const url = new URL('https://slack.com/oauth/v2/authorize')
  url.searchParams.set('client_id', cfg.clientId)
  url.searchParams.set('scope', cfg.scopes)
  url.searchParams.set('redirect_uri', redirect)
  url.searchParams.set('state', state)

  return c.redirect(url.toString(), 302)
})

oauthRouter.get('/oauth/callback', async (c) => {
  const code = c.req.query('code')
  const state = c.req.query('state')
  const err = c.req.query('error')

  if (err) {
    return c.html(callbackHtml({ ok: false, message: `Slack rejected the install: ${err}` }), 400)
  }
  if (!code || !state) {
    return c.html(callbackHtml({ ok: false, message: 'Missing code or state' }), 400)
  }
  const agentId = consumeState(state)
  if (!agentId) {
    return c.html(callbackHtml({ ok: false, message: 'Invalid or expired state' }), 400)
  }

  // Resolve config using THIS agent's creds (per-agent if available).
  // The install link itself was minted using the same creds, so they
  // should still resolve here unless they were deleted between install
  // and callback.
  let cfg
  try {
    cfg = loadSlackOAuthConfig(agentId)
  } catch (e) {
    if (e instanceof SlackConfigError) {
      return c.html(callbackHtml({ ok: false, message: e.message }), 503)
    }
    throw e
  }

  const redirect = `${cfg.publicBaseUrl}/slack/oauth/callback`
  const params = new URLSearchParams()
  params.set('client_id', cfg.clientId)
  params.set('client_secret', cfg.clientSecret)
  params.set('code', code)
  params.set('redirect_uri', redirect)

  let body: SlackOAuthResponse
  try {
    const res = await fetch('https://slack.com/api/oauth.v2.access', {
      method: 'POST',
      headers: { 'content-type': 'application/x-www-form-urlencoded' },
      body: params.toString(),
    })
    body = (await res.json()) as SlackOAuthResponse
  } catch (e) {
    return c.html(
      callbackHtml({ ok: false, message: `Network error exchanging code: ${String(e)}` }),
      502,
    )
  }

  if (!body.ok) {
    return c.html(
      callbackHtml({ ok: false, message: `Slack error: ${body.error ?? 'unknown'}` }),
      400,
    )
  }

  await writeInstallation({
    agent_id: agentId,
    team_id: body.team?.id ?? '',
    team_name: body.team?.name ?? '',
    bot_user_id: body.bot_user_id ?? '',
    bot_token: body.access_token ?? '',
    installer_user_id: body.authed_user?.id ?? '',
    installed_at: new Date().toISOString(),
  })

  // Bounce the operator back to the UI's agent page on success.
  const back = `${cfg.uiBaseUrl}/?slack_connected=${encodeURIComponent(agentId)}`
  return c.html(callbackHtml({ ok: true, message: 'Slack connected.', redirect: back }))
})

function callbackHtml({
  ok,
  message,
  redirect,
}: {
  ok: boolean
  message: string
  redirect?: string
}): string {
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>Slack ${ok ? 'connected' : 'failed'}</title></head>
<body style="font-family:system-ui;padding:32px;max-width:560px;margin:0 auto;text-align:center">
  <h1 style="font-size:18px;margin:0 0 12px">${ok ? '✓ Slack connected' : '✗ Slack install failed'}</h1>
  <p style="color:#555;font-size:14px;line-height:1.5">${message}</p>
  ${redirect ? `<p style="margin-top:24px"><a href="${redirect}">Continue →</a></p><script>setTimeout(()=>{location.href=${JSON.stringify(redirect)}},1200)</script>` : ''}
</body></html>`
}
