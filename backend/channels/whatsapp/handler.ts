/**
 * WhatsApp via Twilio (P3.3.3).
 *
 * Twilio webhook hits POST /whatsapp/inbound when someone messages the
 * operator's Twilio WhatsApp number. We verify Twilio's signature,
 * resolve the owning agent by ``To`` (the bot's number), dispatch to
 * the runtime, and reply via Twilio's REST API.
 *
 * Operator setup (one-time):
 *   1. Twilio Console → Messaging → WhatsApp Sandbox or live number
 *   2. Set Inbound URL to <PUBLIC_BASE_URL>/whatsapp/inbound (POST)
 *   3. Copy Account SID + Auth Token + the Twilio number
 *   4. AgentSheet → Channels → WhatsApp → paste credentials
 */

import crypto from 'node:crypto'
import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'
import {
  clearConfig,
  findAgentByFromNumber,
  readConfig,
  writeConfig,
  type WhatsAppConfig,
} from './storage'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const PUBLIC_BASE_URL = (process.env.PUBLIC_BASE_URL ?? '').replace(/\/+$/, '')


// ─── Operator-facing config (read/write per-agent creds) ─────────────────


export interface AgentWhatsAppStatus {
  configured: boolean
  connected: boolean
  from_number: string | null
  account_sid_mask: string | null
  installed_at: string | null
  inbound_url: string | null
  detail: string | null
}

function maskSid(sid: string): string {
  if (!sid) return ''
  if (sid.length <= 8) return sid.slice(0, 2) + '…' + sid.slice(-2)
  return sid.slice(0, 4) + '…' + sid.slice(-4)
}

export async function getAgentWhatsAppStatus(agentId: string): Promise<AgentWhatsAppStatus> {
  const cfg = await readConfig(agentId)
  const inboundUrl = PUBLIC_BASE_URL ? `${PUBLIC_BASE_URL}/whatsapp/inbound` : null
  return {
    configured: !!inboundUrl,
    connected: cfg !== null,
    from_number: cfg?.from_number ?? null,
    account_sid_mask: cfg ? maskSid(cfg.account_sid) : null,
    installed_at: cfg?.installed_at ?? null,
    inbound_url: inboundUrl,
    detail: inboundUrl
      ? null
      : 'Set PUBLIC_BASE_URL on the backend so Twilio can reach the inbound webhook.',
  }
}

export async function connectAgentWhatsApp(
  agentId: string,
  body: {
    account_sid: string
    auth_token: string
    from_number: string
    installer_email?: string | null
  },
): Promise<void> {
  if (!body.account_sid || !body.auth_token || !body.from_number) {
    throw new Error('account_sid, auth_token, and from_number are required')
  }
  const from = body.from_number.trim()
  // Twilio expects "whatsapp:+<E.164>" in API calls. We always store
  // with the prefix so outbound calls don't have to remember.
  const normalized = from.startsWith('whatsapp:') ? from : `whatsapp:${from}`
  const cfg: WhatsAppConfig = {
    agent_id: agentId,
    provider: 'twilio',
    account_sid: body.account_sid.trim(),
    auth_token: body.auth_token.trim(),
    from_number: normalized,
    installer_email: body.installer_email ?? null,
    installed_at: new Date().toISOString(),
  }
  await writeConfig(cfg)
}

export async function disconnectAgentWhatsApp(agentId: string): Promise<void> {
  await clearConfig(agentId)
}


// ─── Twilio webhook signature verification ──────────────────────────────


/**
 * Twilio signs each webhook with HMAC-SHA1 over the URL + concatenated
 * form params. We rebuild the signing string and compare.
 *
 * https://www.twilio.com/docs/usage/webhooks/webhooks-security
 */
function verifyTwilioSignature(
  authToken: string,
  signature: string | undefined,
  url: string,
  params: Record<string, string>,
): boolean {
  if (!signature) return false
  const sorted = Object.keys(params).sort()
  let data = url
  for (const k of sorted) data += k + params[k]
  const expected = crypto.createHmac('sha1', authToken).update(data).digest('base64')
  // Constant-time compare
  if (signature.length !== expected.length) return false
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected))
}


// ─── Inbound webhook ────────────────────────────────────────────────────


async function postOutbound(
  cfg: WhatsAppConfig,
  to: string,
  body: string,
): Promise<void> {
  const url = `https://api.twilio.com/2010-04-01/Accounts/${cfg.account_sid}/Messages.json`
  const params = new URLSearchParams()
  params.set('From', cfg.from_number)
  params.set('To', to)
  params.set('Body', body)
  const auth = Buffer.from(`${cfg.account_sid}:${cfg.auth_token}`).toString('base64')

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      authorization: `Basic ${auth}`,
      'content-type': 'application/x-www-form-urlencoded',
    },
    body: params.toString(),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    console.error('[whatsapp] outbound failed:', res.status, text.slice(0, 200))
  }
}

async function runOnAgent(agentId: string, text: string): Promise<string> {
  // Twilio webhook flow — no tenant Bearer to propagate (Twilio signs the
  // request, not a tenant token). The runtime resolves the tenant from
  // the agent id itself.
  const res = await fetch(`${RUNTIME_URL}/api/${encodeURIComponent(agentId)}/internal-run`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ request: text }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`runtime ${res.status}: ${body.slice(0, 200)}`)
  }
  const data = (await res.json()) as { response?: string; error?: string }
  return data.response ?? data.error ?? 'Sorry — I have no response.'
}


export const whatsappRouter = new Hono()

whatsappRouter.post('/inbound', async (c) => {
  // Twilio posts form-encoded
  const formText = await c.req.text()
  const params: Record<string, string> = {}
  const sp = new URLSearchParams(formText)
  sp.forEach((v, k) => { params[k] = v })

  const to = params['To'] ?? ''
  const from = params['From'] ?? ''
  const body = (params['Body'] ?? '').trim()
  if (!to || !body) return c.text('', 200) // ignore empty pings

  const cfg = findAgentByFromNumber(to)
  if (!cfg) {
    console.warn('[whatsapp] inbound for unknown To:', to)
    return c.text('', 200)
  }

  // Twilio signature uses the FULL request URL. In dev that's
  // http://127.0.0.1:8002/whatsapp/inbound; in prod the public one.
  const url = (PUBLIC_BASE_URL || `http://${c.req.header('host')}`) + '/whatsapp/inbound'
  const sig = c.req.header('x-twilio-signature')
  if (!verifyTwilioSignature(cfg.auth_token, sig, url, params)) {
    console.warn('[whatsapp] invalid signature')
    return c.text('', 401)
  }

  // Acknowledge fast; do the work in the background. Twilio retries
  // delivery within 15s if we don't ack.
  void (async () => {
    try {
      const reply = await runOnAgent(cfg.agent_id, body)
      await postOutbound(cfg, from, reply)
    } catch (e) {
      console.error('[whatsapp] handle failed:', e instanceof Error ? e.message : String(e))
      try {
        await postOutbound(cfg, from, 'Sorry, something went wrong on my end.')
      } catch {
        /* swallow */
      }
    }
  })()

  return c.text('', 200)
})
