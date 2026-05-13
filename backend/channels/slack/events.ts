/**
 * Slack events webhook (P3.3.2).
 *
 *   POST /v1/slack/events
 *
 * Slack sends the URL verification challenge here on first install,
 * then forwards mentions + DMs to handle. We verify the request
 * signature, dedupe by event_id, resolve which agent owns the
 * workspace, and dispatch to the runtime's public API channel
 * (/api/<agent_id>/chat). Reply gets posted back via chat.postMessage.
 */

import crypto from 'node:crypto'
import { Hono } from 'hono'
import { loadSlackOAuthConfig, SlackConfigError } from './config'
import { findAgentByTeamId, type SlackInstallation } from './storage'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const SIGNATURE_TOLERANCE_SEC = 60 * 5
const DEDUP_TTL_MS = 10 * 60 * 1000
const dedupStore = new Map<string, number>()

function rememberEvent(id: string): boolean {
  const now = Date.now()
  const prior = dedupStore.get(id)
  if (prior !== undefined && now - prior < DEDUP_TTL_MS) return false
  dedupStore.set(id, now)
  return true
}

setInterval(() => {
  const now = Date.now()
  for (const [k, t] of dedupStore.entries()) {
    if (now - t > DEDUP_TTL_MS) dedupStore.delete(k)
  }
}, DEDUP_TTL_MS).unref()

function verifySignature(
  body: string,
  timestamp: string | undefined,
  signature: string | undefined,
  signingSecret: string,
): boolean {
  if (!timestamp || !signature) return false
  const ts = Number(timestamp)
  if (!Number.isFinite(ts)) return false
  if (Math.abs(Date.now() / 1000 - ts) > SIGNATURE_TOLERANCE_SEC) return false

  const base = `v0:${timestamp}:${body}`
  const expected = `v0=${crypto.createHmac('sha256', signingSecret).update(base).digest('hex')}`

  const a = Buffer.from(expected)
  const b = Buffer.from(signature)
  if (a.length !== b.length) return false
  return crypto.timingSafeEqual(a, b)
}

interface SlackInnerEvent {
  type: string
  user?: string
  bot_id?: string
  text?: string
  channel?: string
  channel_type?: string
  thread_ts?: string
  ts?: string
}

interface SlackEventEnvelope {
  type: string
  challenge?: string
  team_id?: string
  event_id?: string
  event?: SlackInnerEvent
}

function stripBotMention(text: string, botUserId: string): string {
  return text.replace(new RegExp(`<@${botUserId}>`, 'g'), '').trim()
}

function isHandleable(ev: SlackInnerEvent, botUserId: string): boolean {
  if (ev.bot_id) return false
  if (ev.user === botUserId) return false
  if (ev.type === 'app_mention') return true
  if (ev.type === 'message' && ev.channel_type === 'im') return true
  return false
}

async function postReply(
  inst: SlackInstallation,
  channel: string,
  text: string,
  threadTs?: string,
): Promise<void> {
  const body: Record<string, unknown> = { channel, text }
  if (threadTs) body.thread_ts = threadTs

  const res = await fetch('https://slack.com/api/chat.postMessage', {
    method: 'POST',
    headers: {
      'content-type': 'application/json; charset=utf-8',
      authorization: `Bearer ${inst.bot_token}`,
    },
    body: JSON.stringify(body),
  })
  const json = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string }
  if (!json.ok) {
    console.error('[slack] chat.postMessage failed:', json.error ?? res.status)
  }
}

/**
 * Route an inbound message to the owning agent's pipeline.
 *
 * We call the runtime's public /api/<agent_id>/chat endpoint with no
 * auth — the runtime trusts the TS gateway (same-origin process pair).
 * That hits the same code path public API callers use.
 */
async function runOnAgent(agentId: string, text: string): Promise<string> {
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

async function handleEvent(inst: SlackInstallation, envelope: SlackEventEnvelope): Promise<void> {
  const ev = envelope.event
  if (!ev || !ev.channel) return
  if (!isHandleable(ev, inst.bot_user_id)) return

  const text = stripBotMention(ev.text ?? '', inst.bot_user_id)
  if (!text) return

  try {
    const reply = await runOnAgent(inst.agent_id, text)
    await postReply(inst, ev.channel, reply, ev.thread_ts ?? ev.ts)
  } catch (e) {
    console.error('[slack] runtime call failed:', e instanceof Error ? e.message : String(e))
    await postReply(
      inst,
      ev.channel,
      `Sorry, something went wrong on my end.`,
      ev.thread_ts ?? ev.ts,
    )
  }
}

export const eventsRouter = new Hono()

eventsRouter.post('/events', async (c) => {
  let cfg
  try {
    cfg = loadSlackOAuthConfig()
  } catch (e) {
    if (e instanceof SlackConfigError) return c.json({ error: e.message }, 503)
    throw e
  }

  const rawBody = await c.req.text()
  const timestamp = c.req.header('x-slack-request-timestamp')
  const signature = c.req.header('x-slack-signature')

  if (!verifySignature(rawBody, timestamp, signature, cfg.signingSecret)) {
    return c.json({ error: 'invalid signature' }, 401)
  }

  let envelope: SlackEventEnvelope
  try {
    envelope = JSON.parse(rawBody) as SlackEventEnvelope
  } catch {
    return c.json({ error: 'invalid json' }, 400)
  }

  // URL verification on first install
  if (envelope.type === 'url_verification' && envelope.challenge) {
    return c.text(envelope.challenge)
  }

  if (envelope.event_id && !rememberEvent(envelope.event_id)) {
    // Slack retries — already handled.
    return c.json({ ok: true })
  }

  if (!envelope.team_id) return c.json({ ok: true })
  const inst = findAgentByTeamId(envelope.team_id)
  if (!inst) {
    console.warn('[slack] event for unknown team_id', envelope.team_id)
    return c.json({ ok: true })
  }

  // Acknowledge fast; do the work in the background. Slack retries
  // events we don't 200 within 3s.
  void handleEvent(inst, envelope).catch((e) =>
    console.error('[slack] handleEvent threw:', e),
  )
  return c.json({ ok: true })
})
