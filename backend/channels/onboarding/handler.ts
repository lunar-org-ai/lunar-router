/**
 * Onboarding channel — proxy to runtime /onboarding/* endpoints (P1.11).
 *
 *  GET  /v1/onboarding/state    → current onboarding config
 *  POST /v1/onboarding/complete → finish day-0 onboarding
 *  POST /v1/onboarding/skip     → dismiss without launching
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const FAST_TIMEOUT_MS = 15_000
// /onboarding/turn spawns `claude --print` headless on the operator's
// machine — model latency + subprocess startup can hit 60+s on the first
// call. Hardcoding the slow timeout per path so other onboarding routes
// stay snappy and surface stuck-runtime errors quickly.
const SLOW_TIMEOUT_MS = 180_000

export const onboardingRouter = new Hono()

const proxy = (
  method: 'GET' | 'POST',
  path: string,
  timeoutMs: number = FAST_TIMEOUT_MS,
) => async (c: import('hono').Context) => {
  let body: unknown = undefined
  if (method === 'POST') {
    try {
      body = await c.req.json()
    } catch {
      body = {}
    }
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  // Forward x-tenant-id so the runtime resolves the right tenant
  // scope. Without this, the runtime falls back to `_default` and
  // every Firebase-provisioned user reads the legacy seed
  // `agent/onboarding.json` (completed=true), skipping the chat
  // onboarding on first login.
  const tenant = proxyHeaders(c)
  try {
    const res = await fetch(RUNTIME_URL + path, {
      method,
      headers:
        method === 'POST'
          ? { 'content-type': 'application/json', ...tenant }
          : tenant,
      body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
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
    return c.json(
      { error: 'runtime call failed', detail: e instanceof Error ? e.message : String(e) },
      502,
    )
  } finally {
    clearTimeout(timer)
  }
}

onboardingRouter.get('/state', proxy('GET', '/onboarding/state'))
onboardingRouter.post('/complete', proxy('POST', '/onboarding/complete'))
onboardingRouter.post('/skip', proxy('POST', '/onboarding/skip'))
onboardingRouter.post('/turn', proxy('POST', '/onboarding/turn', SLOW_TIMEOUT_MS))
onboardingRouter.get('/transport', proxy('GET', '/onboarding/transport'))

// v2 — stateful conversational session with inline cards (Phase A).
// /session is the cold-load fetch; /session/say + /decide + /rewind
// drive the conversation forward. SLOW_TIMEOUT_MS on /say because
// the brain call still goes through Claude Code / Anthropic API and
// can take 30-60s on first hit.
onboardingRouter.get('/session', proxy('GET', '/onboarding/session'))
onboardingRouter.post('/session/say', proxy('POST', '/onboarding/session/say', SLOW_TIMEOUT_MS))
onboardingRouter.post('/session/decide', proxy('POST', '/onboarding/session/decide'))
onboardingRouter.post('/session/rewind', proxy('POST', '/onboarding/session/rewind'))
onboardingRouter.post('/session/reset', proxy('POST', '/onboarding/session/reset'))

// Slack connect — stubs in Phase A; Phase C wires real OAuth.
onboardingRouter.post('/connect/slack/begin', proxy('POST', '/onboarding/connect/slack/begin'))
onboardingRouter.get('/connect/slack/status', proxy('GET', '/onboarding/connect/slack/status'))
