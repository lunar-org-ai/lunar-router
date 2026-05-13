/**
 * Onboarding channel — proxy to runtime /onboarding/* endpoints (P1.11).
 *
 *  GET  /v1/onboarding/state    → current onboarding config
 *  POST /v1/onboarding/complete → finish day-0 onboarding
 *  POST /v1/onboarding/skip     → dismiss without launching
 */

import { Hono } from 'hono'

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
  try {
    const res = await fetch(RUNTIME_URL + path, {
      method,
      headers: method === 'POST' ? { 'content-type': 'application/json' } : undefined,
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
