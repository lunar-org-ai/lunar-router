/**
 * Router channel — proxy to runtime /router/* endpoints (P15.3).
 *
 *  GET  /v1/router/config   → current router_config metadata; cold-start safe.
 *  POST /v1/router/decide   → score a prompt; no LLM call.
 *
 *  P15.3.8 will add PUT /config for manual λ overrides via the AHE
 *  record_manual_change pipeline. Not in this channel yet.
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

export const routerRouter = new Hono()

routerRouter.get('/config', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/router/config', { signal: controller.signal })
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
})

routerRouter.put('/config', async (c) => {
  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/router/config', {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    // 400 (validation), 409 (cold-start), 500 (router_config_invalid) pass through
    // so the UI can render reasons directly.
    if (res.status === 400 || res.status === 409 || res.status === 422 || res.status === 500) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 400 | 409 | 422 | 500)
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
})

routerRouter.post('/decide', async (c) => {
  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/router/decide', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    // 503 (cold-start) and 400/422 (validation) are pass-through so the UI
    // can render "no fitted config" / form errors directly.
    if (res.status === 400 || res.status === 422 || res.status === 503) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 400 | 422 | 503)
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
})
