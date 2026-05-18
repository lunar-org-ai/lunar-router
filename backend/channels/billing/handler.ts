/**
 * Billing channel — proxy to runtime /tenant/billing + Stripe checkout.
 *
 *  GET  /v1/billing            → current tenant's tier, monthly usage, limits.
 *  POST /v1/billing/checkout   → mint a Stripe Checkout URL { tier } → { url }.
 *
 * Forwards tenant headers via `proxyHeaders` so the runtime resolves
 * the active tenant from the request context. In OSS mode the GET
 * returns a synthetic "oss" tier with no caps; the checkout endpoint
 * 400s.
 *
 * The Stripe WEBHOOK is intentionally NOT proxied through here —
 * Stripe needs the raw request body for signature verification, and
 * the runtime exposes /billing/webhook directly. Route it at the
 * gateway / Cloud Run mapping layer.
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 5_000

export const billingRouter = new Hono()

billingRouter.get('/', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(`${RUNTIME_URL}/tenant/billing`, {
      headers: proxyHeaders(c),
      signal: controller.signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return c.json(
        { error: 'runtime error', detail: `${res.status}: ${text.slice(0, 200)}` },
        502,
      )
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

billingRouter.post('/checkout', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const body = await c.req.text()
    const res = await fetch(`${RUNTIME_URL}/billing/checkout`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...proxyHeaders(c),
      },
      body: body || '{}',
      signal: controller.signal,
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      return c.json(data, res.status as 400 | 503 | 502)
    }
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
