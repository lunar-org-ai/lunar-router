/**
 * Policy channel — proxy to runtime /policy endpoint.
 *
 *  GET /v1/policy  → current approval policy (mode + thresholds)
 *  PUT /v1/policy  → update approval policy YAML
 */

import { Hono } from 'hono'
import { proxyHeaders } from '../../auth/proxy_headers'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

export const policyRouter = new Hono()

policyRouter.get('/', async (c) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/policy', { headers: proxyHeaders(c), signal: controller.signal })
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

policyRouter.put('/', async (c) => {
  let body: unknown = {}
  try {
    body = await c.req.json()
  } catch {
    body = {}
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(RUNTIME_URL + '/policy', {
      method: 'PUT',
      headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (res.status === 400) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, 400)
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
