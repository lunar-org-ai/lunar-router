/**
 * Evals channel — proxy to runtime /evals/*.
 *
 *  GET /v1/evals/suites               → list of eval suites
 *  GET /v1/evals/suites/:name         → suite detail (goldens + rubrics)
 *  GET /v1/evals/reports              → recent runs (filterable)
 *  GET /v1/evals/reports/:report_id   → single report with cases
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000

const proxy = (path: string, handle404 = false) => async (c: import('hono').Context) => {
  const qs = c.req.url.split('?')[1] ?? ''
  const url = qs ? `${RUNTIME_URL}${path}?${qs}` : `${RUNTIME_URL}${path}`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(url, { signal: controller.signal })
    if (handle404 && (res.status === 404 || res.status === 400)) {
      const data = await res.json().catch(() => ({}))
      return c.json(data, res.status as 404 | 400)
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
}

export const evalsRouter = new Hono()

evalsRouter.get('/suites', proxy('/evals/suites'))
evalsRouter.get('/suites/:name', async (c) => {
  const name = c.req.param('name')
  return proxy(`/evals/suites/${encodeURIComponent(name)}`, true)(c)
})
evalsRouter.get('/reports', proxy('/evals/reports'))
evalsRouter.get('/reports/:id', async (c) => {
  const id = c.req.param('id')
  return proxy(`/evals/reports/${encodeURIComponent(id)}`, true)(c)
})
