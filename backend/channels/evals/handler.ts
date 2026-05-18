/**
 * Evals channel — proxy to runtime /evals/*.
 *
 *  GET  /v1/evals/suites                                    → list of eval suites
 *  GET  /v1/evals/suites/:name                              → suite detail (goldens + rubrics)
 *  POST /v1/evals/suites/:name/run                          → run a suite synchronously
 *  POST /v1/evals/run_all                                   → run every suite in series
 *  GET  /v1/evals/reports                                   → recent runs (filterable)
 *  GET  /v1/evals/reports/:report_id                        → single report with cases
 *  POST /v1/evals/goldens/promote-from-trace/:trace_id      → promote a trace to a golden (P16.1)
 */

import { Hono } from 'hono'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const TIMEOUT_MS = 15_000
// Stub-LLM suites finish in ms, but we leave a long ceiling for when P1.9 lands.
const RUN_TIMEOUT_MS = 120_000

const proxy = (path: string, handle404 = false, timeoutMs = TIMEOUT_MS) =>
  async (c: import('hono').Context) => {
    const qs = c.req.url.split('?')[1] ?? ''
    const url = qs ? `${RUNTIME_URL}${path}?${qs}` : `${RUNTIME_URL}${path}`
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
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

const proxyPost = (
  path: string,
  opts: { timeoutMs?: number; forwardBody?: boolean } = {},
) =>
  async (c: import('hono').Context) => {
    const url = `${RUNTIME_URL}${path}`
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? RUN_TIMEOUT_MS)

    let bodyText = ''
    if (opts.forwardBody) {
      // Read raw body so empty/invalid JSON still proxies cleanly.
      try {
        bodyText = await c.req.text()
      } catch {
        bodyText = ''
      }
    }

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: opts.forwardBody ? bodyText || '{}' : undefined,
        signal: controller.signal,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        return c.json(data, res.status as 400 | 404 | 500 | 502)
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
  }

export const evalsRouter = new Hono()

evalsRouter.get('/suites', proxy('/evals/suites'))
evalsRouter.get('/suites/:name', async (c) => {
  const name = c.req.param('name')
  return proxy(`/evals/suites/${encodeURIComponent(name)}`, true)(c)
})
evalsRouter.post('/suites/:name/run', async (c) => {
  const name = c.req.param('name')
  return proxyPost(`/evals/suites/${encodeURIComponent(name)}/run`)(c)
})
evalsRouter.post('/run_all', proxyPost('/evals/run_all'))
evalsRouter.get('/reports', proxy('/evals/reports'))
evalsRouter.get('/reports/:id', async (c) => {
  const id = c.req.param('id')
  return proxy(`/evals/reports/${encodeURIComponent(id)}`, true)(c)
})
evalsRouter.post('/goldens/promote-from-trace/:traceId', async (c) => {
  const tid = c.req.param('traceId')
  return proxyPost(`/evals/goldens/promote-from-trace/${encodeURIComponent(tid)}`, {
    forwardBody: true,
    timeoutMs: 15_000,
  })(c)
})
