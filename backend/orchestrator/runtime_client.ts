/**
 * Typed client for the Python runtime service.
 *
 * The runtime exposes POST /run, GET /health, GET /agent. This module is the
 * only place in the backend that knows the runtime's URL or wire format —
 * everything else imports `runAgent` and `checkRuntimeHealth` from here.
 */

import { z } from 'zod'

const StageOutcomeSchema = z.object({
  stage: z.string(),
  technique: z.string(),
  variant: z.string(),
  duration_ms: z.number(),
  docs_in: z.number(),
  docs_out: z.number(),
  routing_model: z.string().nullable(),
  error: z.string().nullable(),
})

const RunResponseSchema = z.object({
  response: z.string().nullable(),
  trace_id: z.string(),
  duration_ms: z.number(),
  success: z.boolean(),
  error: z.string().nullable(),
  agent_version: z.string().nullable(),
  stages: z.array(StageOutcomeSchema),
})

const HealthResponseSchema = z.object({
  status: z.string(),
  agent_version: z.string().nullable(),
})

export interface HistoryMessage {
  role: string
  content: string
}

export type RunResponse = z.infer<typeof RunResponseSchema>
export type HealthResponse = z.infer<typeof HealthResponseSchema>

const DEFAULT_RUNTIME_URL = 'http://127.0.0.1:8001'
const DEFAULT_TIMEOUT_MS = 30_000

export class RuntimeError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public override readonly cause?: unknown,
  ) {
    super(message)
    this.name = 'RuntimeError'
  }
}

interface ClientOptions {
  runtimeUrl?: string
  timeoutMs?: number
}

function resolveUrl(opts: ClientOptions): string {
  return opts.runtimeUrl ?? process.env.RUNTIME_URL ?? DEFAULT_RUNTIME_URL
}

export async function runAgent(
  request: string,
  history: HistoryMessage[] | undefined = undefined,
  opts: ClientOptions = {},
): Promise<RunResponse> {
  const url = resolveUrl(opts) + '/run'
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ request, history }),
      signal: controller.signal,
    })

    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new RuntimeError(
        `runtime returned ${res.status}: ${body.slice(0, 200)}`,
        res.status,
      )
    }

    const json: unknown = await res.json()
    return RunResponseSchema.parse(json)
  } catch (e) {
    if (e instanceof RuntimeError) throw e
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new RuntimeError(`runtime call timed out after ${timeoutMs}ms`, undefined, e)
    }
    throw new RuntimeError(
      `runtime call failed: ${e instanceof Error ? e.message : String(e)}`,
      undefined,
      e,
    )
  } finally {
    clearTimeout(timer)
  }
}

export async function checkRuntimeHealth(opts: ClientOptions = {}): Promise<HealthResponse> {
  const url = resolveUrl(opts) + '/health'
  const res = await fetch(url)
  if (!res.ok) {
    throw new RuntimeError(`runtime health check returned ${res.status}`, res.status)
  }
  const json: unknown = await res.json()
  return HealthResponseSchema.parse(json)
}
