/**
 * API key auth middleware.
 *
 * Reads `BACKEND_API_KEYS` from env (comma-separated) and accepts any request
 * whose Authorization header is `Bearer <key>` for one of those keys.
 *
 * If `BACKEND_API_KEYS` is unset or empty, the middleware lets every request
 * through (dev mode). Production deployments should always set the env var.
 */

import type { MiddlewareHandler } from 'hono'

function loadKeys(): Set<string> {
  const raw = process.env.BACKEND_API_KEYS ?? ''
  const keys = raw
    .split(',')
    .map((k) => k.trim())
    .filter((k) => k.length > 0)
  return new Set(keys)
}

export const apiKeyAuth: MiddlewareHandler = async (c, next) => {
  const keys = loadKeys()

  if (keys.size === 0) {
    // Dev mode: no keys configured → allow all. Logged once per request.
    c.header('x-auth-mode', 'dev-no-keys')
    return next()
  }

  const header = c.req.header('Authorization') ?? ''
  const match = /^Bearer\s+(.+)$/.exec(header)
  if (!match) {
    return c.json({ error: 'missing or malformed Authorization header' }, 401)
  }

  const presented = match[1].trim()
  if (!keys.has(presented)) {
    return c.json({ error: 'invalid api key' }, 401)
  }

  c.header('x-auth-mode', 'api-key')
  return next()
}
