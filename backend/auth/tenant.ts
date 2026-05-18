/**
 * Per-tenant Bearer auth middleware (P16.1).
 *
 * Reads `Authorization: Bearer otrcy_live_<…>`, calls the runtime's
 * `/admin/tokens/resolve` to map it to a `tenant_id`, then forwards
 * that as `x-tenant-id` on the proxy fetch downstream. The runtime's
 * own tenant middleware reads the header and sets the active tenant
 * for the duration of the request.
 *
 * Only mounted when `OPENTRACY_MULTI_TENANT=1`. OSS local mode keeps
 * the single-key `apiKeyAuth` from before P16.1.
 *
 * The resolve call requires the operator admin key — `RUNTIME_ADMIN_KEY`
 * env var — so the public surface can't be used to enumerate tenants
 * by dictionary attack.
 */

import type { MiddlewareHandler } from 'hono'

const RUNTIME_URL =
  process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const RESOLVE_PATH = '/admin/tokens/resolve'

const TOKEN_PREFIX = 'otrcy_live_'

interface ResolveResponse {
  tenant_id: string
}

async function resolveTenant(token: string): Promise<string | null> {
  const adminKey = (process.env.RUNTIME_ADMIN_KEY ?? '').trim()
  if (!adminKey) {
    // Operator misconfiguration. We refuse to authenticate any tenant
    // request rather than fall open. Surface a clear log line so the
    // operator sees what's wrong on first deploy.
    console.warn(
      'tenant_auth: RUNTIME_ADMIN_KEY unset; refusing all tenant tokens',
    )
    return null
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 5_000)
  try {
    const res = await fetch(RUNTIME_URL + RESOLVE_PATH, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${adminKey}`,
      },
      body: JSON.stringify({ token }),
      signal: controller.signal,
    })
    if (!res.ok) return null
    const body = (await res.json()) as ResolveResponse
    if (!body || typeof body.tenant_id !== 'string') return null
    return body.tenant_id
  } catch (e) {
    console.warn('tenant_auth: resolve call failed:', e)
    return null
  } finally {
    clearTimeout(timeout)
  }
}

export const tenantAuth: MiddlewareHandler = async (c, next) => {
  const header = c.req.header('Authorization') ?? ''
  const match = /^Bearer\s+(.+)$/.exec(header)
  if (!match) {
    return c.json(
      { error: 'missing or malformed Authorization header' },
      401,
    )
  }
  const presented = match[1].trim()

  // Cheap shape check before the network call — tokens are well-known
  // shape, no point round-tripping garbage.
  if (!presented.startsWith(TOKEN_PREFIX)) {
    return c.json({ error: 'invalid tenant token' }, 401)
  }

  const tenantId = await resolveTenant(presented)
  if (!tenantId) {
    return c.json({ error: 'invalid tenant token' }, 401)
  }

  c.set('tenant_id', tenantId)
  c.header('x-auth-mode', 'tenant')
  c.header('x-tenant-id', tenantId)
  return next()
}
