/**
 * Helper to thread the tenant context through runtime proxy fetches.
 *
 * Channel handlers proxy incoming /v1/* requests to the python
 * runtime. When `OPENTRACY_MULTI_TENANT=1`, `tenantAuth` middleware
 * stashes the resolved `tenant_id` on the Hono context; this helper
 * surfaces it as an `x-tenant-id` header object so callers can spread
 * it into their fetch headers without each one re-reading the
 * context.
 *
 * In OSS mode the helper returns an empty object — the proxy sends
 * nothing extra and the runtime middleware (also flag-gated) does
 * nothing either, preserving legacy behavior.
 *
 * Usage:
 *
 *     const res = await fetch(url, {
 *       method: 'POST',
 *       headers: { 'content-type': 'application/json', ...proxyHeaders(c) },
 *       body: JSON.stringify(payload),
 *     })
 */

import type { Context } from 'hono'

export function proxyHeaders(c: Context): Record<string, string> {
  const headers: Record<string, string> = {}
  const tenantId = c.get('tenant_id')
  if (typeof tenantId === 'string' && tenantId.length > 0) {
    headers['x-tenant-id'] = tenantId
  }
  return headers
}
