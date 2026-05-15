/**
 * /v1/auth/session — Firebase ID token → tenant Bearer exchange (P16.7).
 *
 * Flow:
 *   1. Client signs in with Firebase Auth (email/password or Google
 *      popup), grabs the resulting ID token JWT.
 *   2. POST /v1/auth/session { idToken } — no Bearer required (this is
 *      the endpoint that *issues* one).
 *   3. We verify the ID token with firebase-admin. Cloud Run picks up
 *      Application Default Credentials from its service account; locally,
 *      GOOGLE_APPLICATION_CREDENTIALS works too. We never trust the
 *      raw email/uid from the request body.
 *   4. We look up a tenant by slug = `fb_<sanitized-uid>`. If missing,
 *      we create one via the runtime admin API.
 *   5. We mint a fresh Bearer (`otrcy_live_...`) on every login. Stale
 *      tokens stay valid until revoked separately — the trade-off here
 *      is that `mint_token` doesn't return plaintext for existing
 *      tokens, so re-issuing is the only way to hand the client a
 *      usable Bearer.
 *   6. Response: { bearer, tenant_id, email, name }.
 *
 * Errors:
 *   401 if the ID token is missing/invalid/expired
 *   502 if the runtime admin API is unreachable
 *   500 for anything else
 */
import { Hono } from 'hono'
import admin from 'firebase-admin'
import { isMultiTenantEnabled } from './feature'

const RUNTIME_URL = process.env.RUNTIME_URL ?? 'http://127.0.0.1:8001'
const RUNTIME_ADMIN_KEY = process.env.RUNTIME_ADMIN_KEY ?? ''

let initialized = false
function ensureFirebase(): void {
  if (initialized) return
  if (admin.apps.length === 0) {
    // applicationDefault() resolves via:
    //   - Cloud Run: the attached service account (no env needed)
    //   - Local dev: GOOGLE_APPLICATION_CREDENTIALS pointing at a JSON key
    admin.initializeApp({
      credential: admin.credential.applicationDefault(),
      projectId: process.env.FIREBASE_PROJECT_ID,
    })
  }
  initialized = true
}

interface SessionRequest {
  idToken?: string
}

function sanitizeUid(uid: string): string {
  // Tenant slugs accept [a-z0-9-_]; the Firebase uid is alnum already
  // but we lowercase-and-strip just to be safe across providers.
  return uid.replace(/[^a-z0-9_-]/gi, '').toLowerCase()
}

// Runtime's tenant summary shape — `id` is the canonical key (when a
// slug is passed to POST /admin/tenants, the runtime uses it as the
// tenant_id, so they're equal in our flow).
interface RuntimeTenant {
  id: string
  name: string
}

async function runtimeFetch(path: string, init?: RequestInit): Promise<Response> {
  if (!RUNTIME_ADMIN_KEY) {
    throw new Error('RUNTIME_ADMIN_KEY not configured on backend')
  }
  const headers = new Headers(init?.headers)
  headers.set('Authorization', `Bearer ${RUNTIME_ADMIN_KEY}`)
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return fetch(`${RUNTIME_URL}${path}`, { ...init, headers })
}

async function findTenantById(id: string): Promise<RuntimeTenant | null> {
  const res = await runtimeFetch('/admin/tenants')
  if (!res.ok) {
    throw new Error(`runtime GET /admin/tenants → ${res.status}`)
  }
  const body = (await res.json()) as { tenants: RuntimeTenant[] }
  return body.tenants.find((t) => t.id === id) ?? null
}

async function createTenant(input: {
  name: string
  slug: string
  description: string
}): Promise<RuntimeTenant> {
  const res = await runtimeFetch('/admin/tenants', {
    method: 'POST',
    body: JSON.stringify(input),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`runtime POST /admin/tenants → ${res.status}: ${detail}`)
  }
  return (await res.json()) as RuntimeTenant
}

async function mintToken(tenantId: string, label: string): Promise<string> {
  const res = await runtimeFetch(`/admin/tenants/${tenantId}/tokens`, {
    method: 'POST',
    body: JSON.stringify({ label }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`runtime POST /admin/tenants/${tenantId}/tokens → ${res.status}: ${detail}`)
  }
  const body = (await res.json()) as { token: string }
  return body.token
}

export const authSessionRouter = new Hono()

// Public probe so the UI can tell, before any login, whether the
// deployment is OSS-local (no gate) or multi-tenant (Firebase login
// required). Lives under /v1/auth/* because that prefix already
// bypasses every auth middleware.
authSessionRouter.get('/mode', (c) => {
  return c.json({ multi_tenant: isMultiTenantEnabled() })
})

authSessionRouter.post('/session', async (c) => {
  let body: SessionRequest
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'invalid_json' }, 400)
  }
  if (!body.idToken || typeof body.idToken !== 'string') {
    return c.json({ error: 'missing_id_token' }, 400)
  }

  ensureFirebase()
  let decoded: admin.auth.DecodedIdToken
  try {
    decoded = await admin.auth().verifyIdToken(body.idToken)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    return c.json({ error: 'invalid_id_token', detail: msg }, 401)
  }

  const uid = decoded.uid
  const email = decoded.email ?? `${uid}@firebase`
  const name = (decoded as { name?: string }).name ?? email.split('@')[0]
  const slug = `fb-${sanitizeUid(uid)}`

  try {
    let tenant = await findTenantById(slug)
    if (!tenant) {
      tenant = await createTenant({
        name: email,
        slug,
        description: `Auto-provisioned from Firebase Auth (uid=${uid})`,
      })
    }
    const bearer = await mintToken(tenant.id, `session-${Date.now()}`)
    return c.json({
      bearer,
      tenant_id: tenant.id,
      email,
      name,
    })
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    return c.json({ error: 'tenant_provision_failed', detail: msg }, 502)
  }
})
