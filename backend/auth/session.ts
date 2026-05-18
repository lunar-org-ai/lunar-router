/**
 * Session router — OSS-local stub.
 *
 * The OSS distribution runs single-tenant on localhost. There is no
 * login, no Firebase, no tenant Bearer exchange. The UI calls
 * `GET /v1/auth/mode` once on boot to learn that and skips its login
 * screens entirely.
 *
 * Hosted/multi-tenant deployments replace this file with the Firebase
 * + tenant-provisioning variant from the private infra repo and set
 * `OPENTRACY_MULTI_TENANT=1`.
 */
import { Hono } from 'hono'
import { isMultiTenantEnabled } from './feature'

export const authSessionRouter = new Hono()

// Public probe so the UI can decide, before any login, whether the
// deployment is OSS-local (no gate) or multi-tenant (login required).
authSessionRouter.get('/mode', (c) => {
  return c.json({ multi_tenant: isMultiTenantEnabled() })
})

// In OSS mode there is no session to mint — the UI never calls this.
// We still respond 404 explicitly so a stray client gets a clean error
// instead of hanging on a missing route.
authSessionRouter.post('/session', (c) => {
  return c.json(
    {
      error: 'sessions_disabled_in_oss_mode',
      detail:
        'This deployment runs single-tenant on localhost. Set OPENTRACY_MULTI_TENANT=1 and supply a Firebase-enabled session.ts to enable logins.',
    },
    404,
  )
})
