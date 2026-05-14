/**
 * Multi-tenant feature gate (P16.1).
 *
 * Mirrors `runtime/tenants/feature.py`. OSS local distribution keeps
 * the single-key auth via `apiKeyAuth`. Hosted/infra deploys set
 * `OPENTRACY_MULTI_TENANT=1` and get the per-tenant Bearer flow with
 * an admin token for `/v1/admin/*`.
 *
 * Read fresh on every call so tests can flip via `process.env`.
 */

export function isMultiTenantEnabled(): boolean {
  return (process.env.OPENTRACY_MULTI_TENANT ?? '').trim() === '1'
}
