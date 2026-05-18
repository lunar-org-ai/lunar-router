/**
 * Global fetch interceptor (P16.7).
 *
 * Wraps window.fetch once at app boot so every same-origin `/v1/*` call
 * automatically carries the tenant Bearer issued by /v1/auth/session.
 * api.ts has 60+ fetch callsites and we don't want to touch each one;
 * the wrapper does it transparently.
 *
 * Skipped automatically:
 *   - `/v1/auth/session` itself — that's the public exchange endpoint
 *     and forcing a Bearer would be circular.
 *   - Any request that already sets an Authorization header.
 *
 * 401 from a /v1/* call short-circuits the session: we clear the
 * Bearer and bounce the user back to /login. Without this, expired
 * tokens silently render every screen empty.
 */
import { getBearer, signOut } from './auth';

let installed = false;

export function installAuthFetch(): void {
  if (installed) return;
  installed = true;

  const original = window.fetch.bind(window);

  const isInternalApi = (url: string) =>
    url.startsWith('/v1/') && !url.startsWith('/v1/auth/session');

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
    if (!isInternalApi(url)) {
      return original(input, init);
    }
    const bearer = getBearer();
    if (!bearer) {
      return original(input, init);
    }
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
    if (!headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${bearer}`);
    }
    const res = await original(input, { ...init, headers });
    // 401s on /v1/admin/* are EXPECTED for tenant users — those routes
    // gate on the operator's BACKEND_API_KEYS, and a regular tenant
    // Bearer is always rejected. Treating that as "stale bearer" would
    // sign the user out the moment getFeatures() or any other admin
    // probe ran. We only sign-out on 401s from tenant-scoped routes,
    // where 401 actually means the Bearer is stale.
    const isAdminPath = url.startsWith('/v1/admin/');
    if (res.status === 401 && !isAdminPath) {
      void signOut().then(() => {
        if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
          window.location.assign('/login');
        }
      });
    }
    return res;
  };
}
