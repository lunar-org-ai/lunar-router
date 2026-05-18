/**
 * Auth session — OSS-local stub.
 *
 * The OSS distribution runs single-tenant on localhost with no login
 * gate (backend's `apiKeyAuth` dev-mode passes every request when
 * `BACKEND_API_KEYS` is unset). This module keeps the same surface the
 * rest of the UI consumed in the multi-tenant variant — `AuthSession`,
 * `getSession`, `isAuthed`, `getAuthMode`, `signOut`,
 * `consumeGoogleRedirect` — but they all resolve to a fixed local user
 * synchronously and never touch Firebase.
 *
 * Hosted/multi-tenant deployments swap this file out from the private
 * infra repo and set `OPENTRACY_MULTI_TENANT=1` on the backend.
 */

const LOCAL_SESSION: AuthSession = {
  bearer: '',
  tenantId: 'local',
  email: 'local@opentracy',
  name: 'Local user',
  signedInAt: new Date(0).toISOString(),
};

export interface AuthSession {
  bearer: string;
  tenantId: string;
  email: string;
  name: string;
  /** ISO timestamp; kept for shape parity with the multi-tenant variant. */
  signedInAt: string;
}

export function getSession(): AuthSession | null {
  return LOCAL_SESSION;
}

export function isAuthed(): boolean {
  return true;
}

export function getBearer(): string | null {
  return null;
}

export interface AuthMode {
  multi_tenant: boolean;
}

/**
 * Probes /v1/auth/mode so the rest of the shell can ask the backend
 * (instead of the build) which mode it is in. In OSS the stub backend
 * returns `multi_tenant: false`; if the probe fails for any reason we
 * also fall back to false so the shell renders rather than hanging.
 */
export async function getAuthMode(): Promise<AuthMode> {
  try {
    const res = await fetch('/v1/auth/mode');
    if (!res.ok) return { multi_tenant: false };
    const body = (await res.json()) as Partial<AuthMode>;
    return { multi_tenant: body.multi_tenant === true };
  } catch {
    return { multi_tenant: false };
  }
}

export async function signOut(): Promise<void> {
  // No-op in OSS: no real session to clear.
}

export async function consumeGoogleRedirect(): Promise<
  { session: AuthSession; error?: undefined } | { session?: undefined; error: string } | null
> {
  return null;
}
