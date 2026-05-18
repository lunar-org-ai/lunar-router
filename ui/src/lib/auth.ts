/**
 * Auth session — Firebase Auth front-end + tenant Bearer exchange (P16.7).
 *
 * The flow:
 *   1. The user signs in with Firebase (email/password or Google popup).
 *   2. We grab the Firebase ID token and POST it to /v1/auth/session.
 *   3. The backend verifies the token, looks up (or creates) a tenant
 *      tied to the Firebase uid, mints a fresh tenant Bearer
 *      (otrcy_live_*), and returns it.
 *   4. We cache the Bearer in sessionStorage and the rest of the app
 *      uses it as the Authorization header for every /v1/* call.
 *
 * Bearer is in sessionStorage (not localStorage) — clears on tab close,
 * which limits the blast radius if the JWT leaks. The Firebase ID
 * token itself never leaves Firebase's SDK; we only persist what the
 * backend gave us back.
 */

import {
  createUserWithEmailAndPassword,
  getRedirectResult,
  sendEmailVerification,
  signInWithEmailAndPassword,
  signInWithRedirect,
  signOut as fbSignOut,
  updateProfile,
  type User,
} from 'firebase/auth';
import { firebaseAuth, googleProvider } from './firebase';

const KEY = 'opentracy.session.v2';

export interface AuthSession {
  bearer: string;
  tenantId: string;
  email: string;
  name: string;
  /** ISO timestamp; lets us decide later whether to expire stale sessions. */
  signedInAt: string;
}

export function getSession(): AuthSession | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (typeof parsed?.bearer !== 'string') return null;
    return parsed as AuthSession;
  } catch {
    return null;
  }
}

export function isAuthed(): boolean {
  return getSession() !== null;
}

export function getBearer(): string | null {
  return getSession()?.bearer ?? null;
}

/**
 * Server-reported deployment mode. The login gate only triggers in
 * `multi_tenant`. In OSS the backend has no auth, so the shell renders
 * straight to Evolution without a session.
 *
 * Public endpoint — no Bearer required (lives under /v1/auth/* which
 * bypasses every auth middleware in the backend).
 */
export interface AuthMode {
  multi_tenant: boolean;
}

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

interface SessionResponse {
  bearer: string;
  tenant_id: string;
  email: string;
  name: string;
}

async function exchangeForBearer(user: User): Promise<AuthSession> {
  const idToken = await user.getIdToken(/* forceRefresh */ true);
  const res = await fetch('/v1/auth/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idToken }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`auth/session failed (${res.status}): ${detail || res.statusText}`);
  }
  const body = (await res.json()) as SessionResponse;
  const session: AuthSession = {
    bearer: body.bearer,
    tenantId: body.tenant_id,
    email: body.email,
    name: body.name,
    signedInAt: new Date().toISOString(),
  };
  sessionStorage.setItem(KEY, JSON.stringify(session));
  return session;
}

/** Thrown when a password user signs in/up without verifying their
 *  email yet. Carries the Firebase User so callers can render a
 *  resend prompt instead of bouncing to /login. */
export class EmailNotVerifiedError extends Error {
  constructor(public readonly user: User) {
    super('email_not_verified');
    this.name = 'EmailNotVerifiedError';
  }
}

function isPasswordProvider(user: User): boolean {
  return user.providerData.some((p) => p.providerId === 'password');
}

export async function signInWithEmail(email: string, password: string): Promise<AuthSession> {
  const cred = await signInWithEmailAndPassword(firebaseAuth(), email, password);
  // Force a refresh of the Firebase profile so a verification that
  // happened on another device shows up here without the user having
  // to sign out and back in.
  await cred.user.reload();
  if (isPasswordProvider(cred.user) && !cred.user.emailVerified) {
    throw new EmailNotVerifiedError(cred.user);
  }
  return exchangeForBearer(cred.user);
}

/** When Firebase sends the verification email, this is where the
 *  user lands after clicking the link. Pointing it at our custom
 *  domain (`app.dev.opentracy.cloud`) instead of `firebaseapp.com`
 *  keeps the brand consistent end-to-end. Firebase appends the
 *  oobCode + mode params to this URL when the user completes the
 *  action. */
function verificationActionSettings() {
  return {
    url: `${window.location.origin}/login`,
    handleCodeInApp: false,
  };
}

export async function signUpWithEmail(
  email: string,
  password: string,
  displayName?: string,
): Promise<never> {
  const cred = await createUserWithEmailAndPassword(firebaseAuth(), email, password);
  if (displayName?.trim()) {
    await updateProfile(cred.user, { displayName: displayName.trim() });
  }
  // Fire-and-forget the verification email. Failure here is non-fatal —
  // we'll let the user resend from the verify screen.
  try {
    await sendEmailVerification(cred.user, verificationActionSettings());
  } catch (e) {
    console.warn('sendEmailVerification failed', e);
  }
  // Block entry until they click the link. Sign-up always lands the
  // user on /verify-email — caller throws + redirects.
  throw new EmailNotVerifiedError(cred.user);
}

/** Re-issues the verification email. The user must be signed in to
 *  Firebase (i.e. came from signInWithEmail / signUpWithEmail or a
 *  prior session). */
export async function resendVerificationEmail(): Promise<void> {
  const user = firebaseAuth().currentUser;
  if (!user) {
    throw new Error('not_signed_in');
  }
  await sendEmailVerification(user, verificationActionSettings());
}

/** Checks if the current Firebase user has verified their email since
 *  the last reload. On success, exchanges for a Bearer + returns the
 *  session; otherwise returns null and leaves the user on the verify
 *  screen. */
export async function pollEmailVerified(): Promise<AuthSession | null> {
  const user = firebaseAuth().currentUser;
  if (!user) return null;
  await user.reload();
  if (!user.emailVerified) return null;
  return exchangeForBearer(user);
}

/** Kicks off Google sign-in via a full-page redirect (popup flows
 *  are blocked by Safari + many other browsers on first interaction).
 *  Resolution lands in consumeGoogleRedirect() after Firebase brings
 *  the user back. */
export async function signInWithGoogle(): Promise<void> {
  await signInWithRedirect(firebaseAuth(), googleProvider);
}

/** Call once at app boot. If the page-load was the return-leg of a
 *  Google redirect, this exchanges the resulting user for a tenant
 *  Bearer and returns the session. Otherwise returns null. */
export async function consumeGoogleRedirect(): Promise<
  { session: AuthSession; error?: undefined } | { session?: undefined; error: string } | null
> {
  try {
    const result = await getRedirectResult(firebaseAuth());
    if (!result?.user) return null;
    const session = await exchangeForBearer(result.user);
    return { session };
  } catch (err) {
    console.error('consumeGoogleRedirect failed', err);
    return { error: readableAuthError(err) };
  }
}

export async function signOut(): Promise<void> {
  sessionStorage.removeItem(KEY);
  try {
    await fbSignOut(firebaseAuth());
  } catch {
    // Firebase sign-out can fail if the app isn't init'd in this tab;
    // we've already cleared the local session, so it's fine.
  }
}

/** Maps Firebase Auth error codes to user-readable strings. Keep these
 *  short; the auth screens render them inline next to the form. */
export function readableAuthError(err: unknown): string {
  const code = (err as { code?: string })?.code ?? '';
  switch (code) {
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return "That email and password don't match. Try again or reset your password.";
    case 'auth/invalid-email':
      return 'That doesn’t look like a valid email.';
    case 'auth/email-already-in-use':
      return 'An account with that email already exists. Try signing in instead.';
    case 'auth/weak-password':
      return 'That password is too weak. Use at least 8 characters with mixed case + a number.';
    case 'auth/popup-closed-by-user':
    case 'auth/cancelled-popup-request':
      return 'Sign-in window closed. Try again when you’re ready.';
    case 'auth/operation-not-allowed':
      return 'This sign-in method isn’t enabled yet. Contact your operator.';
    case 'auth/network-request-failed':
      return 'Network error. Check your connection and try again.';
    case 'auth/too-many-requests':
      return 'Too many attempts. Wait a moment before trying again.';
    default:
      if (err instanceof EmailNotVerifiedError) {
        return 'Check your email to verify your account before signing in.';
      }
      return (err as Error)?.message ?? 'Sign-in failed. Try again.';
  }
}
