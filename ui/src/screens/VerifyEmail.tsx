/**
 * VerifyEmail — pending email-verification gate (P16.7).
 *
 * Reached automatically when:
 *   - A new user signs up with email/password (we send the link
 *     immediately and bounce here instead of into the app).
 *   - An existing password user tries to sign in but hasn't clicked
 *     the link yet.
 *
 * Google users skip this screen entirely — emailVerified is true at
 * sign-in.
 *
 * From here the user can:
 *   - Click the verification link in their inbox, come back, hit
 *     "I verified, continue" → we reload the Firebase user, see
 *     emailVerified=true, exchange for a tenant Bearer, and land in
 *     the app.
 *   - Hit "Resend email" if the first one got lost.
 *   - "Use a different account" → sign out + back to /login.
 */
import { useEffect, useState } from 'react';
import { useNavigate } from '@tanstack/react-router';
import { Icon } from '../components/Icon';
import { AuthFooter, AuthHeader, EvolutionPanel } from './AuthShared';
import { pollEmailVerified, resendVerificationEmail, signOut } from '../lib/auth';
import { firebaseAuth } from '../lib/firebase';

const RESEND_COOLDOWN_SEC = 30;

export const VerifyEmail = () => {
  const navigate = useNavigate();
  // Firebase's `currentUser` is the source of truth for which email
  // we're waiting on. If it's missing (e.g. user opened /verify-email
  // directly without signing up first), bounce them back to /login.
  const [email, setEmail] = useState<string | null>(() => firebaseAuth().currentUser?.email ?? null);
  const [checking, setChecking] = useState(false);
  const [resending, setResending] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  useEffect(() => {
    if (!email) {
      // Race: Firebase may finish initializing slightly after the
      // route mounts. Listen for the first auth-state transition
      // and either pick up the email or bounce.
      const unsub = firebaseAuth().onAuthStateChanged((user) => {
        if (user?.email) setEmail(user.email);
        else navigate({ to: '/login', replace: true });
        unsub();
      });
      return () => unsub();
    }
  }, [email, navigate]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = window.setInterval(() => setCooldown((n) => Math.max(0, n - 1)), 1000);
    return () => window.clearInterval(t);
  }, [cooldown]);

  // Light background polling: every 6s, ping Firebase to see if the
  // user clicked the link in another tab. Most users will click
  // "I verified, continue" first, but this is forgiving.
  useEffect(() => {
    const t = window.setInterval(async () => {
      const session = await pollEmailVerified().catch(() => null);
      if (session) navigate({ to: '/', replace: true });
    }, 6000);
    return () => window.clearInterval(t);
  }, [navigate]);

  const handleContinue = async () => {
    setChecking(true);
    setStatusError(null);
    setInfo(null);
    try {
      const session = await pollEmailVerified();
      if (session) {
        navigate({ to: '/', replace: true });
        return;
      }
      setStatusError('We don’t see a verification yet. Click the link in your inbox, then try again.');
    } catch (err) {
      setStatusError((err as Error)?.message ?? 'Could not check verification. Try again.');
    } finally {
      setChecking(false);
    }
  };

  const handleResend = async () => {
    setResending(true);
    setStatusError(null);
    setInfo(null);
    try {
      await resendVerificationEmail();
      setInfo('Verification email sent. Check your inbox (and spam).');
      setCooldown(RESEND_COOLDOWN_SEC);
    } catch (err) {
      setStatusError((err as Error)?.message ?? 'Could not resend. Try again in a minute.');
    } finally {
      setResending(false);
    }
  };

  const handleSwitchAccount = async () => {
    await signOut();
    navigate({ to: '/login', replace: true });
  };

  return (
    <div className="auth-page">
      <div className="auth-split">
        <div className="auth-split-left">
          <AuthHeader
            swapTo="/login"
            swapPrompt="Wrong account?"
            swapCta="Sign in as someone else"
          />

          <main className="auth-main">
            <div className="auth-card">
              <h1 className="auth-h1">Verify your email</h1>
              <p className="auth-sub">
                We sent a verification link to{' '}
                <strong>{email ?? 'your email'}</strong>. Click it, then come
                back and tap continue.
              </p>

              {statusError && (
                <div className="auth-alert" role="alert">
                  <Icon name="warn" size={14} />
                  <span>{statusError}</span>
                </div>
              )}
              {info && (
                <div className="auth-hint" style={{ color: 'var(--accent-fg)', marginBottom: 14 }}>
                  {info}
                </div>
              )}

              <button
                type="button"
                className={`auth-submit ${checking ? 'is-loading' : ''}`}
                onClick={handleContinue}
                disabled={checking || resending}
              >
                {checking ? <><span className="auth-spinner" /> Checking…</> : <>I verified, continue</>}
              </button>

              <button
                type="button"
                className="auth-google"
                onClick={handleResend}
                disabled={resending || checking || cooldown > 0}
                style={{ marginTop: 12 }}
              >
                {resending
                  ? <><span className="auth-spinner" style={{ borderTopColor: 'var(--fg)', borderColor: 'oklch(0.18 0.005 80 / 0.18)' }} /> Resending…</>
                  : cooldown > 0
                    ? <>Resend in {cooldown}s</>
                    : <>Resend verification email</>}
              </button>

              <div className="auth-foot">
                <button
                  type="button"
                  onClick={handleSwitchAccount}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--fg)',
                    font: 'inherit',
                    fontWeight: 500,
                    cursor: 'pointer',
                    textDecoration: 'underline',
                  }}
                >
                  Use a different account
                </button>
              </div>
            </div>
          </main>

          <AuthFooter />
        </div>
        <aside className="auth-split-right">
          <EvolutionPanel variant="login" />
        </aside>
      </div>
    </div>
  );
};
