/**
 * Login screen (P16.7).
 *
 * Real auth: Firebase Auth (email/password + Google popup), then
 * /v1/auth/session exchanges the Firebase ID token for a tenant Bearer
 * the rest of the app uses on every /v1/* call. See lib/auth.ts.
 */
import { useEffect, useState, type FormEvent } from 'react';
import { useNavigate } from '@tanstack/react-router';
import { Icon } from '../components/Icon';
import {
  AuthFooter,
  AuthHeader,
  EvolutionPanel,
  GoogleG,
  isEmail,
} from './AuthShared';
import {
  EmailNotVerifiedError,
  readableAuthError,
  signInWithEmail,
  signInWithGoogle,
} from '../lib/auth';

interface LoginErrors {
  email?: string;
  password?: string;
}

export const Login = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [errors, setErrors] = useState<LoginErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  // Surface any error left behind by a failed Google redirect — see
  // consumeGoogleRedirect() in __root.tsx. The key is one-shot; we
  // clear it the moment we render it so a hard refresh doesn't
  // re-show the same banner forever.
  const [authError, setAuthError] = useState<string | null>(() => {
    const stored = sessionStorage.getItem('opentracy.auth.lastError');
    if (stored) sessionStorage.removeItem('opentracy.auth.lastError');
    return stored;
  });
  const [touched, setTouched] = useState<{ email?: boolean; password?: boolean }>({});

  // Re-validate on change once a field has been touched.
  useEffect(() => {
    if (!touched.email && !touched.password) return;
    const errs: LoginErrors = {};
    if (touched.email) {
      if (!email.trim()) errs.email = 'Enter your email.';
      else if (!isEmail(email)) errs.email = 'That doesn’t look like a valid email.';
    }
    if (touched.password && !password) errs.password = 'Enter your password.';
    setErrors(errs);
  }, [email, password, touched]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const errs: LoginErrors = {};
    if (!email.trim()) errs.email = 'Enter your email.';
    else if (!isEmail(email)) errs.email = 'That doesn’t look like a valid email.';
    if (!password) errs.password = 'Enter your password.';
    setTouched({ email: true, password: true });
    setErrors(errs);
    setAuthError(null);
    if (Object.keys(errs).length) return;

    setSubmitting(true);
    try {
      await signInWithEmail(email, password);
      navigate({ to: '/' });
    } catch (err) {
      if (err instanceof EmailNotVerifiedError) {
        navigate({ to: '/verify-email' });
        return;
      }
      setAuthError(readableAuthError(err));
    } finally {
      setSubmitting(false);
    }
  };

  const googleSignIn = async () => {
    setGoogleLoading(true);
    setAuthError(null);
    try {
      // Full-page redirect; control returns to consumeGoogleRedirect()
      // in __root.tsx on the way back. We don't reset googleLoading
      // because the page is navigating away.
      await signInWithGoogle();
    } catch (err) {
      setAuthError(readableAuthError(err));
      setGoogleLoading(false);
    }
  };

  const disabled = submitting || googleLoading;

  return (
    <div className="auth-page">
      <div className="auth-split">
        <div className="auth-split-left">
          <AuthHeader
            swapTo="/register"
            swapPrompt="Don't have an account?"
            swapCta="Create one"
          />

          <main className="auth-main">
            <div className="auth-card">
              <h1 className="auth-h1">Sign in</h1>
              <p className="auth-sub">Welcome back. Pick up where your agents left off.</p>

              {authError && (
                <div className="auth-alert" role="alert">
                  <Icon name="warn" size={14}/>
                  <span>{authError}</span>
                </div>
              )}

              <button
                type="button"
                className="auth-google"
                onClick={googleSignIn}
                disabled={disabled}
              >
                {googleLoading
                  ? <><span className="auth-spinner" style={{ borderTopColor: 'var(--fg)', borderColor: 'oklch(0.18 0.005 80 / 0.18)' }}/> Signing in with Google…</>
                  : <><GoogleG/> Continue with Google</>}
              </button>

              <div className="auth-divider">or</div>

              <form className="auth-form" onSubmit={submit} noValidate>
                <div className="auth-field">
                  <label className="auth-label" htmlFor="login-email">Email</label>
                  <div className="auth-input-wrap">
                    <input
                      id="login-email"
                      type="email"
                      autoComplete="email"
                      className={`auth-input ${errors.email ? 'has-error' : ''}`}
                      placeholder="you@company.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      onBlur={() => setTouched((t) => ({ ...t, email: true }))}
                      disabled={disabled}
                    />
                  </div>
                  {errors.email && (
                    <div className="auth-field-error">
                      <Icon name="warn" size={12}/> {errors.email}
                    </div>
                  )}
                </div>

                <div className="auth-field">
                  <div className="auth-field-row">
                    <label className="auth-label" htmlFor="login-password">Password</label>
                    <a href="#" className="auth-label-link">Forgot password?</a>
                  </div>
                  <div className="auth-input-wrap">
                    <input
                      id="login-password"
                      type={showPw ? 'text' : 'password'}
                      autoComplete="current-password"
                      className={`auth-input has-trailing ${errors.password ? 'has-error' : ''}`}
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      onBlur={() => setTouched((t) => ({ ...t, password: true }))}
                      disabled={disabled}
                    />
                    <button
                      type="button"
                      className="auth-eye"
                      onClick={() => setShowPw((s) => !s)}
                      aria-label={showPw ? 'Hide password' : 'Show password'}
                      tabIndex={-1}
                    >
                      <Icon name="eye" size={16}/>
                    </button>
                  </div>
                  {errors.password && (
                    <div className="auth-field-error">
                      <Icon name="warn" size={12}/> {errors.password}
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  className={`auth-submit ${submitting ? 'is-loading' : ''}`}
                  disabled={disabled}
                >
                  {submitting
                    ? <><span className="auth-spinner"/> Signing in…</>
                    : <>Sign in</>}
                </button>
              </form>

              <div className="auth-foot">
                Don't have an account yet? <a href="/register">Create an account</a>
              </div>
            </div>
          </main>

          <AuthFooter/>
        </div>
        <aside className="auth-split-right">
          <EvolutionPanel variant="login"/>
        </aside>
      </div>
    </div>
  );
};
