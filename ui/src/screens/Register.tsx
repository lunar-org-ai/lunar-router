/**
 * Register screen (P16.7).
 *
 * Real auth via Firebase Auth → backend session exchange. See Login.tsx
 * + lib/auth.ts for the matching flow.
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
  passStrength,
  strengthLabel,
} from './AuthShared';
import {
  EmailNotVerifiedError,
  readableAuthError,
  signInWithGoogle,
  signUpWithEmail,
} from '../lib/auth';

interface RegisterErrors {
  name?: string;
  email?: string;
  password?: string;
}

type Touched = { name?: boolean; email?: boolean; password?: boolean };

export const Register = () => {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [errors, setErrors] = useState<RegisterErrors>({});
  const [touched, setTouched] = useState<Touched>({});
  const [submitting, setSubmitting] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const strength = passStrength(password);

  useEffect(() => {
    const errs: RegisterErrors = {};
    if (touched.name && !name.trim()) errs.name = 'Tell us what to call you.';
    if (touched.email) {
      if (!email.trim()) errs.email = 'Enter your email.';
      else if (!isEmail(email)) errs.email = 'That doesn’t look like a valid email.';
    }
    if (touched.password) {
      if (!password) errs.password = 'Choose a password.';
      else if (password.length < 8) errs.password = 'Use at least 8 characters.';
    }
    setErrors(errs);
  }, [name, email, password, touched]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const errs: RegisterErrors = {};
    if (!name.trim()) errs.name = 'Tell us what to call you.';
    if (!email.trim()) errs.email = 'Enter your email.';
    else if (!isEmail(email)) errs.email = 'That doesn’t look like a valid email.';
    if (!password) errs.password = 'Choose a password.';
    else if (password.length < 8) errs.password = 'Use at least 8 characters.';
    setTouched({ name: true, email: true, password: true });
    setErrors(errs);
    setAuthError(null);
    if (Object.keys(errs).length) return;

    setSubmitting(true);
    try {
      // signUpWithEmail intentionally throws EmailNotVerifiedError on
      // success — there's no Bearer to hand back until the user
      // clicks the link.
      await signUpWithEmail(email, password, name);
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

  const googleSignUp = async () => {
    setGoogleLoading(true);
    setAuthError(null);
    try {
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
            swapTo="/login"
            swapPrompt="Already have an account?"
            swapCta="Sign in"
          />

          <main className="auth-main">
            <div className="auth-card">
              <h1 className="auth-h1">Create your account</h1>
              <p className="auth-sub">Build your first agent in under five minutes. No credit card needed.</p>

              {authError && (
                <div className="auth-alert" role="alert">
                  <Icon name="warn" size={14}/>
                  <span>{authError}</span>
                </div>
              )}

              <button
                type="button"
                className="auth-google"
                onClick={googleSignUp}
                disabled={disabled}
              >
                {googleLoading
                  ? <><span className="auth-spinner" style={{ borderTopColor: 'var(--fg)', borderColor: 'oklch(0.18 0.005 80 / 0.18)' }}/> Connecting Google…</>
                  : <><GoogleG/> Sign up with Google</>}
              </button>

              <div className="auth-divider">or</div>

              <form className="auth-form" onSubmit={submit} noValidate>
                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-name">Full name</label>
                  <input
                    id="reg-name"
                    type="text"
                    autoComplete="name"
                    className={`auth-input ${errors.name ? 'has-error' : ''}`}
                    placeholder="Avery Chen"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onBlur={() => setTouched((t) => ({ ...t, name: true }))}
                    disabled={disabled}
                  />
                  {errors.name && (
                    <div className="auth-field-error">
                      <Icon name="warn" size={12}/> {errors.name}
                    </div>
                  )}
                </div>

                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-email">Work email</label>
                  <input
                    id="reg-email"
                    type="email"
                    autoComplete="email"
                    className={`auth-input ${errors.email ? 'has-error' : ''}`}
                    placeholder="you@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    onBlur={() => setTouched((t) => ({ ...t, email: true }))}
                    disabled={disabled}
                  />
                  {errors.email && (
                    <div className="auth-field-error">
                      <Icon name="warn" size={12}/> {errors.email}
                    </div>
                  )}
                </div>

                <div className="auth-field">
                  <label className="auth-label" htmlFor="reg-password">Password</label>
                  <div className="auth-input-wrap">
                    <input
                      id="reg-password"
                      type={showPw ? 'text' : 'password'}
                      autoComplete="new-password"
                      className={`auth-input has-trailing ${errors.password ? 'has-error' : ''}`}
                      placeholder="At least 8 characters"
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
                  {password && !errors.password && (
                    <>
                      <div className={`auth-strength s${strength}`} aria-hidden="true">
                        <i/><i/><i/><i/>
                      </div>
                      <div className="auth-strength-label">
                        Strength: <span className={`s${strength}`}>{strengthLabel(strength) || '—'}</span>
                      </div>
                    </>
                  )}
                  {errors.password && (
                    <div className="auth-field-error">
                      <Icon name="warn" size={12}/> {errors.password}
                    </div>
                  )}
                  {!password && !errors.password && (
                    <div className="auth-hint">8+ characters. Mix letters, numbers, and symbols for a stronger password.</div>
                  )}
                </div>

                <button
                  type="submit"
                  className={`auth-submit ${submitting ? 'is-loading' : ''}`}
                  disabled={disabled}
                >
                  {submitting
                    ? <><span className="auth-spinner"/> Creating account…</>
                    : <>Create account</>}
                </button>
              </form>

              <div className="auth-legal">
                By continuing you agree to our <a href="#">Terms</a> and <a href="#">Privacy Policy</a>.
              </div>

              <div className="auth-foot">
                Already have an account? <a href="/login">Sign in</a>
              </div>
            </div>
          </main>

          <AuthFooter/>
        </div>
        <aside className="auth-split-right">
          <EvolutionPanel variant="register"/>
        </aside>
      </div>
    </div>
  );
};
