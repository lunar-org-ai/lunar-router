/**
 * AccountModal — shows the signed-in user's session info + Logout
 * action (P16.7). Replaces the previous wiring that incorrectly
 * opened AgentSheet on "Account" click — AgentSheet is per-agent
 * config and was rendering empty / broken in that context.
 *
 * Content is everything we already have client-side from the session
 * blob: name, email, tenant id, signed-in timestamp. Workspace
 * settings + billing land as separate screens in a later phase.
 */
import { useEffect } from 'react';
import { Icon } from './Icon';
import { getSession, signOut } from '../lib/auth';

interface AccountModalProps {
  onClose: () => void;
}

function initialsFor(name: string): string {
  return (
    name
      .split(/\s+/)
      .map((s) => s[0])
      .filter(Boolean)
      .slice(0, 2)
      .join('')
      .toUpperCase() || '?'
  );
}

function formatSignedIn(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export const AccountModal = ({ onClose }: AccountModalProps) => {
  const session = getSession();

  // ESC closes — matches NewAgentModal's pattern.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!session) {
    return null;
  }

  const handleLogout = async () => {
    await signOut();
    window.location.assign('/login');
  };

  const initials = initialsFor(session.name);

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal account-modal" role="dialog" aria-label="Account">
        <div className="modal-head">
          <h2>Your account</h2>
          <button type="button" className="btn ghost sm" onClick={onClose} aria-label="Close">
            <Icon name="x" size={14} />
          </button>
        </div>

        <div className="modal-body">
          <div className="account-identity">
            <span className="user-avatar lg" aria-hidden="true">{initials}</span>
            <div className="account-identity-text">
              <div className="account-identity-name">{session.name}</div>
              <div className="account-identity-email mono">{session.email}</div>
            </div>
          </div>

          <dl className="account-meta">
            <div className="account-meta-row">
              <dt>Signed in</dt>
              <dd>{formatSignedIn(session.signedInAt)}</dd>
            </div>
          </dl>
        </div>

        <div className="modal-foot account-foot">
          <button type="button" className="btn ghost" onClick={onClose}>Close</button>
          <button type="button" className="btn danger" onClick={handleLogout}>
            <Icon name="rollback" size={13} />
            Log out
          </button>
        </div>
      </div>
    </>
  );
};
