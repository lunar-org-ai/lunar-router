/**
 * UserMenu — sidebar foot row showing the signed-in user + a popover
 * menu with Account / Workspace / Help / Logout (P16.7).
 *
 * Pulls user info from the active session (set when /v1/auth/session
 * mints the Bearer). Logout clears the session and navigates back to
 * /login; the rest of the menu items are placeholders for now.
 */
import { useEffect, useState } from 'react';
import { Icon } from './Icon';
import { getSession, signOut, type AuthSession } from '../lib/auth';

interface UserMenuProps {
  onOpenAccount?: () => void;
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

export const UserMenu = ({ onOpenAccount }: UserMenuProps) => {
  const [open, setOpen] = useState(false);
  const [session, setSession] = useState<AuthSession | null>(() => getSession());

  // Re-read the session on mount in case it was set after this component
  // first rendered (e.g. Google redirect just finalized).
  useEffect(() => {
    setSession(getSession());
  }, []);

  if (!session) return null;

  const initials = initialsFor(session.name);

  const handleLogout = async () => {
    setOpen(false);
    await signOut();
    // Hard reload to /login so every in-flight fetch, cached state, and
    // sessionStorage scrap is dropped — cheaper than walking the React
    // tree to invalidate everything.
    window.location.assign('/login');
  };

  return (
    <div className="user-row-wrap">
      <button
        className={`user-row ${open ? 'on' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="user-avatar" aria-hidden="true">{initials}</span>
        <span className="user-id">
          <span className="user-name">{session.name}</span>
          <span className="user-email">{session.email}</span>
        </span>
        <Icon name="chevronDown" size={12} />
      </button>

      {open && (
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="user-menu" role="menu">
            <div className="user-menu-head">
              <span className="user-avatar lg" aria-hidden="true">{initials}</span>
              <div>
                <div className="user-name">{session.name}</div>
                <div className="user-email mono">{session.email}</div>
              </div>
            </div>
            <div className="user-menu-section">
              <button
                className="user-menu-item"
                onClick={() => { setOpen(false); onOpenAccount?.(); }}
              >
                <Icon name="user" size={14} /><span>Account</span>
              </button>
              <button className="user-menu-item" disabled>
                <Icon name="sliders" size={14} /><span>Workspace settings</span>
              </button>
              <button className="user-menu-item" disabled>
                <Icon name="book" size={14} /><span>Help &amp; docs</span>
              </button>
            </div>
            <div className="user-menu-section divider">
              <button className="user-menu-item danger" onClick={handleLogout}>
                <Icon name="rollback" size={14} /><span>Log out</span>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
