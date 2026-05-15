/**
 * RootLayout — sidebar + topbar shell, wraps every route.
 *
 * Owns:
 *   - persona toggle (?view=simple|technical), driving the `tech` body class
 *   - accent CSS vars (set once on document root)
 *   - pendingCount badge (async-loaded once via listLessons)
 *   - AgentSheet modal state, exposed to children via RootCtx so screens
 *     like Evolution can open it without prop drilling
 *   - <TanStackRouterDevtools /> in dev only
 */

import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import {
  Link,
  Outlet,
  useMatches,
  useMatchRoute,
  useNavigate,
  useSearch,
} from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';
import { Icon, type IconName } from '../components/Icon';
import { AgentSheet } from '../screens/AgentSheet';
import { AgentSwitcher } from '../components/AgentSwitcher';
import { Onboarding } from '../screens/Onboarding';
import { Loader } from '../components/Loader';
import { Button } from '../components/ui/button';
import {
  ApiError,
  getFeatures,
  getOnboardingState,
  listLessons,
  type FeatureFlags,
  type OnboardingState,
} from '../api';
import { AUTH_ROUTES, preserveSearch, type View } from '../router';
import { consumeGoogleRedirect, getAuthMode, isAuthed } from '../lib/auth';
import { UserMenu } from '../components/UserMenu';
import { NewAgentModal } from '../components/NewAgentModal';
import { AccountModal } from '../components/AccountModal';

interface RootContext {
  openAgent: () => void;
}

const RootCtx = createContext<RootContext>({ openAgent: () => {} });
export const useRootContext = () => useContext(RootCtx);

interface NavItem {
  to: '/' | '/review' | '/versions' | '/talk' | '/policies';
  label: string;
  icon: IconName;
  badge?: 'pending';
}

interface TechNavItem {
  to:
    | '/technical/traces'
    | '/technical/evals'
    | '/technical/router'
    | '/technical/datasets'
    | '/admin/tenants';
  label: string;
  icon: IconName;
  /** Only render this nav entry when the runtime has the named feature
   *  flag on. Undefined = always show. */
  feature?: keyof FeatureFlags;
}

const NAV: NavItem[] = [
  { to: '/', label: 'Evolution', icon: 'timeline' },
  { to: '/review', label: 'Review', icon: 'inbox', badge: 'pending' },
  { to: '/versions', label: 'Versions', icon: 'git' },
  { to: '/talk', label: 'Talk to agent', icon: 'chat' },
  { to: '/policies', label: 'Policies', icon: 'settings' },
];

const TECH_NAV: TechNavItem[] = [
  { to: '/technical/traces', label: 'Traces', icon: 'timeline' },
  { to: '/technical/evals', label: 'Eval suites', icon: 'flask' },
  { to: '/technical/router', label: 'Router config', icon: 'route' },
  { to: '/technical/datasets', label: 'Datasets', icon: 'book' },
  // Multi-tenant admin. Only surfaces when the runtime is in
  // OPENTRACY_MULTI_TENANT=1 mode (gated by `features.multi_tenant`
  // in the render below).
  { to: '/admin/tenants', label: 'Tenants', icon: 'settings', feature: 'multi_tenant' },
];

const ACCENT = {
  primary: 'oklch(0.55 0.14 150)',
  soft: 'oklch(0.94 0.04 150)',
  fg: 'oklch(0.35 0.12 150)',
};

const ROUTE_LABEL: Record<string, string> = {
  '/': 'Evolution',
  '/review': 'Review',
  '/versions': 'Versions',
  '/talk': 'Talk to agent',
  '/policies': 'Policies',
  '/technical/traces': 'Traces',
  '/technical/evals': 'Eval suites',
  '/technical/router': 'Router config',
  '/technical/datasets': 'Datasets',
  '/admin/tenants': 'Tenants',
};

export const RootLayout = () => {
  // ── Hooks (all declared up front so the early-returns below don't
  //    change React's hook count between renders — issue #310 territory).
  const { view } = useSearch({ from: '__root__' });
  const navigate = useNavigate();
  const matches = useMatches();
  const matchRoute = useMatchRoute();

  // `sheetAgent` controls the AgentSheet drawer:
  //   - undefined → sheet closed
  //   - null      → open, loading the active agent (default)
  //   - <id>      → open, peeking at that specific agent (no activation)
  const [sheetAgent, setSheetAgent] = useState<string | null | undefined>(undefined);
  const [pendingCount, setPendingCount] = useState(0);
  // P16.7 — Create-agent modal triggered from AgentSwitcher's footer.
  const [newAgentOpen, setNewAgentOpen] = useState(false);
  // P16.7 — Account modal opened from UserMenu.
  const [accountOpen, setAccountOpen] = useState(false);
  // P1.11 — day-0 onboarding gate. `null` = still loading; we hold the
  // shell back until we know whether to render Onboarding or the routes.
  const [onboarding, setOnboarding] = useState<OnboardingState | null>(null);
  const [onboardingLoaded, setOnboardingLoaded] = useState(false);
  // P16.4 — runtime feature flags drive which operator-only nav items
  // surface (Tenants is hidden in OSS mode). Defaults to all-off so the
  // sidebar doesn't flash an Admin section before the fetch resolves.
  const [features, setFeatures] = useState<FeatureFlags>({
    multi_tenant: false,
    kms: false,
  });

  // P16.6/P16.7 — Auth shell bypass + gate. /login + /register render
  // outside the sidebar+topbar app chrome, and unauthenticated visitors
  // are bounced there from any other route — but only in multi-tenant
  // mode. OSS-local deploys skip the gate entirely (the backend has no
  // auth either, so the redirect would be pure friction).
  const last = matches[matches.length - 1];
  const fullPath = last?.fullPath ?? '/';
  const isAuthRoute = AUTH_ROUTES.has(fullPath);
  // `authedTick` re-renders the gate after a Google redirect lands:
  // consumeGoogleRedirect() writes the session synchronously, but
  // isAuthed() is read at render-time, so we need a state bump to pick
  // up the change.
  const [authedTick, setAuthedTick] = useState(0);
  const authed = isAuthed();
  void authedTick;
  // `redirectPending` covers the window between "page comes back from
  // Google" and "session saved + navigate('/'). During those few
  // hundred ms (longer if the runtime cold-starts) the user is on
  // /login, but the Login form would just sit there. We render the
  // Loader instead so the screen reads as "signing you in".
  const [redirectPending, setRedirectPending] = useState(true);

  // `gateEnabled` is null while we wait for /v1/auth/mode. We hold the
  // render back during that window so an OSS deploy never flashes the
  // /login screen on first paint.
  const [gateEnabled, setGateEnabled] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getAuthMode().then((mode) => {
      if (!cancelled) setGateEnabled(mode.multi_tenant);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (isAuthRoute) document.body.classList.add('auth-body');
    else document.body.classList.remove('auth-body');
    return () => document.body.classList.remove('auth-body');
  }, [isAuthRoute]);

  // P16.7 — finalize a Google sign-in redirect (if any) before the
  // shell makes any /v1/* call. consumeGoogleRedirect() is a no-op
  // when there's no pending redirect, so it's cheap to run on every
  // boot. On success we bump authedTick and the gate re-evaluates.
  // On failure we surface the reason via sessionStorage so the Login
  // screen can render it on the next render — silent fallbacks here
  // looked exactly like "click did nothing" to the operator.
  useEffect(() => {
    let cancelled = false;
    void consumeGoogleRedirect().then((outcome) => {
      if (cancelled) return;
      setRedirectPending(false);
      if (!outcome) return;
      if ('error' in outcome && outcome.error) {
        sessionStorage.setItem('opentracy.auth.lastError', outcome.error);
        setAuthedTick((n) => n + 1);
        return;
      }
      setAuthedTick((n) => n + 1);
      navigate({ to: '/', replace: true });
    });
    return () => {
      cancelled = true;
    };
  }, [navigate]);

  useEffect(() => {
    if (gateEnabled !== true) return;
    if (!isAuthRoute && !authed) {
      navigate({ to: '/login', replace: true });
    }
  }, [gateEnabled, isAuthRoute, authed, navigate]);

  useEffect(() => {
    document.documentElement.style.setProperty('--primary', ACCENT.primary);
    document.documentElement.style.setProperty('--accent-soft', ACCENT.soft);
    document.documentElement.style.setProperty('--accent-fg', ACCENT.fg);
  }, []);

  // Gate the data fetches: skip on auth screens, wait for the auth-mode
  // probe, and in multi-tenant mode wait for a session. In OSS-local
  // mode the backend has no auth so we fire as soon as the mode is known.
  const canFetch =
    !isAuthRoute && gateEnabled !== null && (gateEnabled === false || authed);

  useEffect(() => {
    if (!canFetch) return;
    let cancelled = false;
    (async () => {
      try {
        const state = await getOnboardingState();
        if (!cancelled) setOnboarding(state);
      } catch (e) {
        // Backend down or unreachable — fall through to the normal shell
        // so the operator can still see whatever cached state the UI has.
        if (!(e instanceof ApiError)) console.warn('getOnboardingState failed', e);
      } finally {
        if (!cancelled) setOnboardingLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [canFetch]);

  useEffect(() => {
    if (!canFetch) return;
    let cancelled = false;
    (async () => {
      try {
        const items = await listLessons();
        if (cancelled) return;
        setPendingCount(
          items.filter((l) => l.status === 'pending' || l.status === 'awaiting_review').length,
        );
      } catch (e) {
        if (!(e instanceof ApiError)) console.warn('listLessons failed', e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [canFetch]);

  // P16.4 — pull deployment-mode flags from the runtime. Defensive:
  // getFeatures returns {false, false} on any error so the rest of the
  // shell renders even against an older backend.
  useEffect(() => {
    if (!canFetch) return;
    let cancelled = false;
    (async () => {
      const f = await getFeatures();
      if (!cancelled) setFeatures(f);
    })();
    return () => {
      cancelled = true;
    };
  }, [canFetch]);

  const ctx = useMemo<RootContext>(() => ({ openAgent: () => setSheetAgent(null) }), []);

  // ── End of hooks. Early returns from here on are safe.

  if (isAuthRoute) {
    // Cover the gap between the Google redirect landing and the
    // session being exchanged so the Login form doesn't sit there
    // looking idle. After the first consumeGoogleRedirect() resolves
    // (success or no-op), `redirectPending` drops and the form
    // becomes interactive again.
    if (redirectPending) {
      return <Loader caption="Signing you in…" />;
    }
    return <Outlet />;
  }
  // Hold render with a Loader until we know whether the gate is on
  // (avoids a flash of /login on OSS deploys, and gives the user a
  // friendly waiting state while the backend cold-starts).
  if (gateEnabled === null) {
    return <Loader />;
  }
  // Multi-tenant mode + no session: the useEffect above pushes us to
  // /login on the next tick. Show the Loader meanwhile so we don't
  // either flash the shell or render blank during the redirect.
  if (gateEnabled && !authed) {
    return <Loader caption="Signing you in…" />;
  }
  // Authed but onboarding state still unknown: hold the render with
  // the Loader so a fresh tenant never sees the empty Evolution shell
  // flash before /onboarding/state resolves and dispatches us to the
  // chat. Without this gate, the routes paint first, then a re-render
  // swaps in <Onboarding>, which the user noticed as "agent screen
  // then onboarding".
  if (!onboardingLoaded) {
    return <Loader caption="Preparing your agent…" />;
  }

  const crumbs: string[] = fullPath.startsWith('/lesson')
    ? ['Evolution', 'Lesson']
    : [ROUTE_LABEL[fullPath] ?? 'Evolution'];

  const evolutionActive = !!matchRoute({ to: '/' }) || !!matchRoute({ to: '/lesson/$id' });
  const setView = (v: View) =>
    navigate({
      to: '.',
      params: (prev) => prev,
      search: (prev: Record<string, unknown>) => ({ ...prev, view: v }),
    });

  // P1.11 — render Onboarding instead of the route shell on first visit.
  // We wait for the fetch to resolve (or fail) so the layout never flashes.
  if (onboardingLoaded && onboarding && !onboarding.completed) {
    return (
      <Onboarding
        onDone={(next) => {
          if (next) setOnboarding(next);
          else setOnboarding({ ...(onboarding as OnboardingState), completed: true });
        }}
      />
    );
  }

  // Layout strategy: pull the sidebar out of normal flow with position:fixed
  // and reserve its 240px gutter on .main with margin-left. Inline styles on
  // the critical width/position fields so Tailwind utilities or @layer base
  // resets cannot override them — this isolates the sidebar from any cascade
  // surprises in the shadcn/Tailwind v4 stack.
  const SIDEBAR_W = 240;
  const sidebarStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: 0,
    bottom: 0,
    width: SIDEBAR_W,
    minWidth: SIDEBAR_W,
    maxWidth: SIDEBAR_W,
    flexShrink: 0,
    background: 'var(--card)',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    zIndex: 10,
  };
  const mainStyle: React.CSSProperties = {
    marginLeft: SIDEBAR_W,
    height: '100vh',
    overflow: 'auto',
    display: 'flex',
    flexDirection: 'column',
    minHeight: 0,
  };

  return (
    <RootCtx.Provider value={ctx}>
      <div
        className={`app ${view === 'technical' ? 'tech' : ''}`}
        data-screen-label={`Screen ${last?.pathname ?? '/'}`}
        style={{ minHeight: '100vh' }}
      >
        <aside className="sidebar" style={sidebarStyle}>
          <div className="sidebar-head">
            <div className="sidebar-mark" />
            <div className="sidebar-name">
              OpenTracy <span className="dim">Evolution</span>
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-section-label">Agent</div>
            {NAV.map((n) => {
              const active = n.to === '/' ? evolutionActive : !!matchRoute({ to: n.to });
              return (
                <Link
                  key={n.to}
                  to={n.to}
                  search={preserveSearch}
                  className={`sidebar-item ${active ? 'active' : ''}`}
                >
                  <Icon name={n.icon} size={15} />
                  <span>{n.label}</span>
                  {n.badge === 'pending' && pendingCount > 0 && (
                    <span className="badge warn">{pendingCount}</span>
                  )}
                </Link>
              );
            })}
          </div>

          <div className="sidebar-section tech-only" style={{ marginTop: 8 }}>
            <div className="sidebar-section-label">Technical</div>
            {TECH_NAV.filter((n) => !n.feature || features[n.feature]).map((n) => (
              <Link
                key={n.to}
                to={n.to}
                search={preserveSearch}
                className={`sidebar-item ${matchRoute({ to: n.to }) ? 'active' : ''}`}
              >
                <Icon name={n.icon} size={15} />
                <span>{n.label}</span>
              </Link>
            ))}
          </div>

          <div className="sidebar-foot">
            <UserMenu onOpenAccount={() => setAccountOpen(true)} />
            <div className="persona-switch" title="Switch view">
              <button
                className={view !== 'technical' ? 'on' : ''}
                onClick={() => setView('simple')}
              >
                Simple
              </button>
              <button
                className={view === 'technical' ? 'on' : ''}
                onClick={() => setView('technical')}
              >
                Technical
              </button>
            </div>
          </div>
        </aside>

        <main className="main" style={mainStyle}>
          <div className="topbar">
            <div className="topbar-title">
              {crumbs.map((c, i) => (
                <span key={i}>
                  {i < crumbs.length - 1 ? (
                    <span className="crumb">
                      {c} <span style={{ opacity: 0.4 }}>/</span>
                    </span>
                  ) : (
                    c
                  )}
                </span>
              ))}
            </div>
            <div className="topbar-right">
              <Button
                size="sm"
                onClick={() => setNewAgentOpen(true)}
                title="Create a new agent"
              >
                <Icon name="plus" size={13} />
                <span>New agent</span>
              </Button>
              <Button variant="ghost" size="sm" aria-label="Notifications">
                <Icon name="bell" size={14} />
              </Button>
              <AgentSwitcher
                onOpenSheet={(id) => setSheetAgent(id ?? null)}
                onNewAgent={() => setNewAgentOpen(true)}
              />
            </div>
          </div>
          <Outlet />
        </main>

        {sheetAgent !== undefined && (
          <AgentSheet
            agentId={sheetAgent}
            onClose={() => setSheetAgent(undefined)}
          />
        )}

        {accountOpen && <AccountModal onClose={() => setAccountOpen(false)} />}

        {newAgentOpen && (
          <NewAgentModal
            onClose={() => setNewAgentOpen(false)}
            onCreated={() => {
              // Same reload pattern as activate-agent — every fetcher
              // rebinds against the freshly-active agent on a clean boot.
              window.location.reload();
            }}
            onGuided={() => {
              // Drop the modal and hand control to the full Onboarding
              // flow. Resetting the local `completed` flag is enough —
              // the gate above will render <Onboarding> on next paint.
              setNewAgentOpen(false);
              setOnboarding((prev) =>
                prev ? { ...prev, completed: false } : prev,
              );
            }}
          />
        )}
      </div>

      {import.meta.env.DEV && <TanStackRouterDevtools position="bottom-right" />}
    </RootCtx.Provider>
  );
};
