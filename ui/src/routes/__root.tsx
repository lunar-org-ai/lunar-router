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
import { Button } from '../components/ui/button';
import { ApiError, listLessons } from '../api';
import { preserveSearch, type View } from '../router';

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
    | '/technical/traces/live'
    | '/technical/evals'
    | '/technical/router'
    | '/technical/datasets';
  label: string;
  icon: IconName;
}

interface LabNavItem {
  to: '/lab/assistant-ui' | '/lab/copilot-kit';
  label: string;
  icon: IconName;
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
  { to: '/technical/traces/live', label: 'Traces · Live', icon: 'timeline' },
  { to: '/technical/evals', label: 'Eval suites', icon: 'flask' },
  { to: '/technical/router', label: 'Router config', icon: 'route' },
  { to: '/technical/datasets', label: 'Datasets', icon: 'book' },
];

const LAB_NAV: LabNavItem[] = [
  { to: '/lab/assistant-ui', label: 'assistant-ui', icon: 'sparkles' },
  { to: '/lab/copilot-kit', label: 'CopilotKit', icon: 'bolt' },
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
  '/technical/traces/live': 'Traces · Live',
  '/technical/evals': 'Eval suites',
  '/technical/router': 'Router config',
  '/technical/datasets': 'Datasets',
  '/lab/assistant-ui': 'Lab — assistant-ui',
  '/lab/copilot-kit': 'Lab — CopilotKit',
};

export const RootLayout = () => {
  const { view } = useSearch({ from: '__root__' });
  const navigate = useNavigate();
  const matches = useMatches();
  const matchRoute = useMatchRoute();
  const [agentOpen, setAgentOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);

  useEffect(() => {
    document.documentElement.style.setProperty('--primary', ACCENT.primary);
    document.documentElement.style.setProperty('--accent-soft', ACCENT.soft);
    document.documentElement.style.setProperty('--accent-fg', ACCENT.fg);
  }, []);

  useEffect(() => {
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
  }, []);

  const last = matches[matches.length - 1];
  const fullPath = last?.fullPath ?? '/';
  const pathname = last?.pathname ?? '/';
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

  const ctx = useMemo<RootContext>(() => ({ openAgent: () => setAgentOpen(true) }), []);

  return (
    <RootCtx.Provider value={ctx}>
      <div
        className={`app ${view === 'technical' ? 'tech' : ''}`}
        data-screen-label={`Screen ${pathname}`}
      >
        <aside className="sidebar">
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
            {TECH_NAV.map((n) => (
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

          <div className="sidebar-section" style={{ marginTop: 8 }}>
            <div className="sidebar-section-label">Lab</div>
            {LAB_NAV.map((n) => (
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
            <div className="persona-switch" title="Switch view">
              <button className={view === 'simple' ? 'on' : ''} onClick={() => setView('simple')}>
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

        <main className="main">
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
              <Button variant="ghost" size="sm">
                <Icon name="bell" size={14} />
              </Button>
              <button className="agent-pill" onClick={() => setAgentOpen(true)}>
                <span className="dot" />
                <span>support-agent</span>
                <span className="ver">v0.40 · live</span>
                <Icon name="chevronDown" size={12} />
              </button>
            </div>
          </div>
          <Outlet />
        </main>

        {agentOpen && <AgentSheet onClose={() => setAgentOpen(false)} />}
      </div>

      {import.meta.env.DEV && <TanStackRouterDevtools position="bottom-right" />}
    </RootCtx.Provider>
  );
};
