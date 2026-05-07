import { useCallback, useEffect, useState } from 'react';
import { Icon, type IconName } from './components/Icon';
import { Evolution } from './screens/Evolution';
import { LessonDetail } from './screens/LessonDetail';
import { Review } from './screens/Review';
import { Versions } from './screens/Versions';
import { TalkToAgent } from './screens/TalkToAgent';
import { Policies } from './screens/Policies';
import { AgentSheet } from './screens/AgentSheet';
import { Traces, EvalSuites, RouterConfig, Datasets } from './screens/Technical';
import { lessons } from './data';

type RouteName =
  | 'evolution'
  | 'lesson'
  | 'review'
  | 'versions'
  | 'talk'
  | 'policies'
  | 'traces'
  | 'evals'
  | 'router'
  | 'datasets';

interface Route {
  name: RouteName;
  params: Record<string, string>;
}

interface NavItem {
  id: RouteName;
  label: string;
  icon: IconName;
  badge?: 'pending';
}

const NAV: NavItem[] = [
  { id: 'evolution', label: 'Evolution', icon: 'timeline' },
  { id: 'review', label: 'Review', icon: 'inbox', badge: 'pending' },
  { id: 'versions', label: 'Versions', icon: 'git' },
  { id: 'talk', label: 'Talk to agent', icon: 'chat' },
  { id: 'policies', label: 'Policies', icon: 'settings' },
];

type Accent = 'green' | 'blue' | 'amber' | 'plum';

const ACCENTS: Record<Accent, { accent: string; soft: string; fg: string }> = {
  green: { accent: 'oklch(0.55 0.14 150)', soft: 'oklch(0.94 0.04 150)', fg: 'oklch(0.35 0.12 150)' },
  blue: { accent: 'oklch(0.55 0.14 250)', soft: 'oklch(0.94 0.04 250)', fg: 'oklch(0.35 0.12 250)' },
  amber: { accent: 'oklch(0.65 0.14 70)', soft: 'oklch(0.94 0.04 70)', fg: 'oklch(0.4 0.12 70)' },
  plum: { accent: 'oklch(0.55 0.14 320)', soft: 'oklch(0.94 0.04 320)', fg: 'oklch(0.4 0.12 320)' },
};

export const App = () => {
  const [route, setRoute] = useState<Route>({ name: 'evolution', params: {} });
  const [collapsed] = useState(false);
  const [persona, setPersona] = useState<'simple' | 'technical'>('simple');
  const [agentOpen, setAgentOpen] = useState(false);
  const [accent] = useState<Accent>('green');
  const dayZero = false;

  useEffect(() => {
    const a = ACCENTS[accent] || ACCENTS.green;
    document.documentElement.style.setProperty('--accent', a.accent);
    document.documentElement.style.setProperty('--accent-soft', a.soft);
    document.documentElement.style.setProperty('--accent-fg', a.fg);
  }, [accent]);

  const pendingCount = lessons.filter((l) => l.status === 'pending').length;

  const goTo = useCallback((name: RouteName, params: Record<string, string> = {}) => {
    setRoute({ name, params });
    document.querySelector('.main')?.scrollTo(0, 0);
  }, []);

  let content: React.ReactNode = null;
  let crumbs: string[] = [];
  switch (route.name) {
    case 'evolution':
      crumbs = ['Evolution'];
      content = (
        <Evolution
          onOpenLesson={(id) => goTo('lesson', { id })}
          onNav={(r) => goTo(r as RouteName)}
          onOpenAgent={() => setAgentOpen(true)}
          dayZero={dayZero}
        />
      );
      break;
    case 'lesson': {
      const lesson = lessons.find((l) => l.id === route.params.id);
      crumbs = ['Evolution', lesson?.title || 'Lesson'];
      content = <LessonDetail lessonId={route.params.id} onBack={() => goTo('evolution')} />;
      break;
    }
    case 'review':
      crumbs = ['Review'];
      content = <Review onOpenLesson={(id) => goTo('lesson', { id })} />;
      break;
    case 'versions':
      crumbs = ['Versions'];
      content = <Versions />;
      break;
    case 'talk':
      crumbs = ['Talk to agent'];
      content = <TalkToAgent />;
      break;
    case 'policies':
      crumbs = ['Policies'];
      content = <Policies />;
      break;
    case 'traces':
      crumbs = ['Traces'];
      content = <Traces />;
      break;
    case 'evals':
      crumbs = ['Eval suites'];
      content = <EvalSuites />;
      break;
    case 'router':
      crumbs = ['Router config'];
      content = <RouterConfig />;
      break;
    case 'datasets':
      crumbs = ['Datasets'];
      content = <Datasets />;
      break;
  }

  return (
    <div
      className={`app ${collapsed ? 'collapsed' : ''} ${persona === 'technical' ? 'tech' : ''}`}
      data-screen-label={`Screen ${route.name}`}
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
          {NAV.map((n) => (
            <button
              key={n.id}
              className={`sidebar-item ${
                route.name === n.id || (n.id === 'evolution' && route.name === 'lesson') ? 'active' : ''
              }`}
              onClick={() => goTo(n.id)}
            >
              <Icon name={n.icon} size={15} />
              <span>{n.label}</span>
              {n.badge === 'pending' && pendingCount > 0 && <span className="badge warn">{pendingCount}</span>}
            </button>
          ))}
        </div>

        <div className="sidebar-section tech-only" style={{ marginTop: 8 }}>
          <div className="sidebar-section-label">Technical</div>
          <button
            className={`sidebar-item ${route.name === 'traces' ? 'active' : ''}`}
            onClick={() => goTo('traces')}
          >
            <Icon name="timeline" size={15} />
            <span>Traces</span>
          </button>
          <button
            className={`sidebar-item ${route.name === 'evals' ? 'active' : ''}`}
            onClick={() => goTo('evals')}
          >
            <Icon name="flask" size={15} />
            <span>Eval suites</span>
          </button>
          <button
            className={`sidebar-item ${route.name === 'router' ? 'active' : ''}`}
            onClick={() => goTo('router')}
          >
            <Icon name="route" size={15} />
            <span>Router config</span>
          </button>
          <button
            className={`sidebar-item ${route.name === 'datasets' ? 'active' : ''}`}
            onClick={() => goTo('datasets')}
          >
            <Icon name="book" size={15} />
            <span>Datasets</span>
          </button>
        </div>

        <div className="sidebar-foot">
          <div className="persona-switch" title="Switch view">
            <button className={persona === 'simple' ? 'on' : ''} onClick={() => setPersona('simple')}>
              Simple
            </button>
            <button className={persona === 'technical' ? 'on' : ''} onClick={() => setPersona('technical')}>
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
            <button className="btn sm ghost">
              <Icon name="bell" size={14} />
            </button>
            <button className="agent-pill" onClick={() => setAgentOpen(true)}>
              <span className="dot" />
              <span>support-agent</span>
              <span className="ver">v0.40 · live</span>
              <Icon name="chevronDown" size={12} />
            </button>
          </div>
        </div>
        {content}
      </main>

      {agentOpen && <AgentSheet onClose={() => setAgentOpen(false)} />}
    </div>
  );
};
