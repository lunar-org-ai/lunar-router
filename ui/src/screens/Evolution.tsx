import { Icon } from '../components/Icon';
import { Sparkline } from '../components/Sparkline';
import { Tag, StatusTag, KindIcon, KindLabel } from '../components/Tag';
import { lessons, overallMetrics, trust, fmtDay, type Lesson } from '../data';

interface Props {
  onOpenLesson: (id: string) => void;
  onNav: (route: string) => void;
  onOpenAgent: () => void;
  dayZero: boolean;
}

const tlNodeClass = (l: Lesson) => {
  if (l.status === 'approved' || l.decision === 'auto_promoted') return 'success';
  if (l.status === 'pending') return 'pending';
  if (l.status === 'rolled_back') return 'bad';
  return 'warn';
};

export const Evolution = ({ onOpenLesson, onNav, onOpenAgent, dayZero }: Props) => {
  const pending = lessons.filter((l) => l.status === 'pending');

  if (dayZero) {
    return (
      <div className="content">
        <h1 className="page-title">Welcome to your agent.</h1>
        <p className="page-sub">It's not deployed yet. Three small steps and it starts learning.</p>
        <div className="day-zero">
          <div className="steps">
            <div className="step">
              <div className="n">1</div>
              <div className="name">Give it a brain</div>
              <div className="help">A system prompt and a default model.</div>
            </div>
            <div className="step">
              <div className="n">2</div>
              <div className="name">Give it hands</div>
              <div className="help">Tools or an MCP server it can call.</div>
            </div>
            <div className="step">
              <div className="n">3</div>
              <div className="name">Give it a mouth</div>
              <div className="help">A channel — WhatsApp, Slack, API, or web.</div>
            </div>
          </div>
          <button className="btn primary" onClick={onOpenAgent}>
            Set up agent <Icon name="chevron" size={14} />
          </button>
          <div className="dim" style={{ fontSize: 12.5, marginTop: 14 }}>
            Once it's running, this page fills with everything it learns.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="content">
      <div className="live-strip">
        <span className="pulse-dot" />
        <span style={{ fontWeight: 500 }}>Live now</span>
        <span className="sep" />
        <span className="stat-mini">
          <b>23</b>
          <span className="l">in conversation</span>
        </span>
        <span className="stat-mini">
          <b>284</b>
          <span className="l">today</span>
        </span>
        <span className="stat-mini">
          <b>2</b>
          <span className="l">flagged for review</span>
        </span>
        <span style={{ marginLeft: 'auto' }}>
          <button className="btn ghost sm" onClick={() => onNav('talk')}>
            Ask the agent <Icon name="chevron" size={12} />
          </button>
        </span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 28 }}>
        <div>
          <h1 className="page-title">Agent evolution</h1>
          <p className="page-sub" style={{ margin: 0 }}>
            Everything your agent has learned, why it changed, and how those changes performed in production.
          </p>
        </div>
        <div className="card" style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 18 }}>
          <div>
            <div
              className="dim"
              style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}
            >
              Trust score
            </div>
            <div style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.01em' }}>
              87<span className="dim" style={{ fontSize: 14, fontWeight: 400 }}> / 100</span>
            </div>
          </div>
          <Sparkline data={trust} w={180} h={48} />
        </div>
      </div>

      <div className="metrics-row">
        {overallMetrics.map((m) => (
          <div className="metric" key={m.label}>
            <div className="label">{m.label}</div>
            <div className="value">{m.value}</div>
            <div className={`delta ${m.dir}`}>{m.delta}</div>
          </div>
        ))}
      </div>

      {pending.length > 0 && (
        <div
          className="card"
          style={{
            marginBottom: 28,
            borderColor: 'var(--warn-soft)',
            background: 'linear-gradient(0deg, var(--warn-soft) 0%, var(--bg-elev) 30%)',
          }}
        >
          <div style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                background: 'var(--warn-soft)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--warn-fg)',
              }}
            >
              <Icon name="bell" size={16} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>
                The agent has {pending.length} change{pending.length > 1 ? 's' : ''} waiting for your review
              </div>
              <div className="dim" style={{ fontSize: 12.5, marginTop: 2 }}>
                Review and approve, or let it roll back automatically in 24h.
              </div>
            </div>
            <button className="btn primary" onClick={() => onNav('review')}>
              Review now <Icon name="chevron" size={14} />
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, margin: 0, letterSpacing: '-0.01em' }}>
          What I've been learning
        </h2>
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn sm">All</button>
          <button className="btn sm ghost">Approved</button>
          <button className="btn sm ghost">Rolled back</button>
        </div>
      </div>

      <div className="timeline">
        {lessons.map((l) => (
          <div className="tl-item" key={l.id}>
            <div className={`tl-node ${tlNodeClass(l)}`} />
            <div className="tl-date">{fmtDay(l.date)}</div>
            <div className="card lesson-card" onClick={() => onOpenLesson(l.id)}>
              <div className="head">
                <Tag>
                  <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
                </Tag>
                <span className="mono dim" style={{ fontSize: 11.5 }}>
                  {l.version}
                </span>
                <div style={{ marginLeft: 'auto' }}>
                  <StatusTag status={l.status === 'approved' ? l.decision : l.status} />
                </div>
              </div>
              <div className="quote">{l.voice}”</div>
              <div className="footing">
                <div className="stats">
                  {l.evals.length > 0 && (
                    <span className="stat">
                      <Icon name="flask" size={12} /> {l.evals.length} eval{l.evals.length > 1 ? 's' : ''}
                    </span>
                  )}
                  {l.traces.length > 0 && (
                    <span className="stat">
                      <Icon name="timeline" size={12} /> {l.traces.length} trace{l.traces.length > 1 ? 's' : ''}
                    </span>
                  )}
                  {l.metrics.length > 0 && (
                    <span className="stat">
                      <Icon name="bolt" size={12} />{' '}
                      {l.metrics
                        .map(
                          (m) =>
                            `${m.label} ${m.delta > 0 ? '+' : ''}${m.delta}${
                              typeof m.delta === 'number' && Math.abs(m.delta) > 5 ? '%' : ''
                            }`
                        )
                        .join(' · ')}
                    </span>
                  )}
                </div>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  Open <Icon name="chevron" size={12} />
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
