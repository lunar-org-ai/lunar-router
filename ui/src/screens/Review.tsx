import { useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag, KindIcon, KindLabel } from '../components/Tag';
import { ConfBar } from '../components/ConfBar';
import { Diff } from '../components/Diff';
import { lessons, type Eval } from '../data';

const fileLabelFor = (type: string) =>
  type === 'system_prompt'
    ? 'system_prompt.md'
    : type === 'tool_wrapper'
    ? 'tools/lookup_order.py'
    : type === 'routing'
    ? 'routing.yml'
    : type === 'eval'
    ? 'evals/'
    : 'config.yml';

const fmt = (v: number | null, e: Eval) =>
  v == null
    ? '—'
    : e.scale
    ? v.toFixed(1)
    : e.currency
    ? `$${v.toFixed(3)}`
    : v < 1 && v > 0
    ? `${(v * 100).toFixed(0)}%`
    : v.toFixed(0);

export const Review = ({ onOpenLesson }: { onOpenLesson: (id: string) => void }) => {
  const pending = lessons.filter((l) => l.status === 'pending');
  const [decided, setDecided] = useState<Record<string, 'approve' | 'reject'>>({});
  const remaining = pending.filter((l) => !decided[l.id]);

  const decide = (id: string, action: 'approve' | 'reject') =>
    setDecided((d) => ({ ...d, [id]: action }));

  return (
    <div className="content">
      <h1 className="page-title">Pending review</h1>
      <p className="page-sub">
        The agent has proposed changes based on what it learned. Approve to ship them, reject to discard, or wait —
        anything untouched in 24h follows your default policy.
      </p>

      {remaining.length === 0 && (
        <div className="card card-pad empty" style={{ padding: 64 }}>
          <Icon name="check" size={32} />
          <div style={{ fontSize: 16, fontWeight: 500, color: 'var(--fg)', marginTop: 12 }}>Inbox zero.</div>
          <div style={{ marginTop: 4 }}>The agent will check in again when it has something new to suggest.</div>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {remaining.map((l) => {
          return (
            <div className="proposal" key={l.id}>
              <div className="h">
                <Tag>
                  <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
                </Tag>
                <span className="title">{l.title}</span>
                <span className="dim mono" style={{ marginLeft: 'auto', fontSize: 11.5 }}>
                  {l.id}
                </span>
              </div>
              <div className="b">
                <div>
                  <div className="quote">{l.voice}”</div>
                  <div
                    className="dim"
                    style={{
                      fontSize: 12.5,
                      marginBottom: 8,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      fontWeight: 500,
                    }}
                  >
                    What changes
                  </div>
                  <Diff
                    before={l.proposal.beforeLines}
                    after={l.proposal.afterLines}
                    fileLabel={fileLabelFor(l.proposal.type)}
                  />
                </div>
                <div>
                  <div
                    className="dim"
                    style={{
                      fontSize: 12.5,
                      marginBottom: 10,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      fontWeight: 500,
                    }}
                  >
                    Projected impact
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 0,
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius)',
                      overflow: 'hidden',
                      marginBottom: 14,
                    }}
                  >
                    {l.evals.map((e, i) => {
                      const better = e.isLatency ? e.after < (e.before ?? Infinity) : e.after > (e.before ?? -Infinity);
                      return (
                        <div
                          key={i}
                          style={{
                            padding: 10,
                            borderBottom: i < l.evals.length - 1 ? '1px solid var(--border)' : 'none',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: 13,
                          }}
                        >
                          <span style={{ fontSize: 12.5 }}>{e.name}</span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span className="mono dim" style={{ fontSize: 12 }}>
                              {fmt(e.before, e)}
                            </span>
                            <Icon name="chevron" size={11} />
                            <span
                              className="mono"
                              style={{ fontWeight: 500, color: better ? 'var(--accent-fg)' : 'var(--bad-fg)' }}
                            >
                              {fmt(e.after, e)}
                            </span>
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div
                    className="dim"
                    style={{
                      fontSize: 12.5,
                      marginBottom: 8,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      fontWeight: 500,
                    }}
                  >
                    Confidence
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                    <ConfBar value={l.confidence} />
                    <span className="dim" style={{ fontSize: 12.5 }}>
                      Based on {l.evals.reduce((a, b) => a + (b.sample || 0), 0).toLocaleString()} samples
                    </span>
                  </div>
                  <button
                    className="btn sm ghost"
                    onClick={() => onOpenLesson(l.id)}
                    style={{ padding: '0 10px' }}
                  >
                    <Icon name="eye" size={12} /> See full reasoning
                  </button>
                </div>
              </div>
              <div className="actions">
                <span className="left">
                  Auto-decision in <span className="mono">22h 14m</span> per policy
                </span>
                <button className="btn danger" onClick={() => decide(l.id, 'reject')}>
                  <Icon name="x" size={14} /> Reject
                </button>
                <button className="btn success" onClick={() => decide(l.id, 'approve')}>
                  <Icon name="check" size={14} /> Approve & ship
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {Object.keys(decided).length > 0 && (
        <div style={{ marginTop: 24 }}>
          <div
            className="dim"
            style={{
              fontSize: 12,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: 8,
              fontWeight: 500,
            }}
          >
            Decided just now
          </div>
          {Object.entries(decided).map(([id, action]) => {
            const l = lessons.find((x) => x.id === id);
            if (!l) return null;
            return (
              <div
                key={id}
                className="card"
                style={{
                  padding: '12px 16px',
                  marginBottom: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  fontSize: 13.5,
                }}
              >
                <Tag kind={action === 'approve' ? 'success' : 'bad'}>
                  <span className="dot" /> {action === 'approve' ? 'Approved' : 'Rejected'}
                </Tag>
                <span>{l.title}</span>
                <button
                  className="btn ghost sm"
                  style={{ marginLeft: 'auto' }}
                  onClick={() =>
                    setDecided((d) => {
                      const n = { ...d };
                      delete n[id];
                      return n;
                    })
                  }
                >
                  Undo
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
