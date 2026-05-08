/**
 * Evolution — the home of the agent.
 *
 * AHE paper (arxiv 2604.25850) three pillars structure each card:
 *   - decision   → voice (the agent's first-person rationale, top of card)
 *   - component  → mutations (the file/knob change, mono dim row)
 *   - experience → delta.overall_score / delta.pass_rate (small Δ chip)
 *
 * Operational metrics (CSAT, Resolution rate, Trust score, "in conversation
 * now" ticker) are intentionally absent until the agent is hooked into real
 * production traffic. Showing synthesized numbers here would be theatre.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag, StatusTag, KindIcon, KindLabel } from '../components/Tag';
import { ApiError, listLessons, type LessonSummary } from '../api';

interface Props {
  onOpenLesson: (id: string) => void;
  onNav: (route: string) => void;
  onOpenAgent: () => void;
  dayZero: boolean;
}

const fmtDay = (iso: string | null): string => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return iso;
  }
};

const tlNodeClass = (l: LessonSummary): string => {
  if (l.status === 'approved' || l.status === 'auto_promoted') return 'success';
  if (l.status === 'pending' || l.status === 'awaiting_review') return 'pending';
  if (l.status === 'rolled_back' || l.status === 'human_rejected') return 'bad';
  return 'warn';
};

type Filter = 'all' | 'approved' | 'rolled_back';

export const Evolution = ({ onOpenLesson, onNav, onOpenAgent, dayZero }: Props) => {
  const [lessons, setLessons] = useState<LessonSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await listLessons();
      setLessons(items);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Backend ${e.status}: ${e.message}`
          : `Network error: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const pending = useMemo(
    () => lessons.filter((l) => l.status === 'pending' || l.status === 'awaiting_review'),
    [lessons],
  );

  const filtered = useMemo(() => {
    if (filter === 'approved') {
      return lessons.filter((l) => l.status === 'approved' || l.status === 'auto_promoted');
    }
    if (filter === 'rolled_back') {
      return lessons.filter((l) => l.status === 'rolled_back' || l.status === 'human_rejected');
    }
    return lessons;
  }, [lessons, filter]);

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

  if (loading && lessons.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Agent evolution</h1>
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  if (error && lessons.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Agent evolution</h1>
        <p className="page-sub" style={{ color: 'var(--bad)' }}>
          {error}
        </p>
        <button className="btn" onClick={load}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="content">
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          marginBottom: 28,
          gap: 16,
        }}
      >
        <div>
          <h1 className="page-title">Agent evolution</h1>
          <p className="page-sub" style={{ margin: 0 }}>
            Everything your agent has learned, why it changed, and how those changes performed
            against its evals.
          </p>
        </div>
        <button className="btn ghost sm" onClick={() => onNav('talk')}>
          <Icon name="chat" size={14} /> Ask the agent
        </button>
      </div>

      {error && (
        <div className="card card-pad" style={{ borderColor: 'var(--bad)', marginBottom: 16 }}>
          <p className="dim" style={{ color: 'var(--bad)', margin: 0 }}>
            {error}
          </p>
        </div>
      )}

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
                The agent has {pending.length} change{pending.length > 1 ? 's' : ''} waiting for
                your review
              </div>
              <div className="dim" style={{ fontSize: 12.5, marginTop: 2 }}>
                Review and approve, or let it auto-rollback per policy.
              </div>
            </div>
            <button className="btn primary" onClick={() => onNav('review')}>
              Review now <Icon name="chevron" size={14} />
            </button>
          </div>
        </div>
      )}

      {lessons.length === 0 ? (
        <div className="card card-pad" style={{ textAlign: 'center', padding: '48px 20px' }}>
          <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 6 }}>
            No lessons yet
          </div>
          <div className="dim" style={{ fontSize: 13, maxWidth: 480, margin: '0 auto' }}>
            The agent hasn't promoted any change yet. Run the harness loop to generate
            candidates — promotions land here as they happen.
          </div>
        </div>
      ) : (
        <>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 16,
            }}
          >
            <h2 style={{ fontSize: 16, fontWeight: 600, margin: 0, letterSpacing: '-0.01em' }}>
              What I've been learning
            </h2>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className={`btn sm ${filter === 'all' ? '' : 'ghost'}`}
                onClick={() => setFilter('all')}
              >
                All
              </button>
              <button
                className={`btn sm ${filter === 'approved' ? '' : 'ghost'}`}
                onClick={() => setFilter('approved')}
              >
                Approved
              </button>
              <button
                className={`btn sm ${filter === 'rolled_back' ? '' : 'ghost'}`}
                onClick={() => setFilter('rolled_back')}
              >
                Rolled back
              </button>
            </div>
          </div>

          <div className="timeline">
            {filtered.map((l) => {
              const overall = l.delta?.overall_score;
              return (
                <div className="tl-item" key={l.id}>
                  <div className={`tl-node ${tlNodeClass(l)}`} />
                  <div className="tl-date">{fmtDay(l.promoted_at)}</div>
                  <div className="card lesson-card" onClick={() => onOpenLesson(l.id)}>
                    <div className="head">
                      <Tag>
                        <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
                      </Tag>
                      {l.version && (
                        <span className="mono dim" style={{ fontSize: 11.5 }}>
                          {l.version}
                        </span>
                      )}
                      <div style={{ marginLeft: 'auto' }}>
                        <StatusTag status={l.status} />
                      </div>
                    </div>
                    <div className="quote">{l.voice ? `"${l.voice}"` : l.summary}</div>
                    <div className="footing">
                      <div className="stats">
                        {l.mutations.length > 0 && (
                          <span className="stat" style={{ fontFamily: 'var(--font-mono)' }}>
                            <Icon name="code" size={12} /> {l.mutations[0]}
                            {l.mutations.length > 1 && ` +${l.mutations.length - 1}`}
                          </span>
                        )}
                        {typeof overall === 'number' && (
                          <span className="stat">
                            <Icon name="bolt" size={12} />
                            Δoverall {overall >= 0 ? '+' : ''}
                            {overall.toFixed(4)}
                          </span>
                        )}
                      </div>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        Open <Icon name="chevron" size={12} />
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
            {filtered.length === 0 && (
              <div className="dim" style={{ padding: '24px 0', fontSize: 13 }}>
                No lessons match this filter.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
