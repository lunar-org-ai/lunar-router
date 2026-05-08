/**
 * Review — human-in-the-loop queue.
 *
 * Lists lessons with status="awaiting_review" (provisional lessons the harness
 * loop wrote when policy.mode=review). Approving runs promote_queued() on the
 * backend (snapshots live, copies the candidate dir over, bumps version,
 * writes a promote ledger entry); rejecting writes a rejected entry. Either
 * way the lesson is mutated on disk and the timeline reflects it.
 */

import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag, KindIcon, KindLabel } from '../components/Tag';
import {
  ApiError,
  approveLesson,
  listLessons,
  rejectLesson,
  requeueLesson,
  type LessonSummary,
} from '../api';

const fmtRubric = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(4)}`;

export const Review = ({ onOpenLesson }: { onOpenLesson: (id: string) => void }) => {
  const [queue, setQueue] = useState<LessonSummary[]>([]);
  const [decided, setDecided] = useState<
    { id: string; title: string; action: 'approve' | 'reject' }[]
  >([]);
  const [acting, setActing] = useState<Record<string, 'approve' | 'reject' | null>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const all = await listLessons();
      setQueue(all.filter((l) => l.status === 'awaiting_review'));
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
    void refresh();
  }, [refresh]);

  const onAction = async (l: LessonSummary, action: 'approve' | 'reject') => {
    setActing((s) => ({ ...s, [l.id]: action }));
    try {
      if (action === 'approve') {
        await approveLesson(l.id, 'ui');
      } else {
        await rejectLesson(l.id);
      }
      setDecided((d) => [...d, { id: l.id, title: l.title, action }]);
      await refresh();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${action} failed: ${e.status} — ${e.message}`
          : `${action} failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setActing((s) => ({ ...s, [l.id]: null }));
    }
  };

  if (loading && queue.length === 0 && decided.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Pending review</h1>
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  return (
    <div className="content">
      <h1 className="page-title">Pending review</h1>
      <p className="page-sub">
        The agent has proposed changes based on what it learned. Approve to ship them, reject to
        discard, or wait — anything untouched in 24h follows your default policy.
      </p>

      {error && (
        <div className="card card-pad" style={{ borderColor: 'var(--bad)', marginBottom: 16 }}>
          <p className="dim" style={{ color: 'var(--bad)', margin: 0 }}>
            {error}
          </p>
        </div>
      )}

      {queue.length === 0 && (
        <div
          className="card card-pad empty"
          style={{
            padding: 64,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
          }}
        >
          <Icon name="check" size={32} />
          <div style={{ fontSize: 16, fontWeight: 500, color: 'var(--fg)', marginTop: 12 }}>
            Inbox zero.
          </div>
          <div style={{ marginTop: 4 }}>
            The agent will check in again when it has something new to suggest.
          </div>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {queue.map((l) => {
          const perRubric = l.delta?.per_rubric ?? {};
          const rubricKeys = Object.keys(perRubric);
          const overall = l.delta?.overall_score;
          const busy = acting[l.id];

          return (
            <div className="proposal" key={l.id}>
              <div className="h">
                <Tag>
                  <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
                </Tag>
                <span className="title">{l.title}</span>
                <span
                  className="dim mono"
                  style={{ marginLeft: 'auto', fontSize: 11.5 }}
                >
                  {l.id}
                </span>
              </div>
              <div className="b">
                <div>
                  {l.voice && <div className="quote">"{l.voice}"</div>}
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
                  {l.mutations.length === 0 ? (
                    <div className="dim" style={{ fontSize: 13 }}>No mutations recorded.</div>
                  ) : (
                    <div
                      className="card"
                      style={{
                        background: 'var(--bg-muted)',
                        padding: '10px 12px',
                        fontFamily: 'var(--font-mono)',
                        fontSize: 12.5,
                        lineHeight: 1.7,
                      }}
                    >
                      {l.mutations.map((m, i) => (
                        <div key={i}>{m}</div>
                      ))}
                    </div>
                  )}
                  <div
                    className="dim"
                    style={{ fontSize: 12, marginTop: 10, lineHeight: 1.5 }}
                  >
                    Branched from{' '}
                    <span className="mono">{l.parent_version || '—'}</span>; candidate ID{' '}
                    <span className="mono">{l.id}</span>
                  </div>
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
                    Eval movement
                  </div>
                  {typeof overall === 'number' && (
                    <div
                      style={{
                        padding: '10px 12px',
                        background: 'var(--bg-muted)',
                        borderRadius: 'var(--radius)',
                        marginBottom: 10,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: 13,
                      }}
                    >
                      <span>Δ overall</span>
                      <span
                        className="mono"
                        style={{
                          fontWeight: 500,
                          color:
                            overall > 0
                              ? 'var(--accent-fg)'
                              : overall < 0
                              ? 'var(--bad-fg)'
                              : 'var(--fg-muted)',
                        }}
                      >
                        {fmtRubric(overall)}
                      </span>
                    </div>
                  )}
                  {rubricKeys.length > 0 && (
                    <div
                      style={{
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius)',
                        overflow: 'hidden',
                        marginBottom: 14,
                      }}
                    >
                      {rubricKeys.map((k, i) => {
                        const v = perRubric[k] ?? 0;
                        return (
                          <div
                            key={k}
                            style={{
                              padding: '8px 10px',
                              borderBottom:
                                i < rubricKeys.length - 1 ? '1px solid var(--border)' : 'none',
                              display: 'flex',
                              justifyContent: 'space-between',
                              fontSize: 12.5,
                            }}
                          >
                            <span>{k}</span>
                            <span
                              className="mono"
                              style={{
                                color:
                                  v > 0
                                    ? 'var(--accent-fg)'
                                    : v < 0
                                    ? 'var(--bad-fg)'
                                    : 'var(--fg-muted)',
                                fontWeight: 500,
                              }}
                            >
                              {fmtRubric(v)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
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
                  Auto-decision in <span className="mono">24h</span> per policy
                </span>
                <button
                  className="btn danger"
                  onClick={() => onAction(l, 'reject')}
                  disabled={!!busy}
                >
                  <Icon name="x" size={14} /> {busy === 'reject' ? 'Rejecting…' : 'Reject'}
                </button>
                <button
                  className="btn success"
                  onClick={() => onAction(l, 'approve')}
                  disabled={!!busy}
                >
                  <Icon name="check" size={14} />{' '}
                  {busy === 'approve' ? 'Promoting…' : 'Approve & ship'}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {decided.length > 0 && (
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
          {decided.map(({ id, title, action }) => (
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
              <span>{title}</span>
              <button
                className="btn ghost sm"
                style={{ marginLeft: 'auto' }}
                disabled={!!acting[id]}
                onClick={async () => {
                  setActing((s) => ({ ...s, [id]: 'approve' }));
                  try {
                    await requeueLesson(id);
                    setDecided((d) => d.filter((x) => x.id !== id));
                    await refresh();
                  } catch (e) {
                    setError(
                      e instanceof ApiError
                        ? `Undo failed: ${e.status} — ${e.message}`
                        : `Undo failed: ${e instanceof Error ? e.message : String(e)}`,
                    );
                  } finally {
                    setActing((s) => ({ ...s, [id]: null }));
                  }
                }}
              >
                {acting[id] ? 'Undoing…' : 'Undo'}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
