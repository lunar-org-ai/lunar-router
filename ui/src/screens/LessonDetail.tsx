/**
 * LessonDetail — drills into one promotion.
 *
 * The five tabs map cleanly onto the AHE three pillars (arxiv 2604.25850):
 *
 *   Story     → decision pillar    (the agent's first-person rationale)
 *   Diff      → component pillar   (the mutation applied to the agent config)
 *   Evals     → experience pillar  (rubric delta from the candidate run)
 *   Traces    → experience pillar  (which traces drove the change — TODO link)
 *   Decision  → decision pillar    (proposal_source, parent_version, lineage)
 *
 * The harness today produces minimal but real data for Story/Diff/Evals/Decision.
 * Traces and per-rubric before-numbers are not yet linked at the lesson level —
 * we show honest empty states instead of fabricating placeholder rows.
 */

import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag, StatusTag, KindIcon, KindLabel } from '../components/Tag';
import {
  ApiError,
  getLesson,
  getLessonTraces,
  getPolicy,
  type LessonSummary,
  type LessonTracesResponse,
  type PolicyView,
} from '../api';

const fmtDay = (iso: string | null): string => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return iso;
  }
};

const proposalSourceLabel = (src: string | null): string => {
  if (!src) return 'Unknown source';
  return (
    {
      heuristic: 'Heuristic proposer (sweep)',
      heuristic_sweep: 'Heuristic proposer (sweep)',
      claude_code: 'Claude Code proposer',
      human: 'Human proposer',
    }[src] || src
  );
};

const decisionLabel = (status: string): string => {
  switch (status) {
    case 'auto_promoted':
      return 'auto_promote';
    case 'awaiting_review':
      return 'queue_human';
    case 'approved':
      return 'human_approve';
    case 'human_rejected':
      return 'human_reject';
    case 'rolled_back':
      return 'rollback';
    default:
      return status;
  }
};

const renderRule = (status: string, policy: PolicyView | null) => {
  const mode = policy?.mode || 'review';
  const lift = policy?.auto_min_lift ?? 0.01;
  const liftPct = (lift * 100).toFixed(0);

  if (status === 'auto_promoted') {
    return (
      <>
        <div className="dim">if policy.mode == "auto"</div>
        <div className="dim">and Δoverall &gt;= {liftPct} pp</div>
        <div>
          → <span style={{ color: 'var(--accent-fg)' }}>auto_promote</span>
        </div>
      </>
    );
  }
  if (status === 'awaiting_review') {
    return (
      <>
        <div className="dim">if policy.mode == "{mode}"</div>
        <div>
          → <span style={{ color: 'var(--warn-fg)' }}>queue_human</span>
        </div>
      </>
    );
  }
  if (status === 'approved') {
    return (
      <>
        <div className="dim">if policy.mode == "{mode}" → queue_human</div>
        <div>
          then human →{' '}
          <span style={{ color: 'var(--accent-fg)' }}>approve + promote</span>
        </div>
      </>
    );
  }
  if (status === 'human_rejected') {
    return (
      <>
        <div className="dim">if policy.mode == "{mode}" → queue_human</div>
        <div>
          then human →{' '}
          <span style={{ color: 'var(--bad-fg)' }}>reject</span>
        </div>
      </>
    );
  }
  if (status === 'rolled_back') {
    return (
      <>
        <div className="dim">post-promote regression detected</div>
        <div>
          → <span style={{ color: 'var(--bad-fg)' }}>rollback</span>
        </div>
      </>
    );
  }
  return <div className="dim">No rule recorded for status "{status}".</div>;
};

type TabId = 'story' | 'traces' | 'evals' | 'diff' | 'decision';

export const LessonDetail = ({
  lessonId,
  onBack,
  onNav,
}: {
  lessonId: string;
  onBack: () => void;
  onNav?: (route: string) => void;
}) => {
  const [lesson, setLesson] = useState<LessonSummary | null>(null);
  const [traces, setTraces] = useState<LessonTracesResponse | null>(null);
  const [policy, setPolicy] = useState<PolicyView | null>(null);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>('story');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [l, pol] = await Promise.all([
        getLesson(lessonId),
        getPolicy().catch(() => null),
      ]);
      setLesson(l);
      if (pol) setPolicy(pol);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Backend ${e.status}: ${e.message}`
          : `Network error: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setLoading(false);
    }
  }, [lessonId]);

  useEffect(() => {
    void load();
  }, [load]);

  // Lazy-load traces only when the Traces tab is opened (and cache by lesson).
  useEffect(() => {
    if (tab !== 'traces') return;
    if (traces?.lesson_id === lessonId) return;
    setTracesLoading(true);
    getLessonTraces(lessonId)
      .then(setTraces)
      .catch((e) => {
        setError(
          e instanceof ApiError
            ? `Traces ${e.status}: ${e.message}`
            : `Traces error: ${e instanceof Error ? e.message : String(e)}`,
        );
      })
      .finally(() => setTracesLoading(false));
  }, [tab, lessonId, traces]);

  if (loading) {
    return (
      <div className="content">
        <button className="btn ghost sm" onClick={onBack} style={{ marginBottom: 14, marginLeft: -8 }}>
          <Icon name="chevron" size={12} style={{ transform: 'rotate(180deg)' }} /> Back to evolution
        </button>
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  if (error || !lesson) {
    return (
      <div className="content">
        <button className="btn ghost sm" onClick={onBack} style={{ marginBottom: 14, marginLeft: -8 }}>
          <Icon name="chevron" size={12} style={{ transform: 'rotate(180deg)' }} /> Back to evolution
        </button>
        <p className="page-sub" style={{ color: 'var(--bad)' }}>
          {error || 'Lesson not found.'}
        </p>
        <button className="btn" onClick={load}>
          Retry
        </button>
      </div>
    );
  }

  const l = lesson;
  const perRubric = l.delta?.per_rubric ?? {};
  const rubricKeys = Object.keys(perRubric);
  const overall = l.delta?.overall_score;
  const passRate = l.delta?.pass_rate;

  const traceCount = traces?.cases.length ?? l.n_traces ?? undefined;
  const tabs: Array<{ id: TabId; label: string; count?: number }> = [
    { id: 'story', label: 'Story' },
    { id: 'traces', label: 'Traces', count: traceCount },
    { id: 'evals', label: 'Evals', count: rubricKeys.length },
    { id: 'diff', label: 'Diff', count: l.mutations.length },
    { id: 'decision', label: 'Decision' },
  ];

  return (
    <div className="content">
      <button className="btn ghost sm" onClick={onBack} style={{ marginBottom: 14, marginLeft: -8 }}>
        <Icon name="chevron" size={12} style={{ transform: 'rotate(180deg)' }} /> Back to evolution
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
        <Tag>
          <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
        </Tag>
        <span className="mono dim" style={{ fontSize: 12 }}>{l.id}</span>
        <span className="dim" style={{ fontSize: 12 }}>·</span>
        {l.version && (
          <>
            <span className="mono dim" style={{ fontSize: 12 }}>{l.version}</span>
            <span className="dim" style={{ fontSize: 12 }}>·</span>
          </>
        )}
        <span className="dim" style={{ fontSize: 12 }}>{fmtDay(l.promoted_at)}</span>
        <div style={{ marginLeft: 'auto' }}>
          <StatusTag status={l.status} />
        </div>
      </div>

      <h1 className="page-title">{l.title}</h1>
      <p className="page-sub">{l.summary}</p>

      <div className="tabs">
        {tabs.map((t) => (
          <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
            {t.label}
            {t.count != null && t.count > 0 && <span className="count">{t.count}</span>}
          </button>
        ))}
      </div>

      {tab === 'story' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 32 }}>
          <div>
            {l.voice && (
              <div className="msg msg-agent" style={{ marginBottom: 24 }}>
                <div className="msg-role">In my own words</div>
                <p
                  className="msg-body"
                  style={{ fontStyle: 'italic', fontSize: 16, lineHeight: 1.6 }}
                >
                  "{l.voice}"
                </p>
              </div>
            )}

            <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px' }}>What changed</h3>
            <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--fg)', margin: '0 0 24px' }}>
              {l.summary}
            </p>

            <h3 style={{ fontSize: 14, fontWeight: 600, margin: '24px 0 10px' }}>What triggered it</h3>
            <div className="card card-pad">
              <div className="dim" style={{ fontSize: 13 }}>
                Trigger lineage isn't linked to lessons yet. When the harness wires Prediction
                rationale and trace cohorts to each lesson, it'll appear here — for now the
                Decision tab carries what we can reconstruct.
              </div>
            </div>
          </div>
          <aside style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card card-pad">
              <div
                className="dim"
                style={{
                  fontSize: 11,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  marginBottom: 10,
                }}
              >
                Eval movement
              </div>
              {typeof overall === 'number' ? (
                <>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: '8px 0',
                      borderBottom: '1px solid var(--border)',
                      fontSize: 13,
                    }}
                  >
                    <span>Δ overall</span>
                    <span
                      className="mono"
                      style={{
                        color:
                          overall > 0
                            ? 'var(--accent-fg)'
                            : overall < 0
                            ? 'var(--bad-fg)'
                            : 'var(--fg-muted)',
                        fontWeight: 500,
                      }}
                    >
                      {overall >= 0 ? '+' : ''}
                      {overall.toFixed(4)}
                    </span>
                  </div>
                  {typeof passRate === 'number' && (
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '8px 0',
                        fontSize: 13,
                      }}
                    >
                      <span>Δ pass rate</span>
                      <span
                        className="mono"
                        style={{
                          color:
                            passRate > 0
                              ? 'var(--accent-fg)'
                              : passRate < 0
                              ? 'var(--bad-fg)'
                              : 'var(--fg-muted)',
                          fontWeight: 500,
                        }}
                      >
                        {passRate >= 0 ? '+' : ''}
                        {passRate.toFixed(4)}
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <div className="dim" style={{ fontSize: 13 }}>
                  No eval delta recorded.
                </div>
              )}
            </div>
          </aside>
        </div>
      )}

      {tab === 'traces' && (
        <>
          {tracesLoading && !traces && (
            <div className="dim" style={{ padding: '24px 0', fontSize: 13 }}>Loading traces…</div>
          )}
          {traces && !traces.has_report && (
            <div className="card card-pad">
              <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}>
                Trace lineage not captured
              </div>
              <div className="dim" style={{ fontSize: 13, lineHeight: 1.6 }}>
                {traces.note ||
                  'No candidate report on disk for this lesson. Future lessons will carry full trace lineage.'}
              </div>
              {l.ledger_entry_id && (
                <div
                  className="dim"
                  style={{ fontSize: 12, marginTop: 12, fontFamily: 'var(--font-mono)' }}
                >
                  Ledger entry: {l.ledger_entry_id}
                </div>
              )}
            </div>
          )}
          {traces && traces.has_report && (
            <>
              <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
                {traces.cases.length} cases from suite{' '}
                <span className="mono">{traces.suite || '—'}</span>
                {traces.agent_version && (
                  <>
                    {' '}
                    · agent <span className="mono">{traces.agent_version}</span>
                  </>
                )}
                {traces.candidate_id && (
                  <>
                    {' '}
                    · candidate <span className="mono">{traces.candidate_id}</span>
                  </>
                )}
              </div>
              <div className="card">
                <div
                  style={{
                    padding: '12px 16px',
                    borderBottom: '1px solid var(--border)',
                    fontSize: 12,
                    color: 'var(--fg-muted)',
                    display: 'grid',
                    gridTemplateColumns: '90px 1fr 90px 80px',
                    gap: 12,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    fontWeight: 500,
                  }}
                >
                  <div>Golden</div>
                  <div>Request → response</div>
                  <div>Rubrics</div>
                  <div>Verdict</div>
                </div>
                {traces.cases.map((c, i) => {
                  const passed = c.rubric_results.filter((r) => r.passed).length;
                  const total = c.rubric_results.length;
                  const allPassed = c.success && passed === total && total > 0;
                  const respPreview = (c.response || c.error || '').slice(0, 120);
                  return (
                    <div
                      key={c.trace_id || i}
                      style={{
                        padding: '12px 16px',
                        borderBottom:
                          i < traces.cases.length - 1 ? '1px solid var(--border)' : 'none',
                        display: 'grid',
                        gridTemplateColumns: '90px 1fr 90px 80px',
                        gap: 12,
                        alignItems: 'center',
                        fontSize: 13,
                      }}
                    >
                      <span className="mono dim" style={{ fontSize: 11.5 }}>
                        {c.golden_id}
                      </span>
                      <div style={{ minWidth: 0 }}>
                        <div
                          style={{
                            fontWeight: 500,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {c.request}
                        </div>
                        <div
                          className="dim"
                          style={{
                            fontSize: 12,
                            marginTop: 2,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {respPreview || '(no response)'}
                        </div>
                      </div>
                      <span className="mono" style={{ fontSize: 12 }}>
                        {passed}/{total}
                      </span>
                      <Tag kind={allPassed ? 'success' : 'bad'}>
                        <span className="dot" /> {allPassed ? 'Passed' : 'Failed'}
                      </Tag>
                    </div>
                  );
                })}
                {traces.cases.length === 0 && (
                  <div className="empty" style={{ padding: 32 }}>
                    Report exists but has no cases.
                  </div>
                )}
              </div>
            </>
          )}
        </>
      )}

      {tab === 'evals' && (
        <div className="card">
          <div
            style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border)',
              fontSize: 12,
              color: 'var(--fg-muted)',
              display: 'grid',
              gridTemplateColumns: '1fr 120px',
              gap: 12,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontWeight: 500,
            }}
          >
            <div>Rubric</div>
            <div style={{ textAlign: 'right' }}>Δ vs baseline</div>
          </div>
          {rubricKeys.length === 0 && (
            <div className="empty">No per-rubric delta recorded for this lesson.</div>
          )}
          {rubricKeys.map((rubric) => {
            const v = perRubric[rubric] ?? 0;
            const color =
              v > 0 ? 'var(--accent-fg)' : v < 0 ? 'var(--bad-fg)' : 'var(--fg-muted)';
            return (
              <div
                key={rubric}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 120px',
                  gap: 12,
                  padding: '12px 16px',
                  borderBottom: '1px solid var(--border)',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontWeight: 500, fontSize: 14 }}>{rubric}</div>
                </div>
                <div className="mono" style={{ textAlign: 'right', color, fontWeight: 500 }}>
                  {v >= 0 ? '+' : ''}
                  {v.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {tab === 'diff' && (
        <div>
          <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
            Mutations applied to the agent config in this version.
          </div>
          {l.mutations.length === 0 ? (
            <div className="card card-pad">
              <div className="dim" style={{ fontSize: 13 }}>No mutations recorded.</div>
            </div>
          ) : (
            <div className="card">
              {l.mutations.map((m, i) => (
                <div
                  key={i}
                  style={{
                    padding: '12px 16px',
                    borderBottom: i < l.mutations.length - 1 ? '1px solid var(--border)' : 'none',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 13,
                  }}
                >
                  {m}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === 'decision' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card card-pad">
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
              How the system decided
            </div>
            <ol style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, lineHeight: 1.7 }}>
              <li>
                Proposal came from{' '}
                <span className="mono">{proposalSourceLabel(l.proposal_source)}</span>
                {l.mutations.length > 0 && (
                  <>
                    {' '}— {l.mutations.length === 1
                      ? 'one mutation'
                      : `${l.mutations.length} mutations`}{' '}
                    on <span className="mono">{l.mutations[0].split(':')[0]}</span>
                  </>
                )}
              </li>
              <li>
                Branched candidate <span className="mono">{l.candidate_id || '—'}</span> from{' '}
                <span className="mono">{l.parent_version || 'initial'}</span>
              </li>
              <li>
                Ran offline against{' '}
                <span className="mono">{l.n_traces ?? '—'}</span> eval cases; saw{' '}
                {typeof overall === 'number'
                  ? `Δoverall ${overall >= 0 ? '+' : ''}${(overall * 100).toFixed(2)} pp`
                  : 'no recorded delta'}
              </li>
              <li>
                Routed to{' '}
                <span className="mono">{decisionLabel(l.status)}</span> per policy
              </li>
            </ol>
          </div>
          <div className="card card-pad">
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Rules that fired</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div
                style={{
                  padding: 10,
                  background: 'var(--bg-muted)',
                  borderRadius: 8,
                  fontSize: 12.5,
                  fontFamily: 'var(--font-mono)',
                  lineHeight: 1.7,
                }}
              >
                {renderRule(l.status, policy)}
              </div>
              <button
                className="btn sm ghost"
                style={{ alignSelf: 'flex-start' }}
                onClick={() => onNav?.('policies')}
              >
                <Icon name="settings" size={12} /> Edit policies
              </button>
            </div>
            {l.ledger_entry_id && (
              <div
                className="dim mono"
                style={{ fontSize: 11.5, marginTop: 14, fontFamily: 'var(--font-mono)' }}
              >
                Ledger entry: {l.ledger_entry_id}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
