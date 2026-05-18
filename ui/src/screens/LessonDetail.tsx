/**
 * LessonDetail — drills into one promotion.
 *
 * The five tabs map onto the AHE three pillars (arxiv 2604.25850):
 *   Story     → decision pillar
 *   Diff      → component pillar
 *   Evals     → experience pillar
 *   Traces    → experience pillar
 *   Decision  → decision pillar
 *
 * Tab is in the URL search param (?tab=...). Traces are lazy-loaded.
 */

import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams, useSearch } from '@tanstack/react-router';
import { Icon } from '../components/Icon';
import { Tag, StatusTag, KindIcon, KindLabel } from '../components/Tag';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { preserveSearch } from '../router';
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

const deltaColor = (v: number): string =>
  v > 0 ? 'var(--accent-fg)' : v < 0 ? 'var(--bad-fg)' : 'var(--muted-foreground)';

const fmtDelta = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(4)}`;

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
          then human → <span style={{ color: 'var(--accent-fg)' }}>approve + promote</span>
        </div>
      </>
    );
  }
  if (status === 'human_rejected') {
    return (
      <>
        <div className="dim">if policy.mode == "{mode}" → queue_human</div>
        <div>
          then human → <span style={{ color: 'var(--bad-fg)' }}>reject</span>
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

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div className="dim mb-2.5 text-[11px] font-medium uppercase tracking-[0.05em]">{children}</div>
);

const BackLink = () => (
  <Button asChild variant="ghost" size="sm" className="-ml-2 mb-3.5">
    <Link to="/" search={preserveSearch}>
      <Icon name="chevron" size={12} style={{ transform: 'rotate(180deg)' }} /> Back to evolution
    </Link>
  </Button>
);

export const LessonDetail = () => {
  const { id: lessonId } = useParams({ from: '/lesson/$id' });
  const search = useSearch({ from: '/lesson/$id' });
  const tab: TabId = search.tab ?? 'story';
  const navigate = useNavigate();
  const setTab = (next: TabId) =>
    navigate({
      to: '/lesson/$id',
      params: { id: lessonId },
      search: (prev) => ({ ...prev, tab: next }),
    });

  const [lesson, setLesson] = useState<LessonSummary | null>(null);
  const [traces, setTraces] = useState<LessonTracesResponse | null>(null);
  const [policy, setPolicy] = useState<PolicyView | null>(null);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
        <BackLink />
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  if (error || !lesson) {
    return (
      <div className="content">
        <BackLink />
        <p className="page-sub text-destructive">{error || 'Lesson not found.'}</p>
        <Button onClick={load}>Retry</Button>
      </div>
    );
  }

  const l = lesson;
  const perRubric = l.delta?.per_rubric ?? {};
  const rubricKeys = Object.keys(perRubric);
  const overall = l.delta?.overall_score;
  const passRate = l.delta?.pass_rate;

  const traceCount = traces?.cases.length ?? l.n_traces ?? undefined;
  const tabDefs: Array<{ id: TabId; label: string; count?: number }> = [
    { id: 'story', label: 'Story' },
    { id: 'traces', label: 'Traces', count: traceCount },
    { id: 'evals', label: 'Evals', count: rubricKeys.length },
    { id: 'diff', label: 'Diff', count: l.mutations.length },
    { id: 'decision', label: 'Decision' },
  ];

  return (
    <div className="content">
      <BackLink />

      <div className="mb-2 flex flex-wrap items-center gap-2.5">
        <Tag>
          <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
        </Tag>
        <span className="dim mono text-xs">{l.id}</span>
        <span className="dim text-xs">·</span>
        {l.version && (
          <>
            <span className="dim mono text-xs">{l.version}</span>
            <span className="dim text-xs">·</span>
          </>
        )}
        <span className="dim text-xs">{fmtDay(l.promoted_at)}</span>
        <div className="ml-auto">
          <StatusTag status={l.status} />
        </div>
      </div>

      <h1 className="page-title">{l.title}</h1>
      <p className="page-sub">{l.summary}</p>

      <Tabs value={tab} onValueChange={(v) => setTab(v as TabId)} className="mt-2 gap-6">
        <TabsList variant="line">
          {tabDefs.map((t) => (
            <TabsTrigger key={t.id} value={t.id}>
              {t.label}
              {t.count != null && t.count > 0 && (
                <span className="ml-1 rounded-full bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
                  {t.count}
                </span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>

        {/* STORY */}
        <TabsContent value="story">
          <div className="grid grid-cols-[1fr_280px] gap-8">
            <div>
              {l.voice && (
                <div className="mb-6 rounded-[var(--radius)] bg-muted px-4 py-3.5">
                  <div className="dim mb-1.5 text-xs font-medium">In my own words</div>
                  <p className="text-base italic leading-relaxed text-foreground">"{l.voice}"</p>
                </div>
              )}

              <h3 className="mb-2.5 text-sm font-semibold">What changed</h3>
              <p className="mb-6 text-sm leading-relaxed text-foreground">{l.summary}</p>

              <h3 className="mt-6 mb-2.5 text-sm font-semibold">What triggered it</h3>
              <Card className="px-4 py-3.5">
                <div className="dim text-[13px]">
                  Trigger lineage isn't linked to lessons yet. When the harness wires Prediction
                  rationale and trace cohorts to each lesson, it'll appear here — for now the
                  Decision tab carries what we can reconstruct.
                </div>
              </Card>
            </div>
            <aside className="flex flex-col gap-3">
              <Card className="px-4 py-3.5">
                <SectionLabel>Eval movement</SectionLabel>
                {typeof overall === 'number' ? (
                  <>
                    <div className="flex items-center justify-between border-b border-border py-2 text-[13px]">
                      <span>Δ overall</span>
                      <span
                        className="mono font-medium"
                        style={{ color: deltaColor(overall) }}
                      >
                        {fmtDelta(overall)}
                      </span>
                    </div>
                    {typeof passRate === 'number' && (
                      <div className="flex items-center justify-between py-2 text-[13px]">
                        <span>Δ pass rate</span>
                        <span
                          className="mono font-medium"
                          style={{ color: deltaColor(passRate) }}
                        >
                          {fmtDelta(passRate)}
                        </span>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="dim text-[13px]">No eval delta recorded.</div>
                )}
              </Card>
            </aside>
          </div>
        </TabsContent>

        {/* TRACES */}
        <TabsContent value="traces">
          {tracesLoading && !traces && <div className="dim text-[13px] py-6">Loading traces…</div>}
          {traces && !traces.has_report && (
            <Card className="px-4 py-3.5">
              <div className="mb-2 text-sm font-medium">Trace lineage not captured</div>
              <div className="dim text-[13px] leading-relaxed">
                {traces.note ||
                  'No candidate report on disk for this lesson. Future lessons will carry full trace lineage.'}
              </div>
              {l.ledger_entry_id && (
                <div className="dim mono mt-3 text-xs">Ledger entry: {l.ledger_entry_id}</div>
              )}
            </Card>
          )}
          {traces && traces.has_report && (
            <>
              <div className="dim mb-3 text-[12.5px]">
                {traces.cases.length} cases from suite{' '}
                <span className="mono">{traces.suite || '—'}</span>
                {traces.agent_version && (
                  <>
                    {' '}· agent <span className="mono">{traces.agent_version}</span>
                  </>
                )}
                {traces.candidate_id && (
                  <>
                    {' '}· candidate <span className="mono">{traces.candidate_id}</span>
                  </>
                )}
              </div>
              <Card className="gap-0 py-0">
                <div className="dim grid grid-cols-[90px_1fr_90px_80px] gap-3 border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-[0.05em]">
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
                      className={`grid grid-cols-[90px_1fr_90px_80px] items-center gap-3 px-4 py-3 text-[13px] ${
                        i < traces.cases.length - 1 ? 'border-b border-border' : ''
                      }`}
                    >
                      <span className="mono dim text-[11.5px]">{c.golden_id}</span>
                      <div className="min-w-0">
                        <div className="truncate font-medium">{c.request}</div>
                        <div className="dim mt-0.5 truncate text-xs">
                          {respPreview || '(no response)'}
                        </div>
                      </div>
                      <span className="mono text-xs">
                        {passed}/{total}
                      </span>
                      <Tag kind={allPassed ? 'success' : 'bad'}>
                        {allPassed ? 'Passed' : 'Failed'}
                      </Tag>
                    </div>
                  );
                })}
                {traces.cases.length === 0 && (
                  <div className="empty p-8">Report exists but has no cases.</div>
                )}
              </Card>
            </>
          )}
        </TabsContent>

        {/* EVALS */}
        <TabsContent value="evals">
          <Card className="gap-0 py-0">
            <div className="dim grid grid-cols-[1fr_120px] gap-3 border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-[0.05em]">
              <div>Rubric</div>
              <div className="text-right">Δ vs baseline</div>
            </div>
            {rubricKeys.length === 0 && (
              <div className="empty p-8">No per-rubric delta recorded for this lesson.</div>
            )}
            {rubricKeys.map((rubric, i) => {
              const v = perRubric[rubric] ?? 0;
              return (
                <div
                  key={rubric}
                  className={`grid grid-cols-[1fr_120px] items-center gap-3 px-4 py-3 ${
                    i < rubricKeys.length - 1 ? 'border-b border-border' : ''
                  }`}
                >
                  <div className="text-sm font-medium">{rubric}</div>
                  <div
                    className="mono text-right font-medium"
                    style={{ color: deltaColor(v) }}
                  >
                    {fmtDelta(v)}
                  </div>
                </div>
              );
            })}
          </Card>
        </TabsContent>

        {/* DIFF */}
        <TabsContent value="diff">
          <div className="dim mb-3 text-[12.5px]">
            Mutations applied to the agent config in this version.
          </div>
          {l.mutations.length === 0 ? (
            <Card className="px-4 py-3.5">
              <div className="dim text-[13px]">No mutations recorded.</div>
            </Card>
          ) : (
            <Card className="gap-0 py-0">
              {l.mutations.map((m, i) => (
                <div
                  key={i}
                  className={`mono px-4 py-3 text-[13px] ${
                    i < l.mutations.length - 1 ? 'border-b border-border' : ''
                  }`}
                >
                  {m}
                </div>
              ))}
            </Card>
          )}
        </TabsContent>

        {/* DECISION */}
        <TabsContent value="decision">
          <div className="grid grid-cols-2 gap-4">
            <Card className="px-4 py-3.5">
              <div className="mb-3 text-[13px] font-semibold">How the system decided</div>
              <ol className="m-0 list-decimal pl-5 text-[13.5px] leading-[1.7]">
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
                  Ran offline against <span className="mono">{l.n_traces ?? '—'}</span> eval cases;
                  saw{' '}
                  {typeof overall === 'number'
                    ? `Δoverall ${overall >= 0 ? '+' : ''}${(overall * 100).toFixed(2)} pp`
                    : 'no recorded delta'}
                </li>
                <li>
                  Routed to <span className="mono">{decisionLabel(l.status)}</span> per policy
                </li>
              </ol>
            </Card>
            <Card className="px-4 py-3.5">
              <div className="mb-3 text-[13px] font-semibold">Rules that fired</div>
              <div className="flex flex-col gap-2.5">
                <div className="mono rounded-md bg-muted p-2.5 text-[12.5px] leading-[1.7]">
                  {renderRule(l.status, policy)}
                </div>
                <Button asChild variant="ghost" size="sm" className="self-start">
                  <Link to="/policies" search={preserveSearch}>
                    <Icon name="settings" size={12} /> Edit policies
                  </Link>
                </Button>
              </div>
              {l.ledger_entry_id && (
                <div className="dim mono mt-3.5 text-[11.5px]">
                  Ledger entry: {l.ledger_entry_id}
                </div>
              )}
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
