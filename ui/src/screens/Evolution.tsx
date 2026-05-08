/**
 * Evolution — the home of the agent.
 *
 * AHE paper (arxiv 2604.25850) three pillars structure each lesson card:
 *   - decision   → voice (the agent's first-person rationale, top of card)
 *   - component  → mutations (the file/knob change, mono dim row)
 *   - experience → delta.overall_score (small Δ chip)
 *
 * The lesson timeline is the only panel running on real backend data today
 * (listLessons → /v1/lessons). The live strip, trust score, and overall
 * metrics row are still on illustrative values from data.ts — they'll move
 * to real signals when production telemetry lands.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from '@tanstack/react-router';
import { Icon } from '../components/Icon';
import { Sparkline } from '../components/Sparkline';
import { Tag, StatusTag, KindIcon, KindLabel } from '../components/Tag';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { preserveSearch } from '../router';
import { useRootContext } from '../routes/__root';
import {
  ApiError,
  getMetricsOverview,
  listLessons,
  type LessonSummary,
  type MetricsOverview,
} from '../api';

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

export const Evolution = () => {
  const { openAgent } = useRootContext();
  const dayZero = false;
  const [lessons, setLessons] = useState<LessonSummary[]>([]);
  const [metrics, setMetrics] = useState<MetricsOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [items, m] = await Promise.all([listLessons(), getMetricsOverview().catch(() => null)]);
      setLessons(items);
      if (m) setMetrics(m);
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

  const overallMetrics = useMemo(() => {
    const m = metrics;
    const trustValue = m ? String(m.trust_score) : '—';
    const trustDelta = m
      ? `${m.trust_score_delta_30d >= 0 ? '+' : ''}${m.trust_score_delta_30d} / 30d`
      : '—';
    const trustDir: 'up' | 'down' = m && m.trust_score_delta_30d >= 0 ? 'up' : 'down';

    const res = m?.resolution_rate;
    const resValue = res != null ? `${(res * 100).toFixed(0)}%` : '—';

    return [
      { label: 'Trust score', value: trustValue, delta: trustDelta, dir: trustDir },
      { label: 'Resolution rate', value: resValue, delta: '—', dir: 'up' as const },
      // No real cost / CSAT signal yet (stubs don't track tokens; no customer
      // feedback loop). These cells stay as illustrative placeholders.
      { label: 'Avg cost / conv', value: '—', delta: '—', dir: 'up' as const },
      { label: 'CSAT', value: '—', delta: '—', dir: 'up' as const },
    ];
  }, [metrics]);

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
          <Button onClick={openAgent}>
            Set up agent <Icon name="chevron" size={14} />
          </Button>
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
          <b>{metrics?.active_5min ?? '—'}</b>
          <span className="l">in conversation</span>
        </span>
        <span className="stat-mini">
          <b>{metrics?.today_count ?? '—'}</b>
          <span className="l">today</span>
        </span>
        <span className="stat-mini">
          <b>{metrics?.pending_review ?? pending.length}</b>
          <span className="l">flagged for review</span>
        </span>
        <span style={{ marginLeft: 'auto' }}>
          <Button asChild variant="ghost" size="sm">
            <Link to="/talk" search={preserveSearch}>
              Ask the agent <Icon name="chevron" size={12} />
            </Link>
          </Button>
        </span>
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          marginBottom: 28,
        }}
      >
        <div>
          <h1 className="page-title">Agent evolution</h1>
          <p className="page-sub" style={{ margin: 0 }}>
            Everything your agent has learned, why it changed, and how those changes performed in
            production.
          </p>
        </div>
        <Card className="flex flex-row items-center gap-4 px-4 py-3.5">
          <div>
            <div
              className="dim"
              style={{
                fontSize: 11,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                marginBottom: 2,
              }}
            >
              Trust score
            </div>
            <div style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.01em' }}>
              {metrics?.trust_score ?? '—'}
              <span className="dim" style={{ fontSize: 14, fontWeight: 400 }}>
                {' '}
                / 100
              </span>
            </div>
          </div>
          <Sparkline data={metrics?.trust_history_30d ?? []} w={180} h={48} />
        </Card>
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

      {error && (
        <Card className="mb-4 border-destructive p-4">
          <p className="dim m-0 text-destructive">{error}</p>
        </Card>
      )}

      {pending.length > 0 && (
        <Card
          className="mb-7 flex flex-row items-center gap-3 px-4 py-3.5"
          style={{
            borderColor: 'var(--warn-soft)',
            background: 'linear-gradient(0deg, var(--warn-soft) 0%, var(--card) 30%)',
          }}
        >
          <div
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full"
            style={{ background: 'var(--warn-soft)', color: 'var(--warn-fg)' }}
          >
            <Icon name="bell" size={16} />
          </div>
          <div className="flex-1">
            <div className="text-sm font-semibold">
              The agent has {pending.length} change{pending.length > 1 ? 's' : ''} waiting for your
              review
            </div>
            <div className="dim mt-0.5 text-xs">
              Review and approve, or let it roll back automatically in 24h.
            </div>
          </div>
          <Button asChild>
            <Link to="/review" search={preserveSearch}>
              Review now <Icon name="chevron" size={14} />
            </Link>
          </Button>
        </Card>
      )}

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
        <div className="flex gap-1.5">
          <Button
            variant={filter === 'all' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => setFilter('all')}
          >
            All
          </Button>
          <Button
            variant={filter === 'approved' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => setFilter('approved')}
          >
            Approved
          </Button>
          <Button
            variant={filter === 'rolled_back' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => setFilter('rolled_back')}
          >
            Rolled back
          </Button>
        </div>
      </div>

      {loading && lessons.length === 0 ? (
        <div className="dim" style={{ padding: '24px 0', fontSize: 13 }}>Loading lessons…</div>
      ) : lessons.length === 0 ? (
        <Card className="px-5 py-12 text-center">
          <div className="mb-1.5 text-[15px] font-medium">No lessons yet</div>
          <div className="dim mx-auto max-w-[480px] text-[13px]">
            The agent hasn't promoted any change yet. Run the harness loop to generate
            candidates — promotions land here as they happen.
          </div>
        </Card>
      ) : (
        <div className="timeline">
          {filtered.map((l) => {
            const overall = l.delta?.overall_score;
            return (
              <div className="tl-item" key={l.id}>
                <div className={`tl-node ${tlNodeClass(l)}`} />
                <div className="tl-date">{fmtDay(l.promoted_at)}</div>
                <Link
                  to="/lesson/$id"
                  params={{ id: l.id }}
                  search={preserveSearch}
                  className="card lesson-card"
                >
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
                      {typeof l.n_traces === 'number' && l.n_traces > 0 && (
                        <span className="stat">
                          <Icon name="timeline" size={12} /> {l.n_traces} trace
                          {l.n_traces > 1 ? 's' : ''}
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
                </Link>
              </div>
            );
          })}
          {filtered.length === 0 && (
            <div className="dim" style={{ padding: '24px 0', fontSize: 13 }}>
              No lessons match this filter.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
