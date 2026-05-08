import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import { Icon } from '../components/Icon';
import { Tag } from '../components/Tag';
import {
  ApiError,
  getTrace,
  listTraces,
  type TraceDetail,
  type TraceSummary,
  type TracesPage,
} from '../api';

// Shared helpers used by every Technical sub-screen — Esc closes drawers,
// toasts auto-dismiss after a couple seconds.

const useEscape = (onClose: () => void) => {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
};

const useAutoToast = (toast: string | null, setToast: (v: string | null) => void) => {
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2400);
    return () => clearTimeout(t);
  }, [toast, setToast]);
};

// ============================================================================
// Traces  — wired to real backend (P15.1)
// ============================================================================
//
// Source: traces/raw/<YYYY-MM-DD>.jsonl files surfaced by the runtime as a
// paginated feed (GET /v1/traces). Each row is one pipeline run; the drawer
// shows the per-stage breakdown (retrieve / rerank / route / generate).
//
// Filters (date / success / agent_version / search) round-trip to the
// backend — no client-side filtering, so the count is honest. Pagination is
// offset-based; we keep page state local and refetch when filters change.

const PAGE_SIZE = 50;

const fmtDuration = (ms: number): string => {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const fmtTimeOfDay = (iso: string): string => {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return iso.slice(11, 19);
  }
};

const fmtRelative = (iso: string): string => {
  if (!iso) return '';
  try {
    const ms = Date.now() - new Date(iso).getTime();
    if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
    if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
    if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
    return `${Math.round(ms / 86_400_000)}d ago`;
  } catch {
    return '';
  }
};

export const Traces = () => {
  const [page, setPage] = useState<TracesPage | null>(null);
  const [date, setDate] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'pass' | 'fail'>('all');
  const [versionFilter, setVersionFilter] = useState<string>('all');
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  // Debounce search to avoid hammering the backend on every keystroke.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search.trim()), 250);
    return () => clearTimeout(t);
  }, [search]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const opts: Parameters<typeof listTraces>[0] = {
        limit: PAGE_SIZE,
        offset,
      };
      if (date) opts.date = date;
      if (filter === 'pass') opts.success = true;
      else if (filter === 'fail') opts.success = false;
      if (versionFilter !== 'all') opts.agent_version = versionFilter;
      if (debouncedSearch) opts.q = debouncedSearch;
      const p = await listTraces(opts);
      setPage(p);
      // sync date if backend chose a default
      if (!date && p.date) setDate(p.date);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Backend ${e.status}: ${e.message}`
          : `Network error: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setLoading(false);
    }
  }, [date, filter, versionFilter, debouncedSearch, offset]);

  useEffect(() => {
    void load();
  }, [load]);

  // Reset offset when filters change.
  useEffect(() => {
    setOffset(0);
  }, [date, filter, versionFilter, debouncedSearch]);

  const versionOptions = useMemo(() => {
    if (!page) return ['all'];
    const versions = new Set<string>();
    for (const t of page.items) {
      if (t.agent_version) versions.add(t.agent_version);
    }
    return ['all', ...Array.from(versions).sort()];
  }, [page]);

  const showing = page?.items.length ?? 0;
  const total = page?.total_filtered ?? 0;
  const startIdx = total === 0 ? 0 : offset + 1;
  const endIdx = offset + showing;

  return (
    <div className="content">
      <h1 className="page-title">Traces</h1>
      <p className="page-sub">
        Every pipeline run, raw. Click a row to inspect stages, retrieved docs, and routing
        decisions.
      </p>

      {error && (
        <div className="card card-pad" style={{ borderColor: 'var(--bad)', marginBottom: 16 }}>
          <p className="dim" style={{ color: 'var(--bad)', margin: 0 }}>
            {error}
          </p>
        </div>
      )}

      <div className="trace-toolbar">
        <div className="filter-pills">
          {(['all', 'pass', 'fail'] as const).map((f) => (
            <button
              key={f}
              className={`pill ${filter === f ? 'on' : ''}`}
              onClick={() => setFilter(f)}
            >
              {f === 'all' ? 'All' : f === 'pass' ? 'Successful' : 'Failed'}
            </button>
          ))}
        </div>
        <div className="trace-toolbar-right" style={{ gap: 8 }}>
          {page && page.available_dates.length > 1 && (
            <select
              value={date || page.date}
              onChange={(e) => setDate(e.target.value)}
              style={{
                padding: '6px 10px',
                border: '1px solid var(--border)',
                borderRadius: 6,
                background: 'var(--bg)',
                fontSize: 12.5,
              }}
            >
              {page.available_dates.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          )}
          {versionOptions.length > 1 && (
            <select
              value={versionFilter}
              onChange={(e) => setVersionFilter(e.target.value)}
              style={{
                padding: '6px 10px',
                border: '1px solid var(--border)',
                borderRadius: 6,
                background: 'var(--bg)',
                fontSize: 12.5,
                fontFamily: 'var(--font-mono)',
              }}
            >
              {versionOptions.map((v) => (
                <option key={v} value={v}>
                  {v === 'all' ? 'all versions' : v}
                </option>
              ))}
            </select>
          )}
          <div className="search-input">
            <Icon name="search" size={13} />
            <input
              placeholder="Search request / response…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search && (
              <button className="clear" onClick={() => setSearch('')}>
                ×
              </button>
            )}
          </div>
          <button
            className="btn sm ghost"
            onClick={() => {
              setFilter('all');
              setVersionFilter('all');
              setSearch('');
            }}
            disabled={filter === 'all' && versionFilter === 'all' && !search}
          >
            Reset
          </button>
        </div>
      </div>

      {loading && !page && (
        <div className="dim" style={{ padding: 32, fontSize: 13 }}>
          Loading traces…
        </div>
      )}

      {page && (
        <>
          <div className="card">
            <div
              style={{
                padding: '12px 16px',
                borderBottom: '1px solid var(--border)',
                fontSize: 12,
                color: 'var(--fg-muted)',
                display: 'grid',
                gridTemplateColumns: '110px 1fr 1fr 90px 80px 80px',
                gap: 16,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                fontWeight: 500,
              }}
            >
              <div>Time</div>
              <div>Request</div>
              <div>Response</div>
              <div>Version</div>
              <div>Latency</div>
              <div>Status</div>
            </div>
            {page.items.length === 0 ? (
              <div className="empty-state" style={{ padding: 48 }}>
                <div style={{ fontSize: 13, color: 'var(--fg-muted)' }}>
                  No traces match the current filters for {page.date || 'this date'}.
                </div>
              </div>
            ) : (
              page.items.map((t: TraceSummary) => (
                <div
                  key={t.trace_id}
                  className="trace-row"
                  onClick={() => setOpenId(t.trace_id)}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '110px 1fr 1fr 90px 80px 80px',
                    gap: 16,
                    padding: '12px 16px',
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                    alignItems: 'center',
                    fontSize: 13,
                  }}
                >
                  <div>
                    <div className="mono" style={{ fontSize: 12 }}>
                      {fmtTimeOfDay(t.timestamp)}
                    </div>
                    <div className="dim" style={{ fontSize: 11 }}>
                      {fmtRelative(t.timestamp)}
                    </div>
                  </div>
                  <div
                    style={{
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={t.request}
                  >
                    {t.request || <span className="dim">(empty)</span>}
                  </div>
                  <div
                    className="dim"
                    style={{
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontSize: 12.5,
                    }}
                    title={t.response || ''}
                  >
                    {t.response || (t.error ? <span style={{ color: 'var(--bad-fg)' }}>{t.error}</span> : '—')}
                  </div>
                  <span className="mono dim" style={{ fontSize: 11.5 }}>
                    {t.agent_version || '—'}
                  </span>
                  <span className="mono" style={{ fontSize: 12 }}>
                    {fmtDuration(t.duration_ms)}
                  </span>
                  <Tag kind={t.success ? 'success' : 'bad'}>
                    <span className="dot" /> {t.success ? 'ok' : 'fail'}
                  </Tag>
                </div>
              ))
            )}
          </div>

          <div className="trace-foot" style={{ marginTop: 12 }}>
            <span className="dim">
              {total === 0
                ? '0 traces'
                : `Showing ${startIdx}–${endIdx} of ${total} traces`}
            </span>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
              <button
                className="btn sm ghost"
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0 || loading}
              >
                ← Previous
              </button>
              <button
                className="btn sm ghost"
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={!page.has_more || loading}
              >
                Next →
              </button>
            </div>
          </div>
        </>
      )}

      {openId && <TraceDrawer traceId={openId} onClose={() => setOpenId(null)} />}
    </div>
  );
};

const TraceDrawer = ({ traceId, onClose }: { traceId: string; onClose: () => void }) => {
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEscape(onClose);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getTrace(traceId)
      .then((t) => {
        if (!cancelled) setTrace(t);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(
            e instanceof ApiError
              ? `Backend ${e.status}: ${e.message}`
              : `Error: ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [traceId]);

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet trace-sheet">
        <div className="sheet-head">
          <div style={{ flex: 1 }}>
            <div
              className="dim"
              style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}
            >
              Trace · {trace?.agent_version || '—'}
            </div>
            <h2 className="mono" style={{ marginTop: 4, fontSize: 16 }}>
              {traceId}
            </h2>
            {trace && (
              <div className="dim" style={{ fontSize: 12.5, marginTop: 6 }}>
                {fmtTimeOfDay(trace.timestamp)} · {fmtDuration(trace.duration_ms)} ·{' '}
                {trace.success ? (
                  <span style={{ color: 'var(--accent-fg)' }}>success</span>
                ) : (
                  <span style={{ color: 'var(--bad-fg)' }}>failed</span>
                )}
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {loading && (
          <div className="sheet-body">
            <div className="dim" style={{ fontSize: 13, padding: 24 }}>Loading…</div>
          </div>
        )}
        {error && (
          <div className="sheet-body">
            <div style={{ color: 'var(--bad-fg)', fontSize: 13, padding: 24 }}>{error}</div>
          </div>
        )}
        {trace && !loading && (
          <div className="sheet-body">
            <div className="sheet-section">
              <h3>Request</h3>
              <div
                style={{
                  padding: '12px 14px',
                  background: 'var(--bg-muted)',
                  borderRadius: 8,
                  fontSize: 13.5,
                  lineHeight: 1.55,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {trace.request || <span className="dim">(empty request)</span>}
              </div>
            </div>

            <div className="sheet-section">
              <h3>Response</h3>
              {trace.response ? (
                <div
                  style={{
                    padding: '12px 14px',
                    background: 'var(--bg-muted)',
                    borderRadius: 8,
                    fontSize: 13.5,
                    lineHeight: 1.55,
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {trace.response}
                </div>
              ) : (
                <div
                  className="dim"
                  style={{
                    padding: '12px 14px',
                    background: 'var(--bg-muted)',
                    borderRadius: 8,
                    fontSize: 13,
                  }}
                >
                  No response —{' '}
                  {trace.error ? (
                    <span style={{ color: 'var(--bad-fg)' }}>{trace.error}</span>
                  ) : (
                    'pipeline did not produce output'
                  )}
                </div>
              )}
            </div>

            <div className="sheet-section">
              <h3>
                Pipeline stages{' '}
                <span className="dim mono" style={{ fontSize: 12, fontWeight: 400 }}>
                  · {trace.stages.length}
                </span>
              </h3>
              {trace.stages.length === 0 ? (
                <div className="dim" style={{ fontSize: 13 }}>No stages recorded.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {trace.stages.map((s, i) => {
                    const failed = !!s.error;
                    return (
                      <div
                        key={i}
                        className="card card-pad"
                        style={{
                          padding: '12px 14px',
                          borderColor: failed ? 'var(--bad)' : undefined,
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 10,
                            marginBottom: 6,
                          }}
                        >
                          <span
                            className="mono dim"
                            style={{ fontSize: 11, width: 18 }}
                          >
                            {i + 1}.
                          </span>
                          <span style={{ fontWeight: 500, fontSize: 13.5 }}>
                            {s.stage || s.technique}
                          </span>
                          <span
                            className="mono dim"
                            style={{ fontSize: 11.5, marginLeft: 6 }}
                          >
                            {s.technique}/{s.variant}
                          </span>
                          <span
                            className="mono"
                            style={{ marginLeft: 'auto', fontSize: 12 }}
                          >
                            {fmtDuration(s.duration_ms)}
                          </span>
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            gap: 12,
                            fontSize: 12,
                            color: 'var(--fg-muted)',
                            flexWrap: 'wrap',
                          }}
                        >
                          {(s.docs_in > 0 || s.docs_out > 0) && (
                            <span>
                              docs <span className="mono">{s.docs_in}</span> →{' '}
                              <span className="mono">{s.docs_out}</span>
                            </span>
                          )}
                          {s.routing_model && (
                            <span>
                              routing →{' '}
                              <span className="mono" style={{ color: 'var(--fg)' }}>
                                {s.routing_model}
                              </span>
                            </span>
                          )}
                          {s.response_set != null && (
                            <span>response_set {String(s.response_set)}</span>
                          )}
                        </div>
                        {failed && (
                          <div
                            style={{
                              marginTop: 8,
                              fontSize: 12.5,
                              color: 'var(--bad-fg)',
                              fontFamily: 'var(--font-mono)',
                            }}
                          >
                            {s.error}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {Object.keys(trace.metadata || {}).length > 0 && (
              <div className="sheet-section">
                <h3>Metadata</h3>
                <pre
                  style={{
                    background: 'var(--bg-muted)',
                    padding: '10px 12px',
                    borderRadius: 6,
                    fontSize: 12,
                    fontFamily: 'var(--font-mono)',
                    overflow: 'auto',
                    margin: 0,
                  }}
                >
                  {JSON.stringify(trace.metadata, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
};


// ============================================================================
// Eval suites
// ============================================================================

interface SuiteSample {
  id: string;
  input: string;
  expected: string;
  actual: string;
  score: number;
  pass: boolean;
}

interface Suite {
  id: string;
  name: string;
  desc: string;
  n: number;
  score: number;
  baseline: number | null;
  owner: 'agent' | 'human';
  updated: string;
  running: boolean;
  history: (number | null)[];
  samples: SuiteSample[];
}

const SUITES: Suite[] = [
  {
    id: 'tone', name: 'Tone — empathetic', desc: 'Checks whether the agent acknowledges feelings before solving the problem.',
    n: 412, score: 0.92, baseline: 0.88, owner: 'agent', updated: '2d', running: false,
    history: [0.74, 0.78, 0.81, 0.84, 0.86, 0.88, 0.89, 0.91, 0.92],
    samples: [
      { id: 's1', input: 'You charged me twice this is ridiculous', expected: 'Acknowledges frustration before solving.', actual: "That's frustrating, I'm sorry. Let me find both charges.", score: 0.96, pass: true },
      { id: 's2', input: 'I waited 40 minutes on hold last time', expected: 'Acknowledges wait without over-apologizing.', actual: "I hear you — that's way too long. Let me get this sorted now.", score: 0.94, pass: true },
      { id: 's3', input: 'Why is your shipping always late', expected: 'Acknowledges pattern, takes ownership.', actual: 'Fair point. Let me look up your last few orders.', score: 0.78, pass: true },
      { id: 's4', input: 'You ruined my birthday party', expected: 'Strong acknowledgment.', actual: "Oh no. That's awful, I'm really sorry. Tell me what happened.", score: 0.99, pass: true },
      { id: 's5', input: 'are you a robot', expected: 'Warm answer, not curt.', actual: 'Yes.', score: 0.18, pass: false },
    ],
  },
  {
    id: 'accuracy', name: 'Order lookup accuracy', desc: 'Tool returns the correct order for fuzzy or messy IDs.',
    n: 230, score: 0.96, baseline: 0.71, owner: 'agent', updated: '2d', running: false,
    history: [0.62, 0.66, 0.71, 0.78, 0.84, 0.89, 0.92, 0.95, 0.96],
    samples: [
      { id: 's1', input: 'order ord 3318 a', expected: 'ORD-3318-A', actual: 'ORD-3318-A', score: 1.0, pass: true },
      { id: 's2', input: 'order# 4901B', expected: 'ORD-4901-B', actual: 'ORD-4901-B', score: 1.0, pass: true },
      { id: 's3', input: 'my order from tuesday', expected: 'asks clarification', actual: 'I see two recent orders — was it the navy jacket or the boots?', score: 0.92, pass: true },
      { id: 's4', input: 'ORD2210C', expected: 'ORD-2210-C', actual: 'ORD-2210-C', score: 1.0, pass: true },
      { id: 's5', input: 'orderr 1109a', expected: 'ORD-1109-A', actual: 'ORD-1109-A', score: 1.0, pass: true },
    ],
  },
  {
    id: 'neutrality', name: 'Competitor neutrality', desc: 'Stays neutral when customers mention competitors.',
    n: 30, score: 0.91, baseline: null, owner: 'agent', updated: '11d', running: false,
    history: [null, null, null, null, null, 0.88, 0.90, 0.91, 0.91],
    samples: [
      { id: 's1', input: 'thinking of switching to Acme', expected: 'Neutral, offers comparison.', actual: 'I hear you — pricing matters. Want me to walk through usage?', score: 0.93, pass: true },
      { id: 's2', input: 'Acme has better support', expected: 'No disparagement.', actual: 'Their team is solid. Can I help you decide what fits?', score: 0.89, pass: true },
    ],
  },
  {
    id: 'csat', name: 'Predicted CSAT (LLM judge)', desc: 'A judge model predicts CSAT from the transcript.',
    n: 540, score: 0.86, baseline: 0.81, owner: 'human', updated: '5d', running: false,
    history: [0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.85, 0.86, 0.86],
    samples: [],
  },
  {
    id: 'safety', name: 'Refusal — out of scope', desc: 'Refuses or escalates when a request is outside policy.',
    n: 120, score: 0.99, baseline: 0.97, owner: 'human', updated: '14d', running: false,
    history: [0.95, 0.95, 0.96, 0.96, 0.97, 0.97, 0.98, 0.98, 0.99],
    samples: [],
  },
  {
    id: 'concise', name: 'Concise without losing warmth', desc: 'Stays under 60 words while still acknowledging the customer.',
    n: 200, score: 0.78, baseline: 0.82, owner: 'agent', updated: '8d', running: false,
    history: [0.84, 0.85, 0.83, 0.82, 0.80, 0.79, 0.78, 0.78, 0.78],
    samples: [],
  },
];

export const EvalSuites = () => {
  const [filter, setFilter] = useState<'all' | 'agent' | 'you' | 'regressed'>('all');
  const [search, setSearch] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);
  const [running, setRunning] = useState<Record<string, boolean>>({});
  const [toast, setToast] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [extraSuites, setExtraSuites] = useState<Suite[]>([]);
  useAutoToast(toast, setToast);

  const allSuites = useMemo(() => [...extraSuites, ...SUITES], [extraSuites]);

  const filtered = allSuites.filter((s) => {
    if (filter === 'agent' && s.owner !== 'agent') return false;
    if (filter === 'you' && s.owner !== 'human') return false;
    if (filter === 'regressed') {
      const d = s.baseline == null ? null : s.score - s.baseline;
      if (d == null || d >= 0) return false;
    }
    if (search.trim() && !s.name.toLowerCase().includes(search.trim().toLowerCase())) return false;
    return true;
  });

  const open = allSuites.find((s) => s.id === openId);

  const runSuite = (id: string) => {
    setRunning((r) => ({ ...r, [id]: true }));
    setToast(`Running ${allSuites.find((s) => s.id === id)?.name}…`);
    setTimeout(() => {
      setRunning((r) => ({ ...r, [id]: false }));
      setToast('Done — no regressions detected.');
    }, 1800);
  };

  const counts = {
    all: allSuites.length,
    agent: allSuites.filter((s) => s.owner === 'agent').length,
    you: allSuites.filter((s) => s.owner === 'human').length,
    regressed: allSuites.filter((s) => s.baseline != null && s.score - s.baseline < 0).length,
  };

  const handleCreate = (data: { name: string; desc: string }) => {
    const newSuite: Suite = {
      id: `custom_${Date.now()}`,
      name: data.name,
      desc: data.desc || 'Custom suite.',
      n: 0,
      score: 0,
      baseline: null,
      owner: 'human',
      updated: 'just now',
      running: false,
      history: [null, null, null, null, null, null, null, null, null],
      samples: [],
    };
    setExtraSuites((s) => [newSuite, ...s]);
    setCreating(false);
    setToast(`Created "${data.name}". Run it to get a first score.`);
  };

  const filterPills: [typeof filter, string, number][] = [
    ['all', 'All', counts.all],
    ['agent', 'Agent-authored', counts.agent],
    ['you', 'You authored', counts.you],
    ['regressed', 'Regressed', counts.regressed],
  ];

  return (
    <div className="content">
      <h1 className="page-title">Eval suites</h1>
      <p className="page-sub">The self-tests your agent runs against every change. Some it wrote itself; some you pinned.</p>

      <div className="trace-toolbar">
        <div className="filter-pills">
          {filterPills.map(([id, label, n]) => (
            <button key={id} className={`pill ${filter === id ? 'on' : ''}`} onClick={() => setFilter(id)}>
              {label} <span className="dim mono" style={{ fontSize: 11 }}>{n}</span>
            </button>
          ))}
        </div>
        <div className="trace-toolbar-right">
          <div className="search-input">
            <Icon name="search" size={13} />
            <input placeholder="Search suites…" value={search} onChange={(e) => setSearch(e.target.value)} />
            {search && <button className="clear" onClick={() => setSearch('')}>×</button>}
          </div>
          <button
            className="btn sm"
            onClick={() => {
              setToast('Running all suites in parallel…');
              setTimeout(() => setToast('All suites complete.'), 2000);
            }}
          >
            <Icon name="play" size={11} /> Run all
          </button>
        </div>
      </div>

      <div className="card">
        <div className="suite-head">
          <div>Suite</div>
          <div>Samples</div>
          <div>Score</div>
          <div>vs. baseline</div>
          <div>Author</div>
          <div></div>
        </div>
        {filtered.length === 0 ? (
          <div className="empty-state">
            <div style={{ fontSize: 13, color: 'var(--fg-muted)' }}>No suites match.</div>
            <button className="btn sm ghost" style={{ marginTop: 12 }} onClick={() => { setFilter('all'); setSearch(''); }}>
              Clear filters
            </button>
          </div>
        ) : (
          filtered.map((s) => {
            const delta = s.baseline == null ? null : (s.score - s.baseline) * 100;
            const better = delta == null ? null : delta >= 0;
            return (
              <div className="suite-row" key={s.id} onClick={() => setOpenId(s.id)}>
                <div>
                  <div style={{ fontWeight: 500, fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 8 }}>
                    {s.name}
                    {running[s.id] && <span className="run-dot" />}
                  </div>
                  <div style={{ marginTop: 6 }}>
                    <div className="bar" style={{ width: 200 }}>
                      <span style={{ width: `${s.score * 100}%` }} />
                    </div>
                  </div>
                </div>
                <span className="mono dim" style={{ fontSize: 12.5 }}>{s.n}</span>
                <span className="mono" style={{ fontWeight: 500 }}>{(s.score * 100).toFixed(0)}%</span>
                <span
                  className="mono"
                  style={{
                    fontSize: 12,
                    color: delta == null ? 'var(--fg-muted)' : better ? 'var(--accent-fg)' : 'var(--bad-fg)',
                  }}
                >
                  {delta == null ? 'new suite' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}pp`}
                </span>
                <span><Tag kind={s.owner === 'agent' ? 'info' : ''}>{s.owner === 'agent' ? 'Agent' : 'You'}</Tag></span>
                <button
                  className="row-action"
                  onClick={(e) => {
                    e.stopPropagation();
                    runSuite(s.id);
                  }}
                  title="Run now"
                >
                  <Icon name="play" size={11} />
                </button>
              </div>
            );
          })
        )}
      </div>

      <div className="trace-foot">
        <span className="dim">Showing {filtered.length} of {allSuites.length}</span>
        <span className="dim" style={{ marginLeft: 'auto' }}>Last full run: 2h ago</span>
      </div>

      <button className="add-btn" style={{ marginTop: 16, maxWidth: 320 }} onClick={() => setCreating(true)}>
        + New eval suite
      </button>

      {creating && <NewSuiteModal onCreate={handleCreate} onClose={() => setCreating(false)} />}

      {open && (
        <SuiteDrawer suite={open} running={!!running[open.id]} onClose={() => setOpenId(null)} onRun={() => runSuite(open.id)} />
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
};

const SuiteDrawer = ({
  suite,
  running,
  onClose,
  onRun,
}: {
  suite: Suite;
  running: boolean;
  onClose: () => void;
  onRun: () => void;
}) => {
  const [tab, setTab] = useState<'samples' | 'history' | 'settings'>('samples');
  useEscape(onClose);

  const delta = suite.baseline == null ? null : (suite.score - suite.baseline) * 100;
  const better = delta == null ? null : delta >= 0;
  const passing = suite.samples.filter((s) => s.pass).length;
  const failing = suite.samples.filter((s) => !s.pass).length;

  const validHist = suite.history
    .map((v, i) => ({ v, i }))
    .filter((p): p is { v: number; i: number } => p.v != null);
  const sparkW = 280;
  const sparkH = 56;
  const pad = 4;
  const ys = validHist.map((p) => p.v);
  const minY = Math.min(...ys, 0.6);
  const maxY = Math.max(...ys, 1);
  const points = validHist.map((p) => {
    const x = pad + (p.i / (suite.history.length - 1)) * (sparkW - pad * 2);
    const y = pad + (1 - (p.v - minY) / (maxY - minY)) * (sparkH - pad * 2);
    return [x, y] as const;
  });
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet trace-sheet">
        <div className="sheet-head">
          <div style={{ flex: 1 }}>
            <div className="dim" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {suite.owner === 'agent' ? 'Agent-authored' : 'You authored'} · updated {suite.updated} ago
            </div>
            <h2 style={{ marginTop: 4 }}>{suite.name}</h2>
            <div style={{ fontSize: 13, color: 'var(--fg-muted)', marginTop: 6, lineHeight: 1.5 }}>{suite.desc}</div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="suite-stats">
          <div>
            <div className="dim stat-label">Score</div>
            <div className="stat-val">
              {(suite.score * 100).toFixed(0)}
              <span className="dim mono" style={{ fontSize: 14, fontWeight: 400 }}>%</span>
            </div>
          </div>
          <div>
            <div className="dim stat-label">vs. baseline</div>
            <div
              className="stat-val mono"
              style={{ color: delta == null ? 'var(--fg-muted)' : better ? 'var(--accent-fg)' : 'var(--bad-fg)' }}
            >
              {delta == null ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}pp`}
            </div>
          </div>
          <div>
            <div className="dim stat-label">Samples</div>
            <div className="stat-val mono">{suite.n}</div>
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="dim stat-label">Trend</div>
            <svg className="sparkline" width={sparkW} height={sparkH}>
              <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
              {points.map((p, i) => (
                <circle key={i} cx={p[0]} cy={p[1]} r="2" fill="var(--accent)" />
              ))}
            </svg>
          </div>
        </div>

        <div className="sheet-tabs">
          <button className={`tab ${tab === 'samples' ? 'active' : ''}`} onClick={() => setTab('samples')}>
            Samples <span className="dim mono" style={{ fontSize: 11 }}>{suite.samples.length}</span>
          </button>
          <button className={`tab ${tab === 'history' ? 'active' : ''}`} onClick={() => setTab('history')}>History</button>
          <button className={`tab ${tab === 'settings' ? 'active' : ''}`} onClick={() => setTab('settings')}>Settings</button>
        </div>

        <div className="sheet-body">
          {tab === 'samples' &&
            (suite.samples.length === 0 ? (
              <div className="dim" style={{ fontSize: 13, padding: '20px 0' }}>
                This suite uses an LLM judge instead of fixture samples.
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', gap: 12, marginBottom: 14, fontSize: 12, color: 'var(--fg-muted)' }}>
                  <span><b style={{ color: 'var(--accent-fg)' }}>{passing}</b> passing</span>
                  {failing > 0 && <span><b style={{ color: 'var(--bad-fg)' }}>{failing}</b> failing</span>}
                </div>
                <div className="samples">
                  {suite.samples.map((s) => (
                    <div key={s.id} className={`sample ${s.pass ? 'pass' : 'fail'}`}>
                      <div className="sample-head">
                        <span className="sample-input">"{s.input}"</span>
                        <span
                          className="mono"
                          style={{
                            fontSize: 11.5,
                            fontWeight: 500,
                            color: s.pass ? 'var(--accent-fg)' : 'var(--bad-fg)',
                          }}
                        >
                          {(s.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="sample-row"><span className="sample-label">Expected</span><span>{s.expected}</span></div>
                      <div className="sample-row"><span className="sample-label">Actual</span><span>{s.actual}</span></div>
                    </div>
                  ))}
                </div>
              </>
            ))}
          {tab === 'history' && (
            <div>
              <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>Last 9 runs.</div>
              {suite.history.map((v, i) =>
                v == null ? null : (
                  <div key={i} className="history-row">
                    <span className="dim mono" style={{ fontSize: 11.5, width: 40 }}>run {i + 1}</span>
                    <div className="bar" style={{ flex: 1 }}>
                      <span style={{ width: `${v * 100}%` }} />
                    </div>
                    <span className="mono" style={{ fontSize: 12.5, width: 44, textAlign: 'right' }}>
                      {(v * 100).toFixed(0)}%
                    </span>
                  </div>
                )
              )}
            </div>
          )}
          {tab === 'settings' && (
            <div className="meta-grid">
              <div className="meta-row"><div className="dim">Suite ID</div><div className="mono">{suite.id}</div></div>
              <div className="meta-row"><div className="dim">Author</div><div>{suite.owner === 'agent' ? 'Agent (self-authored)' : 'You'}</div></div>
              <div className="meta-row"><div className="dim">Sample size</div><div className="mono">{suite.n}</div></div>
              <div className="meta-row"><div className="dim">Run cadence</div><div>On every proposed change</div></div>
              <div className="meta-row"><div className="dim">Threshold</div><div className="mono">≥ {suite.baseline ? (suite.baseline * 100).toFixed(0) : '80'}%</div></div>
              <div className="meta-row"><div className="dim">Updated</div><div>{suite.updated} ago</div></div>
            </div>
          )}
        </div>

        <div className="sheet-foot">
          <button className="btn sm" onClick={onRun} disabled={running}>
            <Icon name="play" size={11} /> {running ? 'Running…' : 'Run now'}
          </button>
          <button className="btn sm ghost"><Icon name="copy" size={12} /> Duplicate</button>
          <button className="btn sm ghost" style={{ marginLeft: 'auto', color: 'var(--bad-fg)' }}>Archive</button>
        </div>
      </div>
    </>
  );
};

const NewSuiteModal = ({
  onCreate,
  onClose,
}: {
  onCreate: (d: { name: string; desc: string; template: string }) => void;
  onClose: () => void;
}) => {
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [template, setTemplate] = useState('blank');
  useEscape(onClose);

  const templates = [
    { id: 'blank', name: 'Blank', desc: 'Start from scratch.' },
    { id: 'tone', name: 'Tone check', desc: 'Verify how the agent sounds.' },
    { id: 'accuracy', name: 'Accuracy', desc: 'Check tool / answer correctness.' },
    { id: 'judge', name: 'LLM judge', desc: 'Score live transcripts with a judge model.' },
  ];

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    onCreate({ name: name.trim(), desc: desc.trim(), template });
  };

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="modal">
        <form onSubmit={submit}>
          <div className="modal-head">
            <h2>New eval suite</h2>
            <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">×</button>
          </div>
          <div className="modal-body">
            <div className="field">
              <label>Name</label>
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Refund tone" />
            </div>
            <div className="field">
              <label>Description <span className="dim">(optional)</span></label>
              <textarea value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="What does this suite check for?" rows={2} />
            </div>
            <div className="field">
              <label>Template</label>
              <div className="template-grid">
                {templates.map((t) => (
                  <button key={t.id} type="button" className={`template-card ${template === t.id ? 'on' : ''}`} onClick={() => setTemplate(t.id)}>
                    <div className="template-name">{t.name}</div>
                    <div className="template-desc">{t.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn sm ghost" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn sm" disabled={!name.trim()}>Create suite</button>
          </div>
        </form>
      </div>
    </>
  );
};

// ============================================================================
// Router config
// ============================================================================

interface RouterRule {
  id: string;
  name: string;
  when: string;
  then: string;
  share: number;
  cost: number;
  auth: 'agent' | 'human';
  enabled: boolean;
  isDefault?: boolean;
  rationale: string;
  history: { when: string; what: string }[];
  samples: string[];
}

const ROUTER_RULES: RouterRule[] = [
  {
    id: 'r1', name: 'Fact lookups → Haiku', when: 'intent == "fact_lookup" and turns == 1', then: 'haiku-4.5',
    share: 0.32, cost: 0.001, auth: 'agent', enabled: true,
    rationale: "One-turn factual questions don't need Sonnet. Agent saw 92% pass rate on Haiku in shadow mode.",
    history: [
      { when: '2d ago', what: 'Agent created this rule. Auto-approved (low risk).' },
      { when: '5d ago', what: 'Rule was a Sonnet rule. Agent proposed downgrade.' },
    ],
    samples: ['"what time do you close?"', '"where are you located?"', '"do you ship internationally?"'],
  },
  {
    id: 'r2', name: 'Frustrated + high value → Sonnet', when: 'sentiment.frustrated and order_total > 100', then: 'sonnet-4.5',
    share: 0.08, cost: 0.018, auth: 'human', enabled: true,
    rationale: 'High-stakes recovery. Worth the cost premium.',
    history: [{ when: '14d ago', what: 'You created this rule.' }],
    samples: ['"you charged me TWICE this is ridiculous" (order: $240)', '"i\'ve been waiting 3 weeks" (order: $189)'],
  },
  {
    id: 'r3', name: 'Non-English → Sonnet', when: 'language != "en"', then: 'sonnet-4.5',
    share: 0.14, cost: 0.014, auth: 'human', enabled: true,
    rationale: 'Haiku translations were less reliable on the eval set.',
    history: [{ when: '8d ago', what: 'You created this rule.' }],
    samples: ['"meu pedido não chegou" (pt-BR)', '"¿cuándo llega mi pedido?" (es)'],
  },
  {
    id: 'default', name: 'Default → Sonnet', when: 'default', then: 'sonnet-4.5',
    share: 0.46, cost: 0.012, auth: 'human', enabled: true, isDefault: true,
    rationale: 'Catch-all. Anything not matched by a specific rule.',
    history: [{ when: 'Day 0', what: 'Created with the agent.' }],
    samples: ['Anything not matching above rules.'],
  },
];

const MODELS = ['haiku-4.5', 'sonnet-4.5', 'opus-4.5'];

export const RouterConfig = () => {
  const [rules, setRules] = useState<RouterRule[]>(ROUTER_RULES);
  const [filter, setFilter] = useState<'all' | 'agent' | 'you' | 'haiku' | 'sonnet'>('all');
  const [search, setSearch] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  useAutoToast(toast, setToast);

  const filtered = rules.filter((r) => {
    if (filter === 'agent' && r.auth !== 'agent') return false;
    if (filter === 'you' && r.auth !== 'human') return false;
    if (filter === 'haiku' && r.then !== 'haiku-4.5') return false;
    if (filter === 'sonnet' && r.then !== 'sonnet-4.5') return false;
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      if (!r.name.toLowerCase().includes(q) && !r.when.toLowerCase().includes(q)) return false;
    }
    return true;
  });

  const open = rules.find((r) => r.id === openId);

  const totalCost = rules.reduce((s, r) => s + (r.enabled ? r.share * r.cost : 0), 0);
  const totalShare = rules.filter((r) => r.enabled).reduce((s, r) => s + r.share, 0);

  const toggleEnabled = (id: string) => {
    setRules((rs) => rs.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r)));
    const r = rules.find((rr) => rr.id === id);
    if (r) setToast(`${r.name} ${r.enabled ? 'paused' : 'enabled'}.`);
  };

  const moveRule = (id: string, dir: -1 | 1) => {
    setRules((rs) => {
      const idx = rs.findIndex((r) => r.id === id);
      const target = idx + dir;
      if (target < 0 || target >= rs.length || rs[target].isDefault || rs[idx].isDefault) return rs;
      const next = [...rs];
      [next[idx], next[target]] = [next[target], next[idx]];
      return next;
    });
    setToast('Rule order updated.');
  };

  const deleteRule = (id: string) => {
    const r = rules.find((rr) => rr.id === id);
    if (!r || r.isDefault) return;
    setRules((rs) => rs.filter((x) => x.id !== id));
    setOpenId(null);
    setToast(`Deleted "${r.name}".`);
  };

  const handleCreate = (data: { name: string; when: string; then: string; rationale: string }) => {
    const newRule: RouterRule = {
      id: `rule_${Date.now()}`,
      name: data.name,
      when: data.when,
      then: data.then,
      share: 0,
      cost: data.then === 'haiku-4.5' ? 0.001 : data.then === 'sonnet-4.5' ? 0.012 : 0.06,
      auth: 'human',
      enabled: true,
      rationale: data.rationale || 'Custom rule.',
      history: [{ when: 'just now', what: 'You created this rule.' }],
      samples: [],
    };
    setRules((rs) => {
      const idx = rs.findIndex((r) => r.isDefault);
      const next = [...rs];
      next.splice(idx, 0, newRule);
      return next;
    });
    setCreating(false);
    setToast(`Created "${data.name}".`);
  };

  const counts = {
    all: rules.length,
    agent: rules.filter((r) => r.auth === 'agent').length,
    you: rules.filter((r) => r.auth === 'human').length,
    haiku: rules.filter((r) => r.then === 'haiku-4.5').length,
    sonnet: rules.filter((r) => r.then === 'sonnet-4.5').length,
  };

  const filterPills: [typeof filter, string, number][] = [
    ['all', 'All', counts.all],
    ['agent', 'Agent-authored', counts.agent],
    ['you', 'You authored', counts.you],
    ['haiku', '→ Haiku', counts.haiku],
    ['sonnet', '→ Sonnet', counts.sonnet],
  ];

  return (
    <div className="content">
      <h1 className="page-title">Router config</h1>
      <p className="page-sub">
        Rules that pick a model per request, evaluated top-to-bottom. The agent can propose new rules — they show up in
        Review.
      </p>

      <div className="router-summary">
        <div>
          <div className="dim stat-label">Active rules</div>
          <div className="stat-val mono">{rules.filter((r) => r.enabled).length}</div>
        </div>
        <div>
          <div className="dim stat-label">Avg cost / conv</div>
          <div className="stat-val mono">${totalCost.toFixed(3)}</div>
        </div>
        <div>
          <div className="dim stat-label">Routed share</div>
          <div className="stat-val mono">{(totalShare * 100).toFixed(0)}%</div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="dim stat-label">Distribution</div>
          <div className="dist-bar">
            {rules
              .filter((r) => r.enabled)
              .map((r) => (
                <span
                  key={r.id}
                  className={`seg seg-${r.then.split('-')[0]}`}
                  style={{ flex: r.share }}
                  title={`${r.name}: ${(r.share * 100).toFixed(0)}%`}
                />
              ))}
          </div>
        </div>
      </div>

      <div className="trace-toolbar">
        <div className="filter-pills">
          {filterPills.map(([id, label, n]) => (
            <button key={id} className={`pill ${filter === id ? 'on' : ''}`} onClick={() => setFilter(id)}>
              {label} <span className="dim mono" style={{ fontSize: 11 }}>{n}</span>
            </button>
          ))}
        </div>
        <div className="trace-toolbar-right">
          <div className="search-input">
            <Icon name="search" size={13} />
            <input placeholder="Search rules…" value={search} onChange={(e) => setSearch(e.target.value)} />
            {search && <button className="clear" onClick={() => setSearch('')}>×</button>}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="rule-head">
          <div></div><div>Rule</div><div>Route to</div><div>Share</div><div>Cost</div><div></div><div></div>
        </div>
        {filtered.length === 0 ? (
          <div className="empty-state">
            <div style={{ fontSize: 13, color: 'var(--fg-muted)' }}>No rules match.</div>
            <button className="btn sm ghost" style={{ marginTop: 12 }} onClick={() => { setFilter('all'); setSearch(''); }}>
              Clear filters
            </button>
          </div>
        ) : (
          filtered.map((r, i) => {
            const isFirst = i === 0;
            const isLast = i === filtered.length - 1 || filtered[i + 1]?.isDefault;
            return (
              <div
                className={`rule-row ${!r.enabled ? 'disabled' : ''} ${r.isDefault ? 'default' : ''}`}
                key={r.id}
                onClick={() => setOpenId(r.id)}
              >
                <div className="rule-priority">
                  <span className="mono dim" style={{ fontSize: 11 }}>{r.isDefault ? '–' : i + 1}</span>
                  {!r.isDefault && (
                    <div className="reorder">
                      <button
                        className="row-action xs"
                        onClick={(e) => { e.stopPropagation(); moveRule(r.id, -1); }}
                        disabled={isFirst}
                        title="Move up"
                      >
                        <Icon name="arrowUp" size={10} />
                      </button>
                      <button
                        className="row-action xs"
                        onClick={(e) => { e.stopPropagation(); moveRule(r.id, 1); }}
                        disabled={isLast}
                        title="Move down"
                      >
                        <Icon name="arrowDown" size={10} />
                      </button>
                    </div>
                  )}
                </div>
                <div>
                  <div style={{ fontWeight: 500, fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 8 }}>
                    {r.name}
                    {r.auth === 'agent' && <Tag kind="info">Agent</Tag>}
                  </div>
                  <div className="mono dim" style={{ fontSize: 11.5, marginTop: 4 }}>if {r.when}</div>
                </div>
                <div className="mono" style={{ fontSize: 12.5, fontWeight: 500 }}>{r.then}</div>
                <div className="mono" style={{ fontSize: 12.5 }}>{(r.share * 100).toFixed(0)}%</div>
                <div className="mono dim" style={{ fontSize: 12 }}>${r.cost.toFixed(3)}</div>
                <button
                  className={`toggle-pill ${r.enabled ? 'on' : ''} ${r.isDefault ? 'locked' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!r.isDefault) toggleEnabled(r.id);
                  }}
                  disabled={r.isDefault}
                  title={r.isDefault ? 'Default rule cannot be disabled' : r.enabled ? 'Pause' : 'Enable'}
                >
                  <span className="thumb" />
                </button>
                <Icon name="chevron" size={12} />
              </div>
            );
          })
        )}
      </div>

      <button className="add-btn" style={{ marginTop: 16, maxWidth: 320 }} onClick={() => setCreating(true)}>
        + Add routing rule
      </button>

      {open && (
        <RuleDrawer
          rule={open}
          onClose={() => setOpenId(null)}
          onDelete={() => deleteRule(open.id)}
          onToggle={() => toggleEnabled(open.id)}
        />
      )}
      {creating && <NewRuleModal onCreate={handleCreate} onClose={() => setCreating(false)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
};

const RuleDrawer = ({
  rule,
  onClose,
  onDelete,
  onToggle,
}: {
  rule: RouterRule;
  onClose: () => void;
  onDelete: () => void;
  onToggle: () => void;
}) => {
  const [tab, setTab] = useState<'overview' | 'samples' | 'history'>('overview');
  useEscape(onClose);

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet trace-sheet">
        <div className="sheet-head">
          <div style={{ flex: 1 }}>
            <div className="dim" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {rule.auth === 'agent' ? 'Agent-authored' : 'You authored'} · {rule.enabled ? 'active' : 'paused'}
            </div>
            <h2 style={{ marginTop: 4 }}>{rule.name}</h2>
            <div className="mono" style={{ fontSize: 12.5, color: 'var(--fg-muted)', marginTop: 6 }}>
              if {rule.when}
              <span style={{ margin: '0 8px', opacity: 0.5 }}>→</span>
              <span style={{ color: 'var(--fg)', fontWeight: 500 }}>{rule.then}</span>
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="sheet-tabs">
          <button className={`tab ${tab === 'overview' ? 'active' : ''}`} onClick={() => setTab('overview')}>Overview</button>
          <button className={`tab ${tab === 'samples' ? 'active' : ''}`} onClick={() => setTab('samples')}>Sample matches</button>
          <button className={`tab ${tab === 'history' ? 'active' : ''}`} onClick={() => setTab('history')}>History</button>
        </div>

        <div className="sheet-body">
          {tab === 'overview' && (
            <div>
              <div className="sheet-section">
                <h3>Rationale</h3>
                <p style={{ fontSize: 13, lineHeight: 1.55, color: 'var(--fg-muted)', margin: 0 }}>{rule.rationale}</p>
              </div>
              <div className="meta-grid">
                <div className="meta-row"><div className="dim">Share of traffic</div><div className="mono">{(rule.share * 100).toFixed(0)}%</div></div>
                <div className="meta-row"><div className="dim">Avg cost</div><div className="mono">${rule.cost.toFixed(3)}/conv</div></div>
                <div className="meta-row"><div className="dim">Status</div><div>{rule.enabled ? 'Active' : 'Paused'}</div></div>
                <div className="meta-row"><div className="dim">Author</div><div>{rule.auth === 'agent' ? 'Agent' : 'You'}</div></div>
              </div>
            </div>
          )}
          {tab === 'samples' &&
            (rule.samples.length === 0 ? (
              <div className="dim" style={{ fontSize: 13, padding: '20px 0' }}>No matched samples yet.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {rule.samples.map((s, i) => (
                  <div key={i} className="sample">
                    <div className="sample-input">{s}</div>
                  </div>
                ))}
              </div>
            ))}
          {tab === 'history' && (
            <div>
              {rule.history.map((h, i) => (
                <div
                  key={i}
                  style={{
                    padding: '10px 0',
                    borderBottom: i < rule.history.length - 1 ? '1px solid var(--border)' : 'none',
                    display: 'grid',
                    gridTemplateColumns: '90px 1fr',
                    gap: 12,
                    fontSize: 13,
                  }}
                >
                  <span className="dim" style={{ fontSize: 12 }}>{h.when}</span>
                  <span>{h.what}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="sheet-foot">
          {!rule.isDefault && (
            <button className={`btn sm ${rule.enabled ? 'ghost' : ''}`} onClick={onToggle}>
              {rule.enabled ? 'Pause rule' : 'Enable rule'}
            </button>
          )}
          <button className="btn sm ghost"><Icon name="copy" size={12} /> Duplicate</button>
          {!rule.isDefault && (
            <button className="btn sm ghost" style={{ marginLeft: 'auto', color: 'var(--bad-fg)' }} onClick={onDelete}>
              Delete
            </button>
          )}
        </div>
      </div>
    </>
  );
};

const NewRuleModal = ({
  onCreate,
  onClose,
}: {
  onCreate: (d: { name: string; when: string; then: string; rationale: string }) => void;
  onClose: () => void;
}) => {
  const [name, setName] = useState('');
  const [whenExpr, setWhenExpr] = useState('');
  const [model, setModel] = useState('haiku-4.5');
  const [rationale, setRationale] = useState('');
  useEscape(onClose);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !whenExpr.trim()) return;
    onCreate({ name: name.trim(), when: whenExpr.trim(), then: model, rationale: rationale.trim() });
  };

  const desc = (m: string) =>
    m === 'haiku-4.5'
      ? 'Fast, cheap. ~$0.001/conv.'
      : m === 'sonnet-4.5'
      ? 'Balanced. ~$0.012/conv.'
      : 'Most capable. ~$0.06/conv.';

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="modal" style={{ width: 520 }}>
        <form onSubmit={submit}>
          <div className="modal-head">
            <h2>New routing rule</h2>
            <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">×</button>
          </div>
          <div className="modal-body">
            <div className="field">
              <label>Name</label>
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Long conversations → Sonnet" />
            </div>
            <div className="field">
              <label>If <span className="dim">(condition)</span></label>
              <input
                value={whenExpr}
                onChange={(e) => setWhenExpr(e.target.value)}
                placeholder="e.g. turns > 5"
                className="mono"
                style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5 }}
              />
              <div className="dim" style={{ fontSize: 11.5, marginTop: 6 }}>
                Variables: <code>turns</code>, <code>language</code>, <code>sentiment</code>, <code>order_total</code>, <code>intent</code>
              </div>
            </div>
            <div className="field">
              <label>Route to</label>
              <div className="model-grid">
                {MODELS.map((m) => (
                  <button key={m} type="button" className={`template-card ${model === m ? 'on' : ''}`} onClick={() => setModel(m)}>
                    <div className="template-name mono">{m}</div>
                    <div className="template-desc">{desc(m)}</div>
                  </button>
                ))}
              </div>
            </div>
            <div className="field">
              <label>Rationale <span className="dim">(optional)</span></label>
              <textarea value={rationale} onChange={(e) => setRationale(e.target.value)} placeholder="Why this rule?" rows={2} />
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn sm ghost" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn sm" disabled={!name.trim() || !whenExpr.trim()}>Create rule</button>
          </div>
        </form>
      </div>
    </>
  );
};

// ============================================================================
// Datasets
// ============================================================================

interface DatasetSample {
  id: string;
  preview: string;
  tag: string;
}

interface Dataset {
  id: string;
  name: string;
  desc: string;
  size: number;
  source: string;
  sourceType: 'auto' | 'manual';
  fresh: string;
  use: string[];
  owner: 'agent' | 'human';
  growing: boolean;
  samples: DatasetSample[];
  history: { when: string; what: string }[];
}

const DATASETS_INITIAL: Dataset[] = [
  {
    id: 'd1', name: 'Refund edge cases', desc: 'Hard refund situations the agent should handle well.',
    size: 47, source: 'flagged traces', sourceType: 'auto', fresh: '3d', use: ['Eval', 'Distill'], owner: 'agent', growing: true,
    samples: [
      { id: 's1', preview: '"i want a refund on order from 6 months ago"', tag: 'out of policy' },
      { id: 's2', preview: '"this jacket\'s zipper broke after 2 weeks"', tag: 'warranty' },
      { id: 's3', preview: '"can i return half the order"', tag: 'partial' },
      { id: 's4', preview: '"i\'m disputing this charge with my bank"', tag: 'chargeback' },
      { id: 's5', preview: '"refund without me sending it back?"', tag: 'no return' },
    ],
    history: [
      { when: '3d ago', what: 'Agent added 4 new samples from flagged traces.' },
      { when: '14d ago', what: 'You created this dataset.' },
    ],
  },
  {
    id: 'd2', name: 'Frustrated customers', desc: 'Conversations where sentiment dropped below neutral.',
    size: 128, source: 'feedback signals', sourceType: 'auto', fresh: '1d', use: ['Eval'], owner: 'agent', growing: true,
    samples: [
      { id: 's1', preview: '"you charged me TWICE this is ridiculous"', tag: 'billing' },
      { id: 's2', preview: '"i\'ve been waiting 3 weeks for an answer"', tag: 'response time' },
      { id: 's3', preview: '"this is the third time i\'m asking"', tag: 'repeat' },
    ],
    history: [
      { when: '1d ago', what: 'Agent added 12 new samples.' },
      { when: '21d ago', what: 'Agent created this dataset.' },
    ],
  },
  {
    id: 'd3', name: 'Order ID variants', desc: 'All the ways customers write order IDs.',
    size: 230, source: 'failed lookups', sourceType: 'auto', fresh: '2d', use: ['Eval', 'Distill'], owner: 'agent', growing: true,
    samples: [
      { id: 's1', preview: '"order ord 3318 a"', tag: 'spaces' },
      { id: 's2', preview: '"ORD3318A"', tag: 'no dashes' },
      { id: 's3', preview: '"my number is 3318"', tag: 'partial' },
    ],
    history: [{ when: '2d ago', what: 'Agent added 8 samples.' }],
  },
  {
    id: 'd4', name: 'Multilingual (PT, ES)', desc: 'Non-English conversations for distillation.',
    size: 312, source: 'language router', sourceType: 'auto', fresh: '6h', use: ['Distill'], owner: 'human', growing: true,
    samples: [
      { id: 's1', preview: '"meu pedido não chegou"', tag: 'pt-BR' },
      { id: 's2', preview: '"¿cuándo llega mi pedido?"', tag: 'es' },
    ],
    history: [{ when: '6h ago', what: 'Auto-collection added 18 samples.' }],
  },
  {
    id: 'd5', name: 'Competitor mentions', desc: 'Conversations referencing competitor products.',
    size: 30, source: 'agent self-curated', sourceType: 'manual', fresh: '11d', use: ['Eval'], owner: 'agent', growing: false,
    samples: [
      { id: 's1', preview: '"thinking of switching to Acme"', tag: 'churn risk' },
      { id: 's2', preview: '"Acme has better support"', tag: 'comparison' },
    ],
    history: [{ when: '11d ago', what: 'Agent self-curated this dataset.' }],
  },
];

export const Datasets = () => {
  const [datasets, setDatasets] = useState<Dataset[]>(DATASETS_INITIAL);
  const [filter, setFilter] = useState<'all' | 'eval' | 'distill' | 'agent' | 'you'>('all');
  const [search, setSearch] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  useAutoToast(toast, setToast);

  const filtered = datasets.filter((d) => {
    if (filter === 'eval' && !d.use.includes('Eval')) return false;
    if (filter === 'distill' && !d.use.includes('Distill')) return false;
    if (filter === 'agent' && d.owner !== 'agent') return false;
    if (filter === 'you' && d.owner !== 'human') return false;
    if (search.trim() && !d.name.toLowerCase().includes(search.trim().toLowerCase())) return false;
    return true;
  });

  const open = datasets.find((d) => d.id === openId);

  const handleCreate = (data: { name: string; desc: string; source: string; sourceType: 'auto' | 'manual'; use: string[] }) => {
    const newSet: Dataset = {
      id: `ds_${Date.now()}`,
      name: data.name,
      desc: data.desc || 'Custom dataset.',
      size: 0,
      source: data.source,
      sourceType: data.sourceType,
      fresh: 'just now',
      use: data.use,
      owner: 'human',
      growing: data.sourceType === 'auto',
      samples: [],
      history: [{ when: 'just now', what: 'You created this dataset.' }],
    };
    setDatasets((ds) => [newSet, ...ds]);
    setCreating(false);
    setToast(`Created "${data.name}".`);
  };

  const deleteDataset = (id: string) => {
    const d = datasets.find((dd) => dd.id === id);
    if (!d) return;
    setDatasets((ds) => ds.filter((dd) => dd.id !== id));
    setOpenId(null);
    setToast(`Deleted "${d.name}".`);
  };

  const counts = {
    all: datasets.length,
    eval: datasets.filter((d) => d.use.includes('Eval')).length,
    distill: datasets.filter((d) => d.use.includes('Distill')).length,
    agent: datasets.filter((d) => d.owner === 'agent').length,
    you: datasets.filter((d) => d.owner === 'human').length,
  };

  const totalSamples = datasets.reduce((s, d) => s + d.size, 0);

  const filterPills: [typeof filter, string, number][] = [
    ['all', 'All', counts.all],
    ['eval', 'For evals', counts.eval],
    ['distill', 'For distill', counts.distill],
    ['agent', 'Agent-curated', counts.agent],
    ['you', 'You created', counts.you],
  ];

  return (
    <div className="content">
      <h1 className="page-title">Datasets</h1>
      <p className="page-sub">Curated sets of conversations. The agent uses them to test itself, distill smaller models, or pin examples.</p>

      <div className="trace-toolbar">
        <div className="filter-pills">
          {filterPills.map(([id, label, n]) => (
            <button key={id} className={`pill ${filter === id ? 'on' : ''}`} onClick={() => setFilter(id)}>
              {label} <span className="dim mono" style={{ fontSize: 11 }}>{n}</span>
            </button>
          ))}
        </div>
        <div className="trace-toolbar-right">
          <div className="search-input">
            <Icon name="search" size={13} />
            <input placeholder="Search datasets…" value={search} onChange={(e) => setSearch(e.target.value)} />
            {search && <button className="clear" onClick={() => setSearch('')}>×</button>}
          </div>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div style={{ fontSize: 13, color: 'var(--fg-muted)' }}>No datasets match.</div>
            <button className="btn sm ghost" style={{ marginTop: 12 }} onClick={() => { setFilter('all'); setSearch(''); }}>
              Clear filters
            </button>
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {filtered.map((d) => (
            <div className="dataset-card" key={d.id} onClick={() => setOpenId(d.id)}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <div className="dataset-icon"><Icon name="book" size={16} /></div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                    <div style={{ fontWeight: 500, fontSize: 14 }}>{d.name}</div>
                    {d.growing && <span className="growing-dot" title="Auto-collecting" />}
                  </div>
                  <div className="dim" style={{ fontSize: 12.5 }}>From {d.source} · refreshed {d.fresh} ago</div>
                </div>
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginTop: 14,
                  paddingTop: 14,
                  borderTop: '1px solid var(--border)',
                }}
              >
                <span className="mono" style={{ fontSize: 13, fontWeight: 500 }}>
                  {d.size}
                  <span className="dim" style={{ fontWeight: 400, marginLeft: 4 }}>samples</span>
                </span>
                <div style={{ display: 'flex', gap: 4 }}>
                  {d.use.map((u) => <Tag key={u}>{u}</Tag>)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="trace-foot">
        <span className="dim">
          Showing {filtered.length} of {datasets.length} · {totalSamples.toLocaleString()} total samples
        </span>
      </div>

      <button className="add-btn" style={{ marginTop: 16, maxWidth: 320 }} onClick={() => setCreating(true)}>
        + New dataset
      </button>

      {open && (
        <DatasetDrawer
          dataset={open}
          onClose={() => setOpenId(null)}
          onDelete={() => deleteDataset(open.id)}
          onToast={setToast}
        />
      )}
      {creating && <NewDatasetModal onCreate={handleCreate} onClose={() => setCreating(false)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
};

const DatasetDrawer = ({
  dataset,
  onClose,
  onDelete,
  onToast,
}: {
  dataset: Dataset;
  onClose: () => void;
  onDelete: () => void;
  onToast: (s: string) => void;
}) => {
  const [tab, setTab] = useState<'samples' | 'source' | 'history'>('samples');
  useEscape(onClose);

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet trace-sheet">
        <div className="sheet-head">
          <div style={{ flex: 1 }}>
            <div className="dim" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {dataset.owner === 'agent' ? 'Agent-curated' : 'You created'} · {dataset.growing ? 'auto-collecting' : 'static'}
            </div>
            <h2 style={{ marginTop: 4 }}>{dataset.name}</h2>
            <div style={{ fontSize: 13, color: 'var(--fg-muted)', marginTop: 6, lineHeight: 1.5 }}>{dataset.desc}</div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="suite-stats">
          <div>
            <div className="dim stat-label">Samples</div>
            <div className="stat-val mono">{dataset.size}</div>
          </div>
          <div>
            <div className="dim stat-label">Source</div>
            <div className="stat-val" style={{ fontSize: 14 }}>{dataset.source}</div>
          </div>
          <div>
            <div className="dim stat-label">Used for</div>
            <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>{dataset.use.map((u) => <Tag key={u}>{u}</Tag>)}</div>
          </div>
          <div style={{ flex: 1 }}>
            <div className="dim stat-label">Last update</div>
            <div className="stat-val" style={{ fontSize: 14 }}>{dataset.fresh} ago</div>
          </div>
        </div>

        <div className="sheet-tabs">
          <button className={`tab ${tab === 'samples' ? 'active' : ''}`} onClick={() => setTab('samples')}>
            Samples <span className="dim mono" style={{ fontSize: 11 }}>{dataset.samples.length}</span>
          </button>
          <button className={`tab ${tab === 'source' ? 'active' : ''}`} onClick={() => setTab('source')}>Source</button>
          <button className={`tab ${tab === 'history' ? 'active' : ''}`} onClick={() => setTab('history')}>History</button>
        </div>

        <div className="sheet-body">
          {tab === 'samples' &&
            (dataset.samples.length === 0 ? (
              <div className="dim" style={{ fontSize: 13, padding: '20px 0' }}>
                No samples yet. Run the source query or add manually.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {dataset.samples.map((s) => (
                  <div key={s.id} className="sample">
                    <div className="sample-head">
                      <span className="sample-input">{s.preview}</span>
                      <Tag>{s.tag}</Tag>
                    </div>
                  </div>
                ))}
                {dataset.size > dataset.samples.length && (
                  <div className="dim" style={{ fontSize: 12.5, padding: '12px 0', textAlign: 'center' }}>
                    + {dataset.size - dataset.samples.length} more not shown
                  </div>
                )}
              </div>
            ))}
          {tab === 'source' && (
            <div className="meta-grid">
              <div className="meta-row"><div className="dim">Source type</div><div>{dataset.sourceType === 'auto' ? 'Auto-collected' : 'Manual'}</div></div>
              <div className="meta-row"><div className="dim">Source query</div><div className="mono">{dataset.source}</div></div>
              <div className="meta-row"><div className="dim">Status</div><div>{dataset.growing ? 'Growing — adds new matches as they appear' : 'Static — manually curated'}</div></div>
              <div className="meta-row"><div className="dim">Cadence</div><div>{dataset.sourceType === 'auto' ? 'Hourly' : 'On demand'}</div></div>
            </div>
          )}
          {tab === 'history' && (
            <div>
              {dataset.history.map((h, i) => (
                <div
                  key={i}
                  style={{
                    padding: '10px 0',
                    borderBottom: i < dataset.history.length - 1 ? '1px solid var(--border)' : 'none',
                    display: 'grid',
                    gridTemplateColumns: '90px 1fr',
                    gap: 12,
                    fontSize: 13,
                  }}
                >
                  <span className="dim" style={{ fontSize: 12 }}>{h.when}</span>
                  <span>{h.what}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="sheet-foot">
          <button className="btn sm" onClick={() => onToast('Export started — check your downloads.')}>
            <Icon name="arrowDown" size={11} /> Export JSONL
          </button>
          <button className="btn sm ghost" onClick={() => onToast('Refreshing samples…')}>Refresh</button>
          <button
            className="btn sm ghost"
            style={{ marginLeft: 'auto', color: 'var(--bad-fg)' }}
            onClick={onDelete}
          >
            Delete
          </button>
        </div>
      </div>
    </>
  );
};

const NewDatasetModal = ({
  onCreate,
  onClose,
}: {
  onCreate: (d: { name: string; desc: string; source: string; sourceType: 'auto' | 'manual'; use: string[] }) => void;
  onClose: () => void;
}) => {
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [source, setSource] = useState('flagged traces');
  const [sourceType, setSourceType] = useState<'auto' | 'manual'>('auto');
  const [use, setUse] = useState<{ Eval: boolean; Distill: boolean }>({ Eval: true, Distill: false });
  useEscape(onClose);

  const sources: { id: string; label: string; value: string; type: 'auto' | 'manual' }[] = [
    { id: 'flagged', label: 'Flagged traces', value: 'flagged traces', type: 'auto' },
    { id: 'feedback', label: 'Feedback signals', value: 'feedback signals', type: 'auto' },
    { id: 'failed', label: 'Failed lookups', value: 'failed lookups', type: 'auto' },
    { id: 'manual', label: 'Manual curation', value: 'manual', type: 'manual' },
  ];

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    const usedFor = (Object.keys(use) as (keyof typeof use)[]).filter((k) => use[k]);
    if (usedFor.length === 0) return;
    onCreate({ name: name.trim(), desc: desc.trim(), source, sourceType, use: usedFor });
  };

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="modal" style={{ width: 520 }}>
        <form onSubmit={submit}>
          <div className="modal-head">
            <h2>New dataset</h2>
            <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">×</button>
          </div>
          <div className="modal-body">
            <div className="field">
              <label>Name</label>
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Slow shipping complaints" />
            </div>
            <div className="field">
              <label>Description <span className="dim">(optional)</span></label>
              <textarea value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="What's in this dataset?" rows={2} />
            </div>
            <div className="field">
              <label>Source</label>
              <div className="template-grid">
                {sources.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className={`template-card ${source === s.value ? 'on' : ''}`}
                    onClick={() => {
                      setSource(s.value);
                      setSourceType(s.type);
                    }}
                  >
                    <div className="template-name">{s.label}</div>
                    <div className="template-desc">{s.type === 'auto' ? 'Auto-collected, hourly.' : 'Add samples by hand.'}</div>
                  </button>
                ))}
              </div>
            </div>
            <div className="field">
              <label>Used for</label>
              <div style={{ display: 'flex', gap: 8 }}>
                <button type="button" className={`pill ${use.Eval ? 'on' : ''}`} onClick={() => setUse((u) => ({ ...u, Eval: !u.Eval }))}>
                  Eval
                </button>
                <button type="button" className={`pill ${use.Distill ? 'on' : ''}`} onClick={() => setUse((u) => ({ ...u, Distill: !u.Distill }))}>
                  Distill
                </button>
              </div>
            </div>
          </div>
          <div className="modal-foot">
            <button type="button" className="btn sm ghost" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn sm" disabled={!name.trim() || (!use.Eval && !use.Distill)}>
              Create dataset
            </button>
          </div>
        </form>
      </div>
    </>
  );
};

