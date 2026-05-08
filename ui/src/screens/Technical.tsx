import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import { Icon } from '../components/Icon';
import { Tag } from '../components/Tag';
import {
  ApiError,
  getReport,
  getSession,
  getSuite,
  getTrace,
  listReports,
  listSuites,
  listTraces,
  type ReportDetail,
  type ReportSummary,
  type SessionDetail,
  type SuiteDetail,
  type SuiteSummary,
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
// Traces — wired to real backend, design-faithful (P15.1 + remix)
// ============================================================================
//
// Layout follows the "OpenTracy Evolution" design's Technical.jsx:
//   - Trace row: id + when + channel · verdict dot + excerpt · model · cost
//   - Drawer:
//       header   id mono / preview h2 / meta strip
//       tabs     Transcript / Evals / Metadata
//       footer   pin / flag / copy
//
// Real data we DO have today:
//   trace_id, timestamp, request, response, duration_ms, success, error,
//   agent_version, stages[] (incl. routing_model on the route stage).
//
// Real data we DON'T have yet (rendered as "—" and labeled honestly when
// inspected, no fabrication):
//   channel       — only the webhook adapter exists, so always "webhook"
//   cost / tokens — no real cost tracking until P1.9 (real LLM)
//   turns         — 1 per trace today; real conversations need session join
//   evals         — evals run against candidates, not production traces
//   flagReason    — no flag concept persisted yet (button is local-state)

const PAGE_SIZE = 50;
type Verdict = 'pass' | 'fail';
type FilterMode = 'all' | 'pass' | 'flag' | 'fail';

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

const verdictOf = (t: TraceSummary): Verdict =>
  t.success && !t.error ? 'pass' : 'fail';

const excerptOf = (t: TraceSummary): string => {
  if (t.request) return t.request;
  if (t.response) return t.response.slice(0, 80);
  if (t.error) return `error: ${t.error}`;
  return '(empty)';
};

// Build the conversation thread for a trace.
//
// `trace.history` is every turn the caller already had in context when this
// trace ran; the trace itself contributes one more (request → response).
// Result: the full dialog up to and including this turn, rendered as chat.
//
// If the trace also belongs to a session_id and the session has more turns
// AFTER this one (e.g. another /run call later), the drawer offers a
// "View full session" toggle that swaps in the cross-trace transcript.

type Role = 'user' | 'agent' | 'tool' | 'system';

const normalizeRole = (raw: string): Role => {
  const r = (raw || '').toLowerCase();
  if (r === 'user' || r === 'customer' || r === 'human') return 'user';
  if (r === 'tool') return 'tool';
  if (r === 'system') return 'system';
  return 'agent';
};

// Stub responses from techniques/prompt_strategies/impl.py look like:
//   [stub response] Would have called <model> (max_tokens=…, temperature=…)
//   with prompt template '…' and N retrieved doc(s). Request was: '…'
// Until P1.9 wires the real LLM, every agent response in production traces
// is one of these. Reformat them in the UI so the conversation reads as
// dialog rather than debug output.
const STUB_PREFIX = '[stub response]';
const reformatStubResponse = (text: string | null): string => {
  if (!text) return '';
  const trimmed = text.trim();
  if (!trimmed.startsWith(STUB_PREFIX)) return trimmed;
  // Pull out the model name; fall back to a generic label.
  const m = trimmed.match(/Would have called ([^\s(]+)/);
  if (m && m[1]) {
    return `[Stub reply — would call ${m[1]}. Real LLM lands in P1.9.]`;
  }
  return '[Stub reply — pipeline ran without calling the LLM.]';
};

const buildTranscript = (t: TraceDetail): { role: Role; text: string }[] => {
  const out: { role: Role; text: string }[] = [];
  for (const h of t.history || []) {
    const role = normalizeRole(h.role);
    out.push({
      role,
      text: role === 'agent' ? reformatStubResponse(h.content) : h.content,
    });
  }
  out.push({ role: 'user', text: t.request || '(empty request)' });
  if (t.response) {
    out.push({ role: 'agent', text: reformatStubResponse(t.response) });
  } else if (t.error) {
    out.push({ role: 'agent', text: `[error] ${t.error}` });
  }
  return out;
};

const buildSessionTranscript = (
  s: SessionDetail,
  highlightTraceId: string | null,
): { role: Role; text: string; trace_id: string; isCurrent: boolean }[] => {
  const out: { role: Role; text: string; trace_id: string; isCurrent: boolean }[] = [];
  for (const turn of s.turns) {
    const isCurrent = turn.trace_id === highlightTraceId;
    out.push({
      role: 'user',
      text: turn.request || '(empty)',
      trace_id: turn.trace_id,
      isCurrent,
    });
    if (turn.response) {
      out.push({
        role: 'agent',
        text: reformatStubResponse(turn.response),
        trace_id: turn.trace_id,
        isCurrent,
      });
    } else if (turn.error) {
      out.push({
        role: 'agent',
        text: `[error] ${turn.error}`,
        trace_id: turn.trace_id,
        isCurrent,
      });
    }
  }
  return out;
};

export const Traces = () => {
  const [page, setPage] = useState<TracesPage | null>(null);
  const [date, setDate] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterMode>('all');
  const [versionFilter, setVersionFilter] = useState<string>('all');
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [pinned, setPinned] = useState<Record<string, boolean>>({});
  // Local-only flag set — persisting flag state to the ledger is deferred
  // (no flag concept on the backend yet). When you flag a trace from the
  // drawer, it joins this set; refresh wipes. Sufficient for triage; real
  // persistence wires later alongside the auto-rollback signals.
  const [flagged, setFlagged] = useState<Record<string, boolean>>({});
  const [toast, setToast] = useState<string | null>(null);
  useAutoToast(toast, setToast);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search.trim()), 250);
    return () => clearTimeout(t);
  }, [search]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const opts: Parameters<typeof listTraces>[0] = { limit: PAGE_SIZE, offset };
      if (date) opts.date = date;
      // Flag is a client-side overlay — backend doesn't know about it. We
      // still send the success filter for pass/fail; flagged traces are
      // filtered post-fetch via the `flagged` set.
      if (filter === 'pass') opts.success = true;
      else if (filter === 'fail') opts.success = false;
      if (versionFilter !== 'all') opts.agent_version = versionFilter;
      if (debouncedSearch) opts.q = debouncedSearch;
      const p = await listTraces(opts);
      setPage(p);
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

  const togglePin = (id: string) => {
    setPinned((p) => {
      const next = { ...p, [id]: !p[id] };
      setToast(next[id] ? 'Pinned for the agent to learn from.' : 'Unpinned.');
      return next;
    });
  };

  const flagAsFail = (id: string) => {
    setFlagged((f) => {
      const next = { ...f, [id]: !f[id] };
      setToast(
        next[id]
          ? `Flagged ${id.slice(0, 8)}… (local only until persistence lands)`
          : `Unflagged ${id.slice(0, 8)}…`,
      );
      return next;
    });
  };

  const flaggedCount = Object.values(flagged).filter(Boolean).length;

  const counts = useMemo(() => {
    if (!page) return { all: 0, pass: 0, flag: flaggedCount, fail: 0 };
    return {
      all: page.total_filtered,
      pass: page.items.filter((t) => verdictOf(t) === 'pass').length,
      flag: flaggedCount,
      fail: page.items.filter((t) => verdictOf(t) === 'fail').length,
    };
  }, [page, flaggedCount]);

  // The "flag" filter applies client-side over the page the backend already
  // returned. When `filter === 'flag'`, we drop traces that aren't in the
  // flagged set.
  const visibleItems = useMemo(() => {
    if (!page) return [];
    if (filter !== 'flag') return page.items;
    return page.items.filter((t) => flagged[t.trace_id]);
  }, [page, filter, flagged]);

  const showing = page?.items.length ?? 0;
  const total = page?.total_filtered ?? 0;
  const startIdx = total === 0 ? 0 : offset + 1;
  const endIdx = offset + showing;

  return (
    <div className="content">
      <h1 className="page-title">Traces</h1>
      <p className="page-sub">
        Every conversation the agent has had. Filter, flag, or pin one for the agent to learn from.
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
          {(
            [
              ['all', 'All', counts.all],
              ['pass', 'Passed', counts.pass],
              ['flag', 'Flagged', counts.flag],
              ['fail', 'Failed', counts.fail],
            ] as const
          ).map(([id, label, n]) => (
            <button
              key={id}
              className={`pill ${filter === id ? 'on' : ''}`}
              onClick={() => setFilter(id as FilterMode)}
            >
              {label}{' '}
              <span className="dim mono" style={{ fontSize: 11 }}>
                {n}
              </span>
            </button>
          ))}
        </div>
        <div className="trace-toolbar-right">
          <div className="search-input">
            <Icon name="search" size={13} />
            <input
              placeholder="Search excerpt or trace id…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search && (
              <button className="clear" onClick={() => setSearch('')}>
                ×
              </button>
            )}
          </div>
          <div style={{ position: 'relative' }}>
            <button
              className={`btn sm ${
                showFilters || versionFilter !== 'all' || (page && date && date !== page.available_dates[0])
                  ? ''
                  : 'ghost'
              }`}
              onClick={() => setShowFilters((s) => !s)}
            >
              <Icon name="sliders" size={12} /> Filters
              {(versionFilter !== 'all' ||
                (page && date && date !== page.available_dates[0])) && (
                <span className="filter-dot" />
              )}
            </button>
            {showFilters && (
              <>
                <div className="popover-backdrop" onClick={() => setShowFilters(false)} />
                <div className="popover">
                  {page && page.available_dates.length > 1 && (
                    <div className="popover-section">
                      <div className="popover-label">Date</div>
                      <div className="popover-options">
                        {page.available_dates.map((d) => (
                          <button
                            key={d}
                            className={`pill sm ${date === d ? 'on' : ''}`}
                            onClick={() => setDate(d)}
                          >
                            {d}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {versionOptions.length > 1 && (
                    <div className="popover-section">
                      <div className="popover-label">Agent version</div>
                      <div className="popover-options">
                        {versionOptions.map((v) => (
                          <button
                            key={v}
                            className={`pill sm ${versionFilter === v ? 'on' : ''}`}
                            onClick={() => setVersionFilter(v)}
                          >
                            {v === 'all' ? 'Any' : v}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="popover-section">
                    <div className="popover-label">Channel</div>
                    <div className="popover-options">
                      <button className="pill sm on">Any</button>
                      <button className="pill sm" disabled title="Only webhook adapter exists today">
                        webhook
                      </button>
                    </div>
                  </div>
                  <div className="popover-foot">
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        setVersionFilter('all');
                        setSearch('');
                        setFilter('all');
                        setShowFilters(false);
                      }}
                    >
                      Reset all
                    </button>
                    <button className="btn sm" onClick={() => setShowFilters(false)}>
                      Done
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {loading && !page && (
        <div className="dim" style={{ padding: 32, fontSize: 13 }}>Loading traces…</div>
      )}

      {page && (
        <div className="card">
          <div className="trace-head">
            <div>Trace</div>
            <div>Excerpt</div>
            <div>Model</div>
            <div>Cost</div>
            <div></div>
          </div>
          {visibleItems.length === 0 ? (
            <div className="empty-state">
              <div style={{ fontSize: 13, color: 'var(--fg-muted)' }}>
                {filter === 'flag'
                  ? 'No flagged traces yet. Flag one from the drawer to triage it later.'
                  : 'No traces match those filters.'}
              </div>
              <button
                className="btn sm ghost"
                style={{ marginTop: 12 }}
                onClick={() => {
                  setFilter('all');
                  setVersionFilter('all');
                  setSearch('');
                }}
              >
                Clear filters
              </button>
            </div>
          ) : (
            visibleItems.map((t) => {
              const v = verdictOf(t);
              return (
                <div
                  className="trace-row clickable"
                  key={t.trace_id}
                  onClick={() => setOpenId(t.trace_id)}
                >
                  <div className="cell-trace">
                    <div className="id-row">
                      <span className="id">{t.trace_id.slice(0, 8)}…</span>
                      {pinned[t.trace_id] && <Icon name="pin" size={11} />}
                      {flagged[t.trace_id] && <Icon name="flag" size={11} />}
                    </div>
                    <div className="dim" style={{ fontSize: 11, marginTop: 2 }}>
                      {fmtRelative(t.timestamp)} · webhook
                    </div>
                  </div>
                  <div className="cell-excerpt">
                    <Tag
                      kind={
                        flagged[t.trace_id] ? 'warn' : v === 'pass' ? 'success' : 'bad'
                      }
                    >
                      <span className="dot" />
                    </Tag>
                    <span className="preview">{excerptOf(t)}</span>
                  </div>
                  <span className="mono dim cell-model">
                    {t.routing_model || '—'}
                  </span>
                  <span className="mono dim cell-cost">—</span>
                  <Icon name="chevron" size={12} />
                </div>
              );
            })
          )}
        </div>
      )}

      <div className="trace-foot">
        <span className="dim">
          {total === 0 ? '0 traces' : `Showing ${startIdx}–${endIdx} of ${total} traces`}
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
            disabled={!page?.has_more || loading}
          >
            Next →
          </button>
        </div>
      </div>

      {openId && (
        <TraceDrawer
          traceId={openId}
          pinned={!!pinned[openId]}
          flagged={!!flagged[openId]}
          onClose={() => setOpenId(null)}
          onTogglePin={() => togglePin(openId)}
          onFlag={() => flagAsFail(openId)}
        />
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
};

const TraceDrawer = ({
  traceId,
  pinned,
  flagged,
  onClose,
  onTogglePin,
  onFlag,
}: {
  traceId: string;
  pinned: boolean;
  flagged: boolean;
  onClose: () => void;
  onTogglePin: () => void;
  onFlag: () => void;
}) => {
  const [tab, setTab] = useState<'transcript' | 'stages' | 'evals' | 'meta'>('transcript');
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

  const transcript = trace ? buildTranscript(trace) : [];
  const routingModel = trace?.stages.find((s) => s.stage === 'route')?.routing_model || null;
  const verdict: Verdict = trace ? (trace.success && !trace.error ? 'pass' : 'fail') : 'pass';

  // Pull the full session for traces that carry a session_id. If there's
  // more than one trace with the same session_id, we render the entire
  // dialog (across multiple /run calls) by default — single-turn view is
  // opt-in via "Show this turn only".
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [forceTurnOnly, setForceTurnOnly] = useState(false);
  useEffect(() => {
    if (!trace?.session_id) {
      setSession(null);
      return;
    }
    let cancelled = false;
    getSession(trace.session_id)
      .then((s) => {
        if (!cancelled) setSession(s);
      })
      .catch(() => {
        /* session fetch failure is non-fatal — single-trace view still works */
      });
    return () => {
      cancelled = true;
    };
  }, [trace?.session_id]);

  // When the user navigates to a different trace, reset the toggle so the
  // default (full session view) kicks in again.
  useEffect(() => {
    setForceTurnOnly(false);
  }, [traceId]);

  const sessionHasMore = !!session && session.n_turns > 1;
  const showSession = sessionHasMore && !forceTurnOnly;
  // Each /run call = 1 turn (one user request + one agent reply). The
  // trace.history field carries context messages from previous turns but
  // it doesn't represent additional turns this trace handled.
  const turnIndexInSession = session
    ? session.turns.findIndex((t) => t.trace_id === traceId)
    : -1;

  const copyJson = async () => {
    if (!trace) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(trace, null, 2));
    } catch {
      /* ignore */
    }
  };

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} />
      <div className="sheet trace-sheet">
        <div className="sheet-head">
          <div style={{ flex: 1 }}>
            <div className="mono" style={{ fontSize: 12, color: 'var(--fg-muted)' }}>
              {traceId}
            </div>
            <h2 style={{ marginTop: 2 }}>{trace ? excerptOf(trace) : 'Loading…'}</h2>
            {trace && (
              <div
                style={{
                  display: 'flex',
                  gap: 14,
                  marginTop: 8,
                  fontSize: 12,
                  color: 'var(--fg-muted)',
                  flexWrap: 'wrap',
                }}
              >
                <span>{fmtTimeOfDay(trace.timestamp)}</span>
                <span>·</span>
                <span>webhook</span>
                <span>·</span>
                <span className="mono">{routingModel || '—'}</span>
                <span>·</span>
                <span>
                  {sessionHasMore && session && turnIndexInSession >= 0
                    ? `Turn ${turnIndexInSession + 1} of ${session.n_turns}`
                    : '1 turn'}
                </span>
                <span>·</span>
                <span className="mono">— tok</span>
                <span>·</span>
                <span className="mono">— cost</span>
                <span>·</span>
                <span className="mono">{fmtDuration(trace.duration_ms)}</span>
                <span>·</span>
                <span className="mono">{trace.agent_version || '—'}</span>
                {trace.session_id && (
                  <>
                    <span>·</span>
                    <span className="mono" title="Session id">
                      {trace.session_id.slice(0, 14)}…
                    </span>
                  </>
                )}
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {error && (
          <div className="sheet-body">
            <div style={{ color: 'var(--bad-fg)', fontSize: 13, padding: 16 }}>{error}</div>
          </div>
        )}

        {trace && (
          <>
            {trace.error && (
              <div className="flag-banner">
                <Icon name="flag" size={12} />
                <span>
                  <b>Pipeline error:</b> {trace.error}
                </span>
              </div>
            )}
            <div className="sheet-tabs">
              <button
                className={`tab ${tab === 'transcript' ? 'active' : ''}`}
                onClick={() => setTab('transcript')}
              >
                Transcript
              </button>
              <button
                className={`tab ${tab === 'stages' ? 'active' : ''}`}
                onClick={() => setTab('stages')}
              >
                Stages{' '}
                <span className="dim mono" style={{ fontSize: 11 }}>
                  {trace.stages.length}
                </span>
              </button>
              <button
                className={`tab ${tab === 'evals' ? 'active' : ''}`}
                onClick={() => setTab('evals')}
              >
                Evals{' '}
                <span className="dim mono" style={{ fontSize: 11 }}>
                  0
                </span>
              </button>
              <button
                className={`tab ${tab === 'meta' ? 'active' : ''}`}
                onClick={() => setTab('meta')}
              >
                Metadata
              </button>
            </div>

            <div className="sheet-body">
              {tab === 'transcript' && (
                <>
                  {sessionHasMore && (
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: 12,
                        marginBottom: 12,
                        padding: '8px 12px',
                        background: 'var(--bg-muted)',
                        borderRadius: 8,
                        fontSize: 12.5,
                      }}
                    >
                      <span className="dim">
                        {showSession
                          ? `Showing all ${session?.n_turns} turns of this session.`
                          : `This trace is one turn of a ${session?.n_turns}-turn session.`}
                      </span>
                      <button
                        className="btn sm ghost"
                        onClick={() => setForceTurnOnly((v) => !v)}
                      >
                        {showSession ? 'Show this turn only' : 'View full session'}
                      </button>
                    </div>
                  )}
                  <div className="transcript">
                    {showSession && session
                      ? buildSessionTranscript(session, traceId).map((m, i) => (
                          <div
                            key={i}
                            className={`msg msg-${m.role}`}
                            title={
                              m.trace_id === traceId
                                ? `Trace you opened · ${m.trace_id.slice(0, 8)}…`
                                : `Other trace · ${m.trace_id.slice(0, 8)}…`
                            }
                          >
                            <div className="msg-role">
                              {m.role === 'agent' ? 'Agent' : 'Customer'}
                            </div>
                            <div className="msg-body">{m.text}</div>
                          </div>
                        ))
                      : transcript.map((m, i) => (
                          <div key={i} className={`msg msg-${m.role}`}>
                            <div className="msg-role">
                              {m.role === 'agent'
                                ? 'Agent'
                                : m.role === 'tool'
                                ? 'Tool'
                                : m.role === 'system'
                                ? 'System'
                                : 'Customer'}
                            </div>
                            <div className="msg-body">{m.text}</div>
                          </div>
                        ))}
                    {trace.success &&
                      transcript.length <= 1 &&
                      !showSession &&
                      !sessionHasMore && (
                        <div
                          className="dim"
                          style={{ fontSize: 12, marginTop: 12, lineHeight: 1.5 }}
                        >
                          Single-turn trace — caller didn't pass a history. Future calls that
                          include prior turns will render the full thread here.
                        </div>
                      )}
                  </div>
                </>
              )}

              {tab === 'stages' && (
                trace.stages.length === 0 ? (
                  <div className="dim" style={{ fontSize: 13, padding: '20px 0' }}>
                    No stages recorded.
                  </div>
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
                            <span className="mono dim" style={{ fontSize: 11, width: 18 }}>
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
                            <span className="mono" style={{ marginLeft: 'auto', fontSize: 12 }}>
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
                )
              )}

              {tab === 'evals' && (
                <div className="dim" style={{ fontSize: 13, padding: '20px 0', lineHeight: 1.55 }}>
                  Evals run on candidates, not production traces. Open this trace's lesson in the
                  Evolution tab to see the eval cases that promoted the agent that handled it.
                </div>
              )}

              {tab === 'meta' && (
                <div className="meta-grid">
                  <div className="meta-row">
                    <div className="dim">Trace ID</div>
                    <div className="mono">{trace.trace_id}</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Timestamp</div>
                    <div className="mono">{trace.timestamp}</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Channel</div>
                    <div>webhook</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Routing model</div>
                    <div className="mono">{routingModel || '—'}</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Duration</div>
                    <div className="mono">{fmtDuration(trace.duration_ms)}</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Stages</div>
                    <div className="mono">{trace.stages.length}</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Agent version</div>
                    <div className="mono">{trace.agent_version || '—'}</div>
                  </div>
                  <div className="meta-row">
                    <div className="dim">Verdict</div>
                    <div>
                      <Tag kind={verdict === 'pass' ? 'success' : 'bad'}>{verdict}</Tag>
                    </div>
                  </div>
                  {Object.keys(trace.metadata || {}).length > 0 && (
                    <div className="meta-row">
                      <div className="dim">Metadata</div>
                      <pre
                        style={{
                          margin: 0,
                          fontFamily: 'var(--font-mono)',
                          fontSize: 11.5,
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {JSON.stringify(trace.metadata, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="sheet-foot">
              <button className={`btn sm ${pinned ? '' : 'ghost'}`} onClick={onTogglePin}>
                <Icon name="pin" size={12} /> {pinned ? 'Pinned' : 'Pin for learning'}
              </button>
              <button className={`btn sm ${flagged ? '' : 'ghost'}`} onClick={onFlag}>
                <Icon name="flag" size={12} /> {flagged ? 'Flagged' : 'Flag as failure'}
              </button>
              <button className="btn sm ghost" style={{ marginLeft: 'auto' }} onClick={copyJson}>
                <Icon name="copy" size={12} /> Copy trace
              </button>
            </div>
          </>
        )}

        {loading && !trace && (
          <div className="sheet-body">
            <div className="dim" style={{ fontSize: 13, padding: 24 }}>Loading…</div>
          </div>
        )}
      </div>
    </>
  );
};


// ============================================================================
// Eval suites — wired to real backend (P15.2)
// ============================================================================
//
// Suites read from /v1/evals/suites; the smoke_v0 suite today carries 5
// goldens and 4 rubrics (pipeline_succeeded / response_nonempty /
// keywords_match / under_budget). Reports under /v1/evals/reports include
// both candidate runs (cand_<id>.json, written by experiments/runner.py
// since P12) and baseline runs (smoke_v0_<ts>.json from a plain run_suite).
// We render both as one timeline so the operator can watch a suite's
// pass-rate move over agent versions.

const fmtPercent = (v: number | null | undefined): string =>
  v == null ? '—' : `${(v * 100).toFixed(0)}%`;

const fmtIsoDay = (iso: string | null): string => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
};

export const EvalSuites = () => {
  const [suites, setSuites] = useState<SuiteSummary[] | null>(null);
  const [search, setSearch] = useState('');
  const [openName, setOpenName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  useAutoToast(toast, setToast);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await listSuites();
      setSuites(list);
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

  const filtered = useMemo(() => {
    if (!suites) return [];
    const q = search.trim().toLowerCase();
    if (!q) return suites;
    return suites.filter((s) => s.name.toLowerCase().includes(q));
  }, [suites, search]);

  return (
    <div className="content">
      <h1 className="page-title">Eval suites</h1>
      <p className="page-sub">
        The self-tests every candidate is scored against. Suites are YAML files
        under <span className="mono">evals/suites/</span>; runs land in{' '}
        <span className="mono">evals/reports/</span> and feed the trend below.
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
          <button className="pill on">
            All <span className="dim mono" style={{ fontSize: 11 }}>{suites?.length ?? 0}</span>
          </button>
        </div>
        <div className="trace-toolbar-right">
          <div className="search-input">
            <Icon name="search" size={13} />
            <input
              placeholder="Search suites…"
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
            className="btn sm"
            disabled
            title="Triggering eval runs from the UI is deferred — for now, run via the harness CLI"
          >
            <Icon name="play" size={11} /> Run all
          </button>
        </div>
      </div>

      {loading && !suites && (
        <div className="dim" style={{ padding: 32, fontSize: 13 }}>Loading suites…</div>
      )}

      {suites && (
        <div className="card">
          <div className="suite-head">
            <div>Suite</div>
            <div>Goldens</div>
            <div>Last pass rate</div>
            <div>Last run</div>
            <div>Version</div>
            <div></div>
          </div>
          {filtered.length === 0 ? (
            <div className="empty-state" style={{ padding: 48 }}>
              <div style={{ fontSize: 13, color: 'var(--fg-muted)' }}>
                {suites.length === 0
                  ? 'No suites defined yet. Add a YAML to evals/suites/.'
                  : 'No suites match.'}
              </div>
            </div>
          ) : (
            filtered.map((s) => (
              <div className="suite-row" key={s.name} onClick={() => setOpenName(s.name)}>
                <div>
                  <div
                    style={{
                      fontWeight: 500,
                      fontSize: 13.5,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                    }}
                  >
                    {s.name}
                  </div>
                  {s.description && (
                    <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                      {s.description}
                    </div>
                  )}
                  <div style={{ marginTop: 6 }}>
                    <div className="bar" style={{ width: 200 }}>
                      <span style={{ width: `${(s.last_pass_rate ?? 0) * 100}%` }} />
                    </div>
                  </div>
                </div>
                <span className="mono dim" style={{ fontSize: 12.5 }}>{s.n_goldens}</span>
                <span className="mono" style={{ fontWeight: 500 }}>
                  {fmtPercent(s.last_pass_rate)}
                </span>
                <span className="dim" style={{ fontSize: 12 }}>{fmtIsoDay(s.last_run_at)}</span>
                <span className="mono dim" style={{ fontSize: 11.5 }}>
                  {s.last_agent_version || '—'}
                </span>
                <Icon name="chevron" size={12} />
              </div>
            ))
          )}
        </div>
      )}

      <div className="trace-foot">
        <span className="dim">
          Showing {filtered.length} of {suites?.length ?? 0}
        </span>
        <span className="dim" style={{ marginLeft: 'auto' }}>
          {suites && suites.length > 0
            ? `Total runs across all suites: ${suites.reduce(
                (s, x) => s + x.n_runs,
                0,
              )}`
            : ''}
        </span>
      </div>

      <button
        className="add-btn"
        style={{ marginTop: 16, maxWidth: 320 }}
        disabled
        title="Creating suites from the UI is deferred — add YAML files to evals/suites/ for now"
      >
        + New eval suite (coming soon)
      </button>

      {openName && (
        <SuiteDrawer
          name={openName}
          onClose={() => setOpenName(null)}
          onToast={setToast}
        />
      )}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
};

const SuiteDrawer = ({
  name,
  onClose,
  onToast,
}: {
  name: string;
  onClose: () => void;
  onToast: (s: string) => void;
}) => {
  const [tab, setTab] = useState<'goldens' | 'history' | 'rubrics'>('goldens');
  const [detail, setDetail] = useState<SuiteDetail | null>(null);
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openReport, setOpenReport] = useState<string | null>(null);
  useEscape(onClose);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getSuite(name), listReports({ suite: name, limit: 20 })])
      .then(([d, rs]) => {
        if (cancelled) return;
        setDetail(d);
        setReports(rs);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(
            e instanceof ApiError
              ? `Backend ${e.status}: ${e.message}`
              : `Error: ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [name]);

  // Sparkline of pass_rate over time (oldest → newest, left → right)
  const sparkline = useMemo(() => {
    if (!reports || reports.length === 0) return null;
    const ordered = [...reports].reverse(); // chronological
    const w = 280;
    const h = 56;
    const pad = 4;
    const pts = ordered.map((r, i) => {
      const x = pad + (i / Math.max(1, ordered.length - 1)) * (w - pad * 2);
      const y = pad + (1 - r.pass_rate) * (h - pad * 2);
      return [x, y, r] as const;
    });
    const path = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
    return { w, h, pts, path };
  }, [reports]);

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
              Suite · {detail?.aggregation || 'mean'} aggregation
            </div>
            <h2 style={{ marginTop: 4 }}>{name}</h2>
            {detail?.description && (
              <div style={{ fontSize: 13, color: 'var(--fg-muted)', marginTop: 6, lineHeight: 1.5 }}>
                {detail.description}
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
        </div>

        {error && (
          <div className="sheet-body">
            <div style={{ color: 'var(--bad-fg)', fontSize: 13, padding: 16 }}>{error}</div>
          </div>
        )}

        {detail && (
          <>
            <div className="suite-stats">
              <div>
                <div className="dim stat-label">Pass rate</div>
                <div className="stat-val">
                  {detail.last_pass_rate != null
                    ? (detail.last_pass_rate * 100).toFixed(0)
                    : '—'}
                  <span className="dim mono" style={{ fontSize: 14, fontWeight: 400 }}>%</span>
                </div>
              </div>
              <div>
                <div className="dim stat-label">Overall score</div>
                <div className="stat-val mono">
                  {detail.last_overall_score != null
                    ? detail.last_overall_score.toFixed(3)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="dim stat-label">Goldens</div>
                <div className="stat-val mono">{detail.n_goldens}</div>
              </div>
              <div>
                <div className="dim stat-label">Runs</div>
                <div className="stat-val mono">{detail.n_runs}</div>
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="dim stat-label">Pass-rate trend</div>
                {sparkline && sparkline.pts.length >= 2 ? (
                  <svg className="sparkline" width={sparkline.w} height={sparkline.h}>
                    <path
                      d={sparkline.path}
                      fill="none"
                      stroke="var(--accent)"
                      strokeWidth="1.6"
                    />
                    {sparkline.pts.map((p, i) => (
                      <circle key={i} cx={p[0]} cy={p[1]} r="2" fill="var(--accent)" />
                    ))}
                  </svg>
                ) : (
                  <div className="dim" style={{ fontSize: 12.5, paddingTop: 12 }}>
                    {sparkline ? '1 run — need ≥ 2 for trend' : '—'}
                  </div>
                )}
              </div>
            </div>

            <div className="sheet-tabs">
              <button
                className={`tab ${tab === 'goldens' ? 'active' : ''}`}
                onClick={() => setTab('goldens')}
              >
                Goldens{' '}
                <span className="dim mono" style={{ fontSize: 11 }}>{detail.goldens.length}</span>
              </button>
              <button
                className={`tab ${tab === 'history' ? 'active' : ''}`}
                onClick={() => setTab('history')}
              >
                History{' '}
                <span className="dim mono" style={{ fontSize: 11 }}>{reports?.length ?? 0}</span>
              </button>
              <button
                className={`tab ${tab === 'rubrics' ? 'active' : ''}`}
                onClick={() => setTab('rubrics')}
              >
                Rubrics{' '}
                <span className="dim mono" style={{ fontSize: 11 }}>{detail.rubrics.length}</span>
              </button>
            </div>

            <div className="sheet-body">
              {tab === 'goldens' && (
                detail.goldens.length === 0 ? (
                  <div className="dim" style={{ fontSize: 13, padding: '20px 0' }}>
                    No goldens linked.
                  </div>
                ) : (
                  <div className="samples">
                    {detail.goldens.map((g) => {
                      const expected = g.expected as { contains?: string[]; category?: string };
                      return (
                        <div key={g.id} className="sample">
                          <div className="sample-head">
                            <span className="sample-input">"{g.request}"</span>
                            <span className="mono dim" style={{ fontSize: 11.5 }}>{g.id}</span>
                          </div>
                          {expected?.contains && expected.contains.length > 0 && (
                            <div className="sample-row">
                              <span className="sample-label">Expects keywords</span>
                              <span className="mono" style={{ fontSize: 12 }}>
                                {expected.contains.join(', ')}
                              </span>
                            </div>
                          )}
                          {expected?.category && (
                            <div className="sample-row">
                              <span className="sample-label">Category</span>
                              <span>{String(expected.category)}</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )
              )}
              {tab === 'history' && (
                reports == null || reports.length === 0 ? (
                  <div className="dim" style={{ fontSize: 13, padding: '20px 0' }}>
                    No runs recorded for this suite yet.
                  </div>
                ) : (
                  <div>
                    <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
                      {reports.length} run{reports.length === 1 ? '' : 's'} · click a row for case detail.
                    </div>
                    {reports.map((r) => (
                      <div
                        key={r.report_id}
                        className="history-row"
                        style={{ cursor: 'pointer' }}
                        onClick={() => setOpenReport(r.report_id)}
                      >
                        <span className="dim mono" style={{ fontSize: 11.5, width: 100 }}>
                          {fmtIsoDay(r.finished_at)}
                        </span>
                        <span className="mono dim" style={{ fontSize: 11.5, width: 60 }}>
                          {r.agent_version}
                        </span>
                        <div className="bar" style={{ flex: 1 }}>
                          <span style={{ width: `${r.pass_rate * 100}%` }} />
                        </div>
                        <span className="mono" style={{ fontSize: 12.5, width: 70, textAlign: 'right' }}>
                          {r.n_passed}/{r.n_total}
                        </span>
                        {r.is_candidate && <Tag kind="info">cand</Tag>}
                      </div>
                    ))}
                  </div>
                )
              )}
              {tab === 'rubrics' && (
                <div className="meta-grid">
                  {detail.rubrics.map((r) => (
                    <div key={r.name} className="meta-row">
                      <div className="dim mono">{r.name}</div>
                      <div>
                        <span className="mono" style={{ fontSize: 12.5 }}>{r.type}</span>
                        {Object.keys(r.params || {}).length > 0 && (
                          <div className="dim" style={{ fontSize: 11.5, marginTop: 4 }}>
                            params: {JSON.stringify(r.params)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="sheet-foot">
              <button
                className="btn sm"
                disabled
                title="Triggering eval runs from the UI is deferred — use the harness CLI"
              >
                <Icon name="play" size={11} /> Run now (coming soon)
              </button>
              <button
                className="btn sm ghost"
                onClick={() => onToast(`Suite YAML at evals/suites/${name}.yaml`)}
              >
                <Icon name="file" size={12} /> Show file path
              </button>
            </div>
          </>
        )}
      </div>
      {openReport && <ReportDrawer reportId={openReport} onClose={() => setOpenReport(null)} />}
    </>
  );
};

const ReportDrawer = ({
  reportId,
  onClose,
}: {
  reportId: string;
  onClose: () => void;
}) => {
  const [report, setReport] = useState<ReportDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEscape(onClose);

  useEffect(() => {
    let cancelled = false;
    getReport(reportId)
      .then((r) => {
        if (!cancelled) setReport(r);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(
            e instanceof ApiError
              ? `Backend ${e.status}: ${e.message}`
              : `Error: ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [reportId]);

  return (
    <>
      <div
        className="sheet-backdrop"
        onClick={onClose}
        style={{ zIndex: 12 }}
      />
      <div className="sheet trace-sheet" style={{ zIndex: 13 }}>
        <div className="sheet-head">
          <div style={{ flex: 1 }}>
            <div
              className="dim"
              style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}
            >
              Report · {report?.is_candidate ? 'candidate run' : 'baseline run'}
            </div>
            <h2 className="mono" style={{ marginTop: 4, fontSize: 15 }}>{reportId}</h2>
            {report && (
              <div className="dim" style={{ fontSize: 12.5, marginTop: 6 }}>
                {fmtIsoDay(report.finished_at)} · {report.agent_version} · {report.n_passed}/{report.n_total} passed
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="sheet-body">
          {error && (
            <div style={{ color: 'var(--bad-fg)', fontSize: 13 }}>{error}</div>
          )}
          {!report && !error && (
            <div className="dim" style={{ fontSize: 13 }}>Loading…</div>
          )}
          {report && (
            <>
              <div className="sheet-section">
                <h3>Per-rubric</h3>
                <div className="meta-grid">
                  {Object.entries(report.per_rubric || {}).map(([k, v]) => (
                    <div key={k} className="meta-row">
                      <div className="dim mono">{k}</div>
                      <div className="mono">{typeof v === 'number' ? v.toFixed(3) : String(v)}</div>
                    </div>
                  ))}
                  {Object.keys(report.per_rubric || {}).length === 0 && (
                    <div className="dim" style={{ fontSize: 13 }}>No per-rubric breakdown.</div>
                  )}
                </div>
              </div>
              <div className="sheet-section">
                <h3>
                  Cases <span className="dim mono" style={{ fontSize: 12, fontWeight: 400 }}>· {report.cases.length}</span>
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {report.cases.map((c, i) => {
                    const passed = c.rubric_results.filter((r) => r.passed).length;
                    const total = c.rubric_results.length;
                    return (
                      <div key={c.trace_id || i} className={`sample ${c.success ? 'pass' : 'fail'}`}>
                        <div className="sample-head">
                          <span className="sample-input">"{c.request}"</span>
                          <span
                            className="mono"
                            style={{
                              fontSize: 11.5,
                              fontWeight: 500,
                              color: c.success ? 'var(--accent-fg)' : 'var(--bad-fg)',
                            }}
                          >
                            {passed}/{total}
                          </span>
                        </div>
                        <div className="sample-row">
                          <span className="sample-label">Golden</span>
                          <span className="mono" style={{ fontSize: 12 }}>{c.golden_id}</span>
                        </div>
                        {c.response && (
                          <div className="sample-row">
                            <span className="sample-label">Response</span>
                            <span style={{ fontSize: 12.5 }}>{c.response.slice(0, 200)}…</span>
                          </div>
                        )}
                        {c.trace_id && (
                          <div className="sample-row">
                            <span className="sample-label">Trace</span>
                            <span className="mono dim" style={{ fontSize: 11.5 }}>{c.trace_id}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
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

