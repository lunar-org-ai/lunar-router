import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from '@tanstack/react-router';
import { Icon } from '../components/Icon';
import {
  ApiError,
  listTraces,
  subscribeTraces,
  type LiveTraceEvent,
  type TraceSummary,
} from '../api';

type Row = LiveTraceEvent & { _live: boolean };

const BACKFILL = 100;

const fromSummary = (t: TraceSummary): Row => ({
  trace_id: t.trace_id,
  timestamp: t.timestamp,
  session_id: t.session_id,
  agent_version: t.agent_version,
  duration_ms: t.duration_ms,
  success: t.success,
  error: t.error,
  n_stages: t.n_stages,
  n_turns: t.n_turns,
  request_preview: t.request,
  _live: false,
});

const fmtRelative = (iso: string): string => {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const ageS = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (ageS < 5) return 'just now';
  if (ageS < 60) return `${ageS}s ago`;
  if (ageS < 3600) return `${Math.floor(ageS / 60)}m ago`;
  if (ageS < 86400) return `${Math.floor(ageS / 3600)}h ago`;
  return new Date(t).toLocaleDateString();
};

export const TracesLive = () => {
  const navigate = useNavigate();
  const goTraces = () => navigate({ to: '/technical/traces' });

  const [rows, setRows] = useState<Row[]>([]);
  const [paused, setPaused] = useState(false);
  const [status, setStatus] = useState<'connecting' | 'live' | 'paused' | 'error' | 'idle'>(
    'connecting',
  );
  const [error, setError] = useState<string | null>(null);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // Backfill once on mount, then attach SSE.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const page = await listTraces({ limit: BACKFILL });
        if (cancelled) return;
        setRows(page.items.map(fromSummary));
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError) setError(`backfill: ${e.message}`);
        else setError('backfill failed');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const close = subscribeTraces({
      onOpen: () => setStatus((s) => (s === 'paused' ? s : 'live')),
      onError: () => setStatus('error'),
      onTrace: (event) => {
        if (pausedRef.current) return;
        setRows((prev) => {
          // Replace if already present (backfill might have it), else prepend.
          const filtered = prev.filter((r) => r.trace_id !== event.trace_id);
          return [{ ...event, _live: true }, ...filtered].slice(0, 500);
        });
      },
    });
    return close;
  }, []);

  useEffect(() => {
    setStatus(paused ? 'paused' : 'live');
  }, [paused]);

  const liveCount = useMemo(() => rows.filter((r) => r._live).length, [rows]);

  return (
    <div className="screen-pad">
      <div className="card-row" style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h2 style={{ margin: 0 }}>Live traces</h2>
          <StatusPill status={status} />
          <span className="dim" style={{ fontSize: 12 }}>
            {rows.length} loaded · {liveCount} streamed since backfill
          </span>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button
            className="btn sm ghost"
            onClick={() => setPaused((p) => !p)}
            title={paused ? 'Resume the live stream' : 'Pause — new traces will be ignored'}
          >
            <Icon name={paused ? 'play' : 'pause'} size={12} />
            <span>{paused ? 'Resume' : 'Pause'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div
          className="card card-pad"
          style={{ borderColor: 'var(--bad)', marginBottom: 12, fontSize: 13 }}
        >
          {error}
        </div>
      )}

      <div className="card">
        <div className="trace-list">
          {rows.length === 0 ? (
            <div className="card-pad dim" style={{ fontSize: 13 }}>
              {status === 'connecting' ? 'Connecting…' : 'No traces yet — fire a request.'}
            </div>
          ) : (
            rows.map((r) => (
              <div
                key={r.trace_id}
                className={`trace-row clickable ${r._live ? 'live-fresh' : ''}`}
                onClick={goTraces}
              >
                <div className="cell-trace">
                  <div className="id-row">
                    <span className="id">{r.trace_id.slice(0, 8)}…</span>
                    {r._live && (
                      <span
                        className="badge"
                        style={{ marginLeft: 6, fontSize: 9, padding: '0 6px' }}
                      >
                        LIVE
                      </span>
                    )}
                  </div>
                  <div className="dim" style={{ fontSize: 11, marginTop: 2 }}>
                    {fmtRelative(r.timestamp)} ·{' '}
                    {r.session_id ? (
                      <button
                        className="link"
                        onClick={(e) => {
                          e.stopPropagation();
                          goTraces();
                        }}
                        style={{
                          background: 'none',
                          border: 0,
                          padding: 0,
                          color: 'inherit',
                          textDecoration: 'underline',
                          cursor: 'pointer',
                          font: 'inherit',
                        }}
                      >
                        session {r.session_id.slice(0, 10)}
                      </button>
                    ) : (
                      'no session'
                    )}
                  </div>
                </div>
                <div className="cell-excerpt">
                  <span
                    className="dot"
                    style={{
                      background: r.success ? 'var(--good, #22c55e)' : 'var(--bad, #ef4444)',
                    }}
                  />
                  <span className="preview">{r.request_preview}</span>
                </div>
                <span className="mono dim cell-model">{r.agent_version || '—'}</span>
                <span className="mono dim cell-cost">{r.duration_ms.toFixed(1)}ms</span>
                <Icon name="chevron" size={12} />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const StatusPill = ({
  status,
}: {
  status: 'connecting' | 'live' | 'paused' | 'error' | 'idle';
}) => {
  const map: Record<typeof status, { label: string; color: string }> = {
    connecting: { label: 'Connecting…', color: 'var(--accent, #888)' },
    live: { label: 'Live', color: 'var(--good, #22c55e)' },
    paused: { label: 'Paused', color: 'var(--accent, #888)' },
    error: { label: 'Disconnected', color: 'var(--bad, #ef4444)' },
    idle: { label: 'Idle', color: 'var(--accent, #888)' },
  };
  const { label, color } = map[status];
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 11,
        padding: '2px 8px',
        borderRadius: 999,
        border: '1px solid var(--border, #333)',
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: color,
          display: 'inline-block',
        }}
      />
      {label}
    </span>
  );
};
