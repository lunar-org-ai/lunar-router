/**
 * ReplayTraces — full-viewport overlay opened from Versions.
 *
 * Ported from Claude Design's ReplayTraces (in Compare.jsx). The actual
 * replay backend doesn't exist yet — this is the UX shell. When the replay
 * runner lands (re-execute trace.request through agent at version X, diff
 * outputs), it plugs into this component's three phases (idle/running/done).
 */

import { useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import type { VersionInfo } from '../api';

interface Props {
  version: VersionInfo;
  live: VersionInfo;
  onClose: () => void;
}

type Phase = 'idle' | 'running' | 'done';
type Scope = 'recent50' | 'failed' | 'flagged';

interface Row {
  id: string;
  preview: string;
  live: 'pass' | 'fail' | 'same';
  saved: 'pass' | 'fail' | 'same';
  latencyDelta: number;
}

interface Results {
  better: number;
  worse: number;
  same: number;
  latency: string;
  cost: string;
  routing: string;
  rows: Row[];
}

export const ReplayTraces = ({ version, live, onClose }: Props) => {
  const isLive = version.id === live.id;
  const [scope, setScope] = useState<Scope>('recent50');
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<Results | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const total = scope === 'recent50' ? 50 : scope === 'failed' ? 23 : 12;

  const start = () => {
    setPhase('running');
    setProgress(0);
    setResults(null);
    const tick = () => {
      setProgress((p) => {
        const next = Math.min(total, p + Math.ceil(total / 22));
        if (next >= total) {
          setTimeout(() => {
            setResults(buildResults(total, isLive));
            setPhase('done');
          }, 250);
          return total;
        }
        setTimeout(tick, 70);
        return next;
      });
    };
    setTimeout(tick, 100);
  };

  const reset = () => {
    setPhase('idle');
    setProgress(0);
    setResults(null);
  };

  return (
    <div className="fs-overlay">
      <div className="fs-backdrop" onClick={onClose} />
      <div className="fs-panel" role="dialog" aria-label="Replay traces">
        <div className="fs-head">
          <div className="fs-title">
            <h2>Replay traces</h2>
            <div className="cmp-pair">
              <span className="dim" style={{ fontSize: 13 }}>
                Replay against
              </span>
              <span className="cmp-pill saved">
                <span className="mono">{version.id}</span>
                {version.lesson?.title && <> · {version.lesson.title}</>}
              </span>
            </div>
          </div>
          <button className="fs-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="fs-body">
          {phase === 'idle' && (
            <div className="rp-setup">
              <div className="rp-step-label">Choose which traces to replay</div>
              <div className="rp-scope">
                {(
                  [
                    {
                      id: 'recent50' as Scope,
                      label: 'Recent 50 conversations',
                      sub: 'Last 24 hours from production',
                    },
                    {
                      id: 'failed' as Scope,
                      label: '23 failed traces',
                      sub: 'Conversations the live agent could not resolve',
                    },
                    {
                      id: 'flagged' as Scope,
                      label: '12 flagged by team',
                      sub: 'Manually pinned for review',
                    },
                  ] as const
                ).map((o) => (
                  <button
                    key={o.id}
                    className={`rp-scope-card ${scope === o.id ? 'on' : ''}`}
                    onClick={() => setScope(o.id)}
                  >
                    <div className="rp-scope-radio">
                      <span />
                    </div>
                    <div>
                      <div style={{ fontWeight: 500, fontSize: 14 }}>{o.label}</div>
                      <div className="dim" style={{ fontSize: 12.5, marginTop: 2 }}>
                        {o.sub}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
              <div className="rp-explain">
                <Icon name="sparkles" size={14} />
                <span>
                  Each trace is re-run through <span className="mono">{version.id}</span>
                  {!isLive && <> and the current live version</>}, then the outputs are diffed.
                  Backend replay runner is a stub — results below will be illustrative until it
                  lands.
                </span>
              </div>
              <div style={{ marginTop: 18 }}>
                <button className="btn primary" onClick={start}>
                  <Icon name="play" size={14} /> Replay {total} traces
                </button>
              </div>
            </div>
          )}

          {phase === 'running' && (
            <div className="rp-running">
              <div className="rp-progress-num">
                {progress} <span className="dim">/ {total}</span>
              </div>
              <div className="rp-progress-bar">
                <span style={{ width: `${(progress / total) * 100}%` }} />
              </div>
              <div className="dim" style={{ fontSize: 13 }}>
                {progress < total ? 'Replaying conversations…' : 'Comparing outputs…'}
              </div>
            </div>
          )}

          {phase === 'done' && results && (
            <div className="rp-results">
              <div className="rp-summary">
                <div className="rp-sum-tile better">
                  <div className="rp-sum-num">{results.better}</div>
                  <div className="rp-sum-label">Better</div>
                </div>
                <div className="rp-sum-tile worse">
                  <div className="rp-sum-num">{results.worse}</div>
                  <div className="rp-sum-label">Worse</div>
                </div>
                <div className="rp-sum-tile same">
                  <div className="rp-sum-num">{results.same}</div>
                  <div className="rp-sum-label">Unchanged</div>
                </div>
                <div className="rp-headline">
                  {results.better > results.worse ? (
                    <>
                      <strong>Net improvement.</strong> Saved version handled{' '}
                      {results.better - results.worse} more conversations correctly than live.
                    </>
                  ) : results.worse > results.better ? (
                    <>
                      <strong>Net regression.</strong> Saved version did worse on{' '}
                      {results.worse - results.better} conversations.
                    </>
                  ) : (
                    <>
                      <strong>Neutral.</strong> No meaningful difference on this set.
                    </>
                  )}
                </div>
              </div>

              <div className="rp-deltas">
                <div className="rp-delta">
                  <span className="dim">Avg latency</span>
                  <span className="mono">{results.latency}</span>
                </div>
                <div className="rp-delta">
                  <span className="dim">Avg cost / conv</span>
                  <span className="mono">{results.cost}</span>
                </div>
                <div className="rp-delta">
                  <span className="dim">Routing changes</span>
                  <span className="mono">{results.routing}</span>
                </div>
              </div>

              <div className="rp-table-head">
                <div>Trace</div>
                <div>Live verdict</div>
                <div>Saved verdict</div>
                <div>Latency Δ</div>
              </div>
              <div className="rp-table">
                {results.rows.map((r, i) => (
                  <div key={i} className="rp-row">
                    <div className="rp-trace">
                      <div className="mono dim" style={{ fontSize: 11 }}>
                        {r.id}
                      </div>
                      <div className="rp-preview">{r.preview}</div>
                    </div>
                    <div>
                      <Verdict v={r.live} />
                    </div>
                    <div>
                      <Verdict v={r.saved} />
                    </div>
                    <div
                      className={`mono ${
                        r.latencyDelta < 0 ? 'pos' : r.latencyDelta > 0 ? 'neg' : ''
                      }`}
                    >
                      {r.latencyDelta > 0 ? '+' : ''}
                      {r.latencyDelta} ms
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', gap: 8, marginTop: 18 }}>
                <button className="btn" onClick={reset}>
                  <Icon name="play" size={14} /> Run again
                </button>
                <button className="btn ghost" onClick={onClose}>
                  Close
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const Verdict = ({ v }: { v: 'pass' | 'fail' | 'same' }) => (
  <span className={`v-pill v-${v}`}>
    <span className="dot" /> {v === 'pass' ? 'Resolved' : v === 'fail' ? 'Failed' : 'Same'}
  </span>
);

const buildResults = (total: number, isLive: boolean): Results => {
  if (isLive) {
    return {
      better: 0,
      worse: 0,
      same: total,
      latency: '0 ms',
      cost: '$0.000',
      routing: `0 of ${total}`,
      rows: Array.from({ length: 5 }, () => ({
        id: 'trc_' + Math.random().toString(16).slice(2, 6) + '…',
        preview: 'Identical output — same model, same response.',
        live: 'pass' as const,
        saved: 'pass' as const,
        latencyDelta: 0,
      })),
    };
  }
  const better = Math.round(total * 0.18);
  const worse = Math.round(total * 0.04);
  const same = total - better - worse;
  const previews = [
    'Customer asked about #ORD-3318-A. Saved version normalized and resolved.',
    '"What time do you close?" — both versions answered correctly.',
    'Refund request — same response, same latency.',
    'Multi-step troubleshooting — saved version asked one fewer clarifying question.',
    '"Are you a robot?" — both said no, same wording.',
    'Order status inquiry with typo — saved version handled better.',
    'Billing dispute escalation — both routed to human.',
  ];
  const rows: Row[] = previews.map((p, i) => {
    const verdict =
      i < better
        ? { live: 'fail' as const, saved: 'pass' as const }
        : i < better + worse
        ? { live: 'pass' as const, saved: 'fail' as const }
        : { live: 'pass' as const, saved: 'pass' as const };
    const latencyDelta =
      i < better ? -120 + i * 30 : i < better + worse ? 210 : Math.round((Math.random() - 0.5) * 40);
    return {
      id: 'trc_' + (0xa3f2 + i * 47).toString(16) + '…',
      preview: p,
      ...verdict,
      latencyDelta,
    };
  });
  return {
    better,
    worse,
    same,
    latency: '+6 ms',
    cost: '+$0.002',
    routing: `${better} of ${total}`,
    rows,
  };
};
