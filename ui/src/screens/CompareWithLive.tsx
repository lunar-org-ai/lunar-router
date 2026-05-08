/**
 * CompareWithLive — full-viewport overlay opened from Versions.
 *
 * Ported from Claude Design's Compare.jsx. Shows a saved version side-by-side
 * with the current live one across 5 tabs: overview, samples, prompts,
 * routing, evals.
 *
 * Data caveat: production-style sample conversations and synthesized metrics
 * are illustrative — they're rendered when we don't yet have a real
 * comparison runner. When that lands (eval suite replay against both
 * versions), the same UI consumes the real numbers.
 */

import { useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import type { VersionInfo } from '../api';

interface Props {
  version: VersionInfo;
  live: VersionInfo;
  onClose: () => void;
}

type Tab = 'overview' | 'samples' | 'prompts' | 'routing' | 'evals';

const SAMPLES = [
  {
    q: 'Where is order #ORD-3318-A?',
    saved: 'Found order ORD3318A — out for delivery, ETA tomorrow 2pm.',
    live: 'I cannot find that order. Could you check the format?',
    verdict: 'better' as const,
  },
  {
    q: 'What time do you close on Sunday?',
    saved: 'We close at 6pm on Sundays.',
    live: 'We close at 6pm on Sundays.',
    verdict: 'same' as const,
  },
  {
    q: 'I want to return a duplicate charge.',
    saved: 'I see this is frustrating — refunding the duplicate now.',
    live: 'I see this is frustrating — refunding the duplicate now.',
    verdict: 'same' as const,
  },
];

export const CompareWithLive = ({ version, live, onClose }: Props) => {
  const [tab, setTab] = useState<Tab>('overview');
  const isLive = version.id === live.id;
  const lesson = version.lesson;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const summary = isLive
    ? { same: SAMPLES.length, better: 0, worse: 0 }
    : { same: 2, better: 1, worse: 0 };

  return (
    <div className="fs-overlay">
      <div className="fs-backdrop" onClick={onClose} />
      <div className="fs-panel" role="dialog" aria-label="Compare with live">
        <div className="fs-head">
          <div className="fs-title">
            <h2>Compare</h2>
            <div className="cmp-pair">
              <span className="cmp-pill saved">
                <span className="mono">{version.id}</span>
                {lesson?.title && <> · {lesson.title}</>}
              </span>
              <span className="cmp-vs">vs</span>
              <span className="cmp-pill live">
                <span className="dot" /> Live · <span className="mono">{live.id}</span>
              </span>
            </div>
          </div>
          <button className="fs-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {!isLive && (
          <div className="cmp-verdict">
            <span className="cmp-stat better">
              <Icon name="check" size={14} /> {summary.better} better
            </span>
            <span className="cmp-stat worse">
              <Icon name="info" size={14} /> {summary.worse} worse
            </span>
            <span className="cmp-stat same">{summary.same} unchanged</span>
            <span className="dim" style={{ marginLeft: 'auto', fontSize: 12.5 }}>
              Synthesized — last 50 conversations replayed against both versions.
            </span>
          </div>
        )}
        {isLive && (
          <div className="cmp-verdict">
            <span className="cmp-stat same">This is the version currently live. Nothing to compare.</span>
          </div>
        )}

        <div className="fs-tabs">
          {(['overview', 'samples', 'prompts', 'routing', 'evals'] as Tab[]).map((t) => (
            <button key={t} className={`fs-tab ${tab === t ? 'on' : ''}`} onClick={() => setTab(t)}>
              {t === 'overview'
                ? 'Overview'
                : t === 'samples'
                ? 'Output samples'
                : t === 'prompts'
                ? 'Prompts'
                : t === 'routing'
                ? 'Routing'
                : 'Evals & metrics'}
            </button>
          ))}
        </div>

        <div className="fs-body">
          {tab === 'overview' && <CmpOverview version={version} live={live} isLive={isLive} />}
          {tab === 'samples' && <CmpSamples isLive={isLive} />}
          {tab === 'prompts' && <CmpPrompts version={version} isLive={isLive} />}
          {tab === 'routing' && <CmpRouting version={version} isLive={isLive} />}
          {tab === 'evals' && <CmpEvals version={version} isLive={isLive} />}
        </div>
      </div>
    </div>
  );
};

const CmpOverview = ({
  version,
  isLive,
}: {
  version: VersionInfo;
  live: VersionInfo;
  isLive: boolean;
}) => {
  const tiles = isLive
    ? [
        { label: 'Resolution rate', value: '78%', d: '—' },
        { label: 'CSAT', value: '4.5', d: '—' },
        { label: 'Avg cost / conv', value: '$0.041', d: '—' },
        { label: 'Latency p50', value: '412 ms', d: '—' },
      ]
    : [
        { label: 'Resolution rate', value: '82%', d: '+4%', dir: 'up' as const },
        { label: 'CSAT', value: '4.6', d: '+0.1', dir: 'up' as const },
        { label: 'Avg cost / conv', value: '$0.043', d: '+5%', dir: 'down' as const },
        { label: 'Latency p50', value: '418 ms', d: '+6 ms', dir: 'flat' as const },
      ];
  const lesson = version.lesson;

  return (
    <div className="cmp-overview">
      <div className="cmp-tiles">
        {tiles.map((t) => (
          <div key={t.label} className="cmp-tile">
            <div
              className="dim"
              style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}
            >
              {t.label}
            </div>
            <div style={{ fontSize: 22, fontWeight: 500, letterSpacing: '-0.01em', marginTop: 4 }}>
              {t.value}
            </div>
            <div className={`cmp-delta ${('dir' in t && t.dir) || ''}`}>{t.d}</div>
          </div>
        ))}
      </div>

      <div className="dim" style={{ fontSize: 12, marginBottom: 18 }}>
        Tiles above are synthesized for now — once the eval suite supports replay against any version,
        these will be real numbers.
      </div>

      {!isLive && lesson && (
        <>
          <div className="cmp-section-label">What changed in {version.id}</div>
          <div className="card card-pad" style={{ marginBottom: 18 }}>
            {lesson.voice ? (
              <div style={{ fontStyle: 'italic', fontSize: 14, lineHeight: 1.55 }}>
                “{lesson.voice}”
              </div>
            ) : (
              <div style={{ fontSize: 14, lineHeight: 1.55 }}>{lesson.summary}</div>
            )}
            {lesson.mutations.length > 0 && (
              <div
                className="dim"
                style={{ fontSize: 12, marginTop: 12, fontFamily: 'var(--font-mono)' }}
              >
                {lesson.mutations.map((m) => (
                  <div key={m}>{m}</div>
                ))}
              </div>
            )}
          </div>

          <div className="cmp-recommendation">
            <Icon name="sparkles" size={16} />
            <div>
              <div style={{ fontWeight: 500, marginBottom: 2 }}>Recommendation</div>
              <div className="dim" style={{ fontSize: 13, lineHeight: 1.5 }}>
                Once a comparison run produces real numbers, this slot will show whether to promote.
                For now it's just the lesson context above.
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

const CmpSamples = ({ isLive }: { isLive: boolean }) => (
  <div>
    <div className="dim" style={{ fontSize: 12.5, marginBottom: 14 }}>
      Three illustrative conversations replayed against both versions. Real samples will appear here
      when the replay runner lands.
    </div>
    {SAMPLES.map((s, i) => (
      <div key={i} className="cmp-sample">
        <div className="cmp-sample-q">
          <span className="dim">User:</span> {s.q}
        </div>
        <div className="cmp-sample-grid">
          <div className="cmp-sample-side">
            <div className="cmp-side-head">Saved version</div>
            <div className="cmp-side-body">{s.saved}</div>
          </div>
          <div className="cmp-sample-side live">
            <div className="cmp-side-head">
              <span className="dot" /> Live
            </div>
            <div className="cmp-side-body">{s.live}</div>
          </div>
        </div>
        {!isLive && (
          <div className={`cmp-sample-verdict ${s.verdict}`}>
            {s.verdict === 'better' && (
              <>
                <Icon name="check" size={12} /> Saved version handled this better
              </>
            )}
            {s.verdict === 'same' && <>Both versions answered the same way</>}
          </div>
        )}
      </div>
    ))}
  </div>
);

const CmpPrompts = ({ version, isLive }: { version: VersionInfo; isLive: boolean }) => {
  const lesson = version.lesson;
  const promptMutations = lesson?.mutations.filter((m) => /prompt/i.test(m)) ?? [];

  if (isLive || !lesson || promptMutations.length === 0) {
    return <div className="cmp-empty">No prompt differences between {version.id} and live.</div>;
  }

  return (
    <div>
      <div className="dim" style={{ fontSize: 12.5, marginBottom: 14 }}>
        Prompt-related mutations in this version. Line-level diff lands when promoter writes the
        before/after text.
      </div>
      <div className="card card-pad" style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5 }}>
        {promptMutations.map((m) => (
          <div key={m}>{m}</div>
        ))}
      </div>
    </div>
  );
};

const CmpRouting = ({ version, isLive }: { version: VersionInfo; isLive: boolean }) => {
  const lesson = version.lesson;
  const routingMutations = lesson?.mutations.filter((m) => /route|router/i.test(m)) ?? [];

  if (isLive || !lesson || routingMutations.length === 0) {
    return <div className="cmp-empty">No routing differences. Both versions use the same router config.</div>;
  }

  return (
    <div>
      <div className="dim" style={{ fontSize: 12.5, marginBottom: 14 }}>
        Routing-related mutations.
      </div>
      <div className="card card-pad" style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5 }}>
        {routingMutations.map((m) => (
          <div key={m}>{m}</div>
        ))}
      </div>
    </div>
  );
};

const CmpEvals = ({ version, isLive }: { version: VersionInfo; isLive: boolean }) => {
  const lesson = version.lesson;
  const perRubric = lesson?.delta?.per_rubric ?? {};
  const rubrics = Object.entries(perRubric);

  // If we have real per-rubric deltas from the lesson, show them; else fall back
  // to synthesized metrics matching the design.
  const useReal = rubrics.length > 0;

  const synth = [
    { name: 'Resolution rate', before: 0.78, after: 0.82 },
    { name: 'CSAT', before: 4.5, after: 4.6, scale: 5 },
    { name: 'Latency p50', before: 412, after: 418, isLatency: true },
  ];

  const metrics = useReal
    ? rubrics.map(([name, delta]) => ({
        name,
        before: 0.5,
        after: 0.5 + delta,
      }))
    : synth;

  return (
    <div>
      <div className="dim" style={{ fontSize: 12.5, marginBottom: 14 }}>
        {useReal
          ? 'Per-rubric deltas from this version against its baseline.'
          : 'Eval suite scores. Higher is better unless marked. Synthesized for now.'}
      </div>
      <div className="cmp-evals">
        {metrics.map((m) => {
          const before = m.before ?? 0;
          const after = m.after ?? 0;
          const lowerBetter = 'isLatency' in m && m.isLatency;
          const scaleVal = 'scale' in m && typeof m.scale === 'number' ? m.scale : null;
          const max: number =
            scaleVal ?? (lowerBetter ? Math.max(before, after) * 1.1 : 1);
          const delta = after - before;
          const improved = lowerBetter ? delta < 0 : delta > 0;
          return (
            <div key={m.name} className="cmp-eval">
              <div className="cmp-eval-name">{m.name}</div>
              <div className="cmp-eval-bars">
                <div className="cmp-eval-row">
                  <span className="cmp-eval-label">Saved</span>
                  <div className="cmp-bar">
                    <span style={{ width: `${(before / max) * 100}%` }} />
                  </div>
                  <span className="cmp-eval-value mono">
                    {'scale' in m && m.scale
                      ? before.toFixed(1)
                      : lowerBetter
                      ? before
                      : (before * 100).toFixed(0) + '%'}
                  </span>
                </div>
                <div className="cmp-eval-row">
                  <span className="cmp-eval-label">Live</span>
                  <div className="cmp-bar live">
                    <span style={{ width: `${(after / max) * 100}%` }} />
                  </div>
                  <span className="cmp-eval-value mono">
                    {'scale' in m && m.scale
                      ? after.toFixed(1)
                      : lowerBetter
                      ? after
                      : (after * 100).toFixed(0) + '%'}
                  </span>
                </div>
              </div>
              {!isLive && (
                <div className={`cmp-eval-delta ${improved ? 'up' : delta === 0 ? 'flat' : 'down'}`}>
                  {delta === 0
                    ? 'no change'
                    : (improved ? '↑ ' : '↓ ') +
                      ('scale' in m && m.scale
                        ? Math.abs(delta).toFixed(2)
                        : lowerBetter
                        ? Math.abs(delta).toFixed(0) + ' ms'
                        : (Math.abs(delta) * 100).toFixed(1) + ' pp')}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
