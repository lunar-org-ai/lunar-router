import { useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag, StatusTag, KindIcon, KindLabel } from '../components/Tag';
import { ConfBar } from '../components/ConfBar';
import { Diff } from '../components/Diff';
import { lessons, fmtDay, type Eval } from '../data';

const triggerLabel = (kind: string) =>
  kind === 'failed_traces'
    ? 'Failed traces in production'
    : kind === 'feedback'
    ? 'Customer feedback signals'
    : kind === 'cost'
    ? 'Cost optimization scan'
    : kind === 'experiment'
    ? 'A/B experiment regression'
    : kind;

const fileLabelFor = (type: string) =>
  type === 'system_prompt'
    ? 'system_prompt.md'
    : type === 'tool_wrapper'
    ? 'tools/lookup_order.py'
    : type === 'routing'
    ? 'routing.yml'
    : type === 'eval'
    ? 'evals/'
    : 'config.yml';

const fmtEval = (v: number | null, e: Eval) =>
  v == null
    ? '—'
    : e.scale
    ? v.toFixed(1)
    : e.currency
    ? `$${v.toFixed(3)}`
    : v < 1 && v > 0
    ? `${(v * 100).toFixed(0)}%`
    : v.toFixed(0);

export const LessonDetail = ({ lessonId, onBack }: { lessonId: string; onBack: () => void }) => {
  const l = lessons.find((x) => x.id === lessonId);
  const [tab, setTab] = useState<'story' | 'traces' | 'evals' | 'diff' | 'decision'>('story');
  if (!l) return null;

  const tabs = [
    { id: 'story', label: 'Story' },
    { id: 'traces', label: 'Traces', count: l.traces.length },
    { id: 'evals', label: 'Evals', count: l.evals.length },
    { id: 'diff', label: 'Diff' },
    { id: 'decision', label: 'Decision' },
  ] as const;

  return (
    <div className="content">
      <button className="btn ghost sm" onClick={onBack} style={{ marginBottom: 14, marginLeft: -8 }}>
        <Icon name="chevron" size={12} style={{ transform: 'rotate(180deg)' }} /> Back to evolution
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
        <Tag>
          <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
        </Tag>
        <span className="mono dim" style={{ fontSize: 12 }}>
          {l.id}
        </span>
        <span className="dim" style={{ fontSize: 12 }}>
          ·
        </span>
        <span className="mono dim" style={{ fontSize: 12 }}>
          {l.version}
        </span>
        <span className="dim" style={{ fontSize: 12 }}>
          ·
        </span>
        <span className="dim" style={{ fontSize: 12 }}>
          {fmtDay(l.date)}
        </span>
        <div style={{ marginLeft: 'auto' }}>
          <StatusTag status={l.status === 'approved' ? l.decision : l.status} />
        </div>
      </div>

      <h1 className="page-title">{l.title}</h1>
      <p className="page-sub">{l.summary}</p>

      <div className="tabs">
        {tabs.map((t) => (
          <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
            {t.label}
            {'count' in t && t.count != null && <span className="count">{t.count}</span>}
          </button>
        ))}
      </div>

      {tab === 'story' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 32 }}>
          <div>
            <div className="msg msg-agent" style={{ marginBottom: 24 }}>
              <div className="msg-role">In my own words</div>
              <p className="msg-body" style={{ fontStyle: 'italic', fontSize: 16, lineHeight: 1.6 }}>
                {l.voice}”
              </p>
            </div>

            <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px' }}>Why I made this change</h3>
            <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--fg)', margin: '0 0 24px' }}>{l.reasoning}</p>

            <h3 style={{ fontSize: 14, fontWeight: 600, margin: '24px 0 10px' }}>What triggered it</h3>
            <div className="card card-pad">
              <div className="kv">
                <div className="k">Source</div>
                <div>{triggerLabel(l.trigger.kind)}</div>
                {l.trigger.count != null && (
                  <>
                    <div className="k">Volume</div>
                    <div>{l.trigger.count} signals</div>
                  </>
                )}
                <div className="k">Window</div>
                <div>{l.trigger.window}</div>
                <div className="k">Confidence</div>
                <div>
                  <ConfBar value={l.confidence} />{' '}
                  <span className="dim" style={{ marginLeft: 8, fontSize: 12 }}>
                    {l.confidence}/5
                  </span>
                </div>
              </div>
            </div>
          </div>
          <aside style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card card-pad">
              <div
                className="dim"
                style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10 }}
              >
                Impact
              </div>
              {l.metrics.length === 0 && (
                <div className="dim" style={{ fontSize: 13 }}>
                  No production data yet.
                </div>
              )}
              {l.metrics.map((m) => (
                <div
                  key={m.label}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '8px 0',
                    borderBottom: '1px solid var(--border)',
                    fontSize: 13,
                  }}
                >
                  <span>{m.label}</span>
                  <span
                    className={m.delta > 0 ? '' : 'mono'}
                    style={{ color: m.delta > 0 ? 'var(--accent-fg)' : 'var(--bad-fg)', fontWeight: 500 }}
                  >
                    {m.delta > 0 ? '+' : ''}
                    {m.delta}
                    {Math.abs(m.delta) > 5 ? '%' : ''}
                  </span>
                </div>
              ))}
            </div>
            {l.status === 'pending' && (
              <div className="card card-pad" style={{ background: 'var(--warn-soft)', borderColor: 'transparent' }}>
                <div style={{ fontSize: 12.5, color: 'var(--warn-fg)', fontWeight: 500, marginBottom: 8 }}>
                  Awaiting your review
                </div>
                <div style={{ fontSize: 12.5, color: 'var(--fg-muted)', marginBottom: 12 }}>
                  Will auto-approve in 22h if untouched.
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn success sm" style={{ flex: 1, justifyContent: 'center' }}>
                    <Icon name="check" size={12} /> Approve
                  </button>
                  <button className="btn danger sm" style={{ flex: 1, justifyContent: 'center' }}>
                    <Icon name="x" size={12} /> Reject
                  </button>
                </div>
              </div>
            )}
          </aside>
        </div>
      )}

      {tab === 'traces' && (
        <div className="card">
          <div
            style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border)',
              fontSize: 12,
              color: 'var(--fg-muted)',
              display: 'grid',
              gridTemplateColumns: '80px 1fr 100px 80px 24px',
              gap: 16,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontWeight: 500,
            }}
          >
            <div>Trace</div>
            <div>Excerpt</div>
            <div>Verdict</div>
            <div>Used as</div>
            <div></div>
          </div>
          {l.traces.length === 0 && <div className="empty">No traces linked to this lesson.</div>}
          {l.traces.map((t) => (
            <div className="trace-row" key={t.id}>
              <span className="id">{t.id}</span>
              <span className="preview">{t.preview}</span>
              <span>
                <Tag kind={t.verdict === 'pass' ? 'success' : 'bad'}>
                  {t.verdict === 'pass' ? 'Passed' : 'Failed'}
                </Tag>
              </span>
              <span className="dim" style={{ fontSize: 12 }}>
                {t.verdict === 'fail' ? 'Trigger' : 'Validation'}
              </span>
              <Icon name="chevron" size={12} />
            </div>
          ))}
        </div>
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
              gridTemplateColumns: '1fr 80px 80px 90px',
              gap: 12,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              fontWeight: 500,
            }}
          >
            <div>Eval</div>
            <div>Before</div>
            <div>After</div>
            <div>Δ</div>
          </div>
          {l.evals.map((e, i) => {
            const better = e.isLatency ? e.after < (e.before ?? Infinity) : e.after > (e.before ?? -Infinity);
            const delta = e.before == null ? null : ((e.after - e.before) / Math.abs(e.before)) * 100;
            const pct = e.scale ? e.after / e.scale : Math.min(1, e.after);
            return (
              <div className="eval-row" key={i}>
                <div>
                  <div style={{ fontWeight: 500 }}>{e.name}</div>
                  <div className="dim" style={{ fontSize: 11.5, marginTop: 2 }}>
                    n = {e.sample.toLocaleString()}
                  </div>
                  <div className="bar" style={{ marginTop: 8 }}>
                    <span style={{ width: `${pct * 100}%`, background: better ? 'var(--accent)' : 'var(--bad)' }} />
                  </div>
                </div>
                <span className="mono dim">{fmtEval(e.before, e)}</span>
                <span className="mono" style={{ fontWeight: 500 }}>
                  {fmtEval(e.after, e)}
                </span>
                <span
                  className="mono"
                  style={{
                    color: delta == null ? 'var(--fg-muted)' : better ? 'var(--accent-fg)' : 'var(--bad-fg)',
                    fontWeight: 500,
                  }}
                >
                  {delta == null ? 'new' : `${delta > 0 ? '+' : ''}${delta.toFixed(1)}%`}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {tab === 'diff' && (
        <div>
          <div className="dim" style={{ fontSize: 12.5, marginBottom: 12 }}>
            Showing the exact change applied to <span className="mono">{fileLabelFor(l.proposal.type)}</span>
          </div>
          <Diff before={l.proposal.beforeLines} after={l.proposal.afterLines} fileLabel={fileLabelFor(l.proposal.type)} />
        </div>
      )}

      {tab === 'decision' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card card-pad">
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>How the system decided</div>
            <ol style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, lineHeight: 1.7 }}>
              <li>
                Detected {l.trigger.count || '—'} signals matching pattern{' '}
                <span className="mono dim">{l.trigger.kind}</span>
              </li>
              <li>
                Generated {l.proposal.type === 'eval' ? 'a new eval suite' : '3 candidate proposals'}; picked the one
                with highest projected eval lift
              </li>
              <li>
                Ran offline against {l.evals[0]?.sample.toLocaleString() || '—'} samples; saw{' '}
                {l.evals[0]
                  ? `${((l.evals[0].after - (l.evals[0].before || 0)) * 100).toFixed(0)} pp improvement`
                  : 'positive results'}
              </li>
              <li>
                Routed to{' '}
                <span className="mono">
                  {l.decision === 'auto_promoted'
                    ? 'auto-promote'
                    : l.decision === 'awaiting_review'
                    ? 'human review'
                    : 'rollback'}
                </span>{' '}
                per policy
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
                }}
              >
                <div className="dim">
                  if change_kind in [
                  {l.kind === 'router' || l.kind === 'prompt' ? 'prompt, router' : 'tool_wrapper, policy'}]
                </div>
                <div className="dim">and projected_lift &gt; 3pp</div>
                <div>
                  →{' '}
                  <span style={{ color: 'var(--accent-fg)' }}>
                    {l.decision === 'auto_promoted' ? 'auto_promote' : 'require_review'}
                  </span>
                </div>
              </div>
              <button className="btn sm ghost" style={{ alignSelf: 'flex-start' }}>
                <Icon name="settings" size={12} /> Edit policies
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
