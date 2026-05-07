import { useState } from 'react';
import { Icon } from '../components/Icon';

type Mode = 'auto' | 'review' | 'off';
type PolicyKey = 'prompt' | 'router' | 'tool' | 'policy' | 'eval';

export const Policies = () => {
  const [policies, setPolicies] = useState<Record<PolicyKey, Mode>>({
    prompt: 'auto',
    router: 'auto',
    tool: 'review',
    policy: 'review',
    eval: 'auto',
  });
  const set = (k: PolicyKey, v: Mode) => setPolicies((p) => ({ ...p, [k]: v }));

  const rows: { key: PolicyKey; name: string; desc: string }[] = [
    { key: 'prompt', name: 'Prompt edits', desc: 'Changes to the system prompt or instructions' },
    { key: 'router', name: 'Routing changes', desc: 'Which model handles which type of request' },
    { key: 'tool', name: 'Tool wrappers', desc: 'Pre/post-processing on tool calls' },
    { key: 'policy', name: 'Behavior policies', desc: 'Higher-level rules — escalation thresholds, refusal logic' },
    { key: 'eval', name: 'New evals', desc: 'Self-tests the agent writes for itself' },
  ];

  return (
    <div className="content">
      <h1 className="page-title">Policies</h1>
      <p className="page-sub">
        Set how the agent decides what to ship without you. Anything set to auto-promote will roll back automatically
        if metrics regress.
      </p>

      <div className="card" style={{ marginBottom: 24 }}>
        <div
          style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border)',
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          Approval mode by change type
        </div>
        {rows.map((r) => (
          <div className="policy-row" key={r.key}>
            <div>
              <div className="pname">{r.name}</div>
              <div className="pdesc">{r.desc}</div>
            </div>
            <div className="dim" style={{ fontSize: 12.5 }}>
              {policies[r.key] === 'auto'
                ? 'Ship if eval lift > 3pp, no regression'
                : policies[r.key] === 'review'
                ? 'Wait for your approval'
                : 'Never apply automatically'}
            </div>
            <div className="toggle">
              <button className={policies[r.key] === 'auto' ? 'on' : ''} onClick={() => set(r.key, 'auto')}>
                Auto
              </button>
              <button className={policies[r.key] === 'review' ? 'on' : ''} onClick={() => set(r.key, 'review')}>
                Review
              </button>
              <button className={policies[r.key] === 'off' ? 'on' : ''} onClick={() => set(r.key, 'off')}>
                Off
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div
          style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border)',
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          Auto-rollback
        </div>
        <div className="policy-row">
          <div>
            <div className="pname">Trigger threshold</div>
            <div className="pdesc">How much regression before the agent rolls itself back</div>
          </div>
          <div className="dim mono" style={{ fontSize: 12.5 }}>
            CSAT drop ≥ 0.3 within 24h, OR resolution rate drop ≥ 5%
          </div>
          <button className="btn sm">Edit</button>
        </div>
        <div className="policy-row">
          <div>
            <div className="pname">Notify on rollback</div>
            <div className="pdesc">Where the agent posts a heads-up</div>
          </div>
          <div className="dim" style={{ fontSize: 12.5 }}>
            Email + Slack #agent-evolution
          </div>
          <button className="btn sm">Edit</button>
        </div>
      </div>

      <div className="card card-pad" style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'var(--info-soft)',
            color: 'var(--info-fg)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Icon name="info" size={16} />
        </div>
        <div>
          <div style={{ fontWeight: 500, marginBottom: 4, fontSize: 13.5 }}>How auto-promote works</div>
          <div className="dim" style={{ fontSize: 13, lineHeight: 1.6 }}>
            The agent runs each candidate change against your eval set offline before shipping. Auto mode promotes only
            when projected lift exceeds the threshold and no regression is detected. Production traffic ramps gradually
            over 6 hours. Rollback is automatic if any auto-rollback rule fires.
          </div>
        </div>
      </div>
    </div>
  );
};
