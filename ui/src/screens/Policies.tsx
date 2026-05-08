/**
 * Policies — operator's trust dial.
 *
 * Layout matches OpenTracy Evolution design verbatim:
 *   - "Approval mode by change type" (5 per-kind rows + global)
 *   - "Auto-rollback" (trigger threshold + notify channels)
 *   - "How auto-promote works" info card
 *
 * Per-kind overrides are honored by harness.approver.policy.decide() — the
 * approver looks up policy.mode_for(kind) before falling back to the global
 * mode. Kinds in the design (prompt/router/tool/policy/eval) cover the
 * mock's surface; the harness today emits kinds {rag, rerank, router,
 * prompt, memory, other}, so anything emitted that isn't in the table
 * falls through to the global mode (safe default).
 *
 * Auto-rollback values persist to YAML and are visible to the approver,
 * but the metric watcher that triggers them on production telemetry doesn't
 * exist yet — that's tied to P1.9 (real LLM) + production telemetry hooks.
 */

import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import {
  ApiError,
  getPolicy,
  updatePolicy,
  type PolicyView,
  type PolicyMode,
} from '../api';

const KIND_ROWS: { key: string; name: string; desc: string }[] = [
  { key: 'prompt', name: 'Prompt edits', desc: 'Changes to the system prompt or instructions' },
  { key: 'router', name: 'Routing changes', desc: 'Which model handles which type of request' },
  { key: 'tool', name: 'Tool wrappers', desc: 'Pre/post-processing on tool calls' },
  { key: 'policy', name: 'Behavior policies', desc: 'Higher-level rules — escalation thresholds, refusal logic' },
  { key: 'eval', name: 'New evals', desc: 'Self-tests the agent writes for itself' },
];

const modeDesc = (mode: string, autoLiftPct: string): string => {
  if (mode === 'auto') return `Ship if Δoverall ≥ ${autoLiftPct} pp, no critic block`;
  if (mode === 'review') return 'Wait for your approval';
  return 'Never apply automatically';
};

export const Policies = () => {
  const [policy, setPolicy] = useState<PolicyView | null>(null);
  const [draft, setDraft] = useState<PolicyView | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [editingThreshold, setEditingThreshold] = useState(false);
  const [editingNotify, setEditingNotify] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const p = await getPolicy();
      setPolicy(p);
      setDraft(p);
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

  const dirty =
    !!policy && !!draft && JSON.stringify(policy) !== JSON.stringify(draft);

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await updatePolicy({
        mode: draft.mode,
        auto_min_lift: draft.auto_min_lift,
        overrides: draft.overrides,
        auto_rollback: draft.auto_rollback,
      });
      setPolicy(updated);
      setDraft(updated);
      setSavedAt(Date.now());
      setEditingThreshold(false);
      setEditingNotify(false);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Save failed: ${e.status} — ${e.message}`
          : `Save failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setSaving(false);
    }
  };

  if (loading && !policy) {
    return (
      <div className="content">
        <h1 className="page-title">Policies</h1>
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  if (!policy || !draft) {
    return (
      <div className="content">
        <h1 className="page-title">Policies</h1>
        <p className="page-sub" style={{ color: 'var(--bad)' }}>
          {error || 'Policy unavailable.'}
        </p>
        <button className="btn" onClick={load}>Retry</button>
      </div>
    );
  }

  const setKindMode = (kind: string, mode: PolicyMode) => {
    setDraft((d) =>
      d ? { ...d, overrides: { ...d.overrides, [kind]: mode } } : d,
    );
  };
  const effectiveMode = (kind: string): string =>
    draft.overrides[kind] || draft.mode;

  const autoLiftPct = (draft.auto_min_lift * 100).toFixed(1);

  return (
    <div className="content">
      <h1 className="page-title">Policies</h1>
      <p className="page-sub">
        Set how the agent decides what to ship without you. Anything set to auto-promote will roll
        back automatically if metrics regress.
      </p>

      {error && (
        <div className="card card-pad" style={{ borderColor: 'var(--bad)', marginBottom: 16 }}>
          <p className="dim" style={{ color: 'var(--bad)', margin: 0 }}>{error}</p>
        </div>
      )}

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
        <div className="policy-row">
          <div>
            <div className="pname">Default for everything</div>
            <div className="pdesc">
              Falls through to this mode for any change kind without an override below
            </div>
          </div>
          <div className="dim" style={{ fontSize: 12.5 }}>{modeDesc(draft.mode, autoLiftPct)}</div>
          <div className="toggle">
            {(['auto', 'review', 'off'] as const).map((m) => (
              <button
                key={m}
                className={draft.mode === m ? 'on' : ''}
                onClick={() => setDraft({ ...draft, mode: m })}
              >
                {m === 'auto' ? 'Auto' : m === 'review' ? 'Review' : 'Off'}
              </button>
            ))}
          </div>
        </div>
        {KIND_ROWS.map((r) => {
          const overrideMode = draft.overrides[r.key];
          const eff = effectiveMode(r.key);
          const usingOverride = !!overrideMode;
          return (
            <div className="policy-row" key={r.key}>
              <div>
                <div className="pname">{r.name}</div>
                <div className="pdesc">
                  {r.desc}
                  {!usingOverride && (
                    <span className="dim" style={{ marginLeft: 8, fontSize: 11 }}>
                      (using default)
                    </span>
                  )}
                </div>
              </div>
              <div className="dim" style={{ fontSize: 12.5 }}>{modeDesc(eff, autoLiftPct)}</div>
              <div className="toggle">
                {(['auto', 'review', 'off'] as const).map((m) => (
                  <button
                    key={m}
                    className={overrideMode === m ? 'on' : ''}
                    onClick={() => setKindMode(r.key, m)}
                  >
                    {m === 'auto' ? 'Auto' : m === 'review' ? 'Review' : 'Off'}
                  </button>
                ))}
                {usingOverride && (
                  <button
                    title="Clear override"
                    onClick={() =>
                      setDraft((d) => {
                        if (!d) return d;
                        const next = { ...d.overrides };
                        delete next[r.key];
                        return { ...d, overrides: next };
                      })
                    }
                    style={{
                      background: 'transparent',
                      color: 'var(--fg-muted)',
                      padding: '0 8px',
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          );
        })}
        <div className="policy-row" style={{ background: 'var(--bg-muted)' }}>
          <div>
            <div className="pname">Auto-promote threshold</div>
            <div className="pdesc">
              Minimum Δoverall on the eval suite before <span className="mono">auto</span>{' '}
              promotes. Only used in auto mode.
            </div>
          </div>
          <div className="dim mono" style={{ fontSize: 12.5 }}>
            auto_min_lift: {draft.auto_min_lift.toFixed(4)} ({autoLiftPct} pp)
          </div>
          <input
            type="number"
            step="0.001"
            min="0"
            max="1"
            value={draft.auto_min_lift}
            onChange={(e) =>
              setDraft({
                ...draft,
                auto_min_lift: Math.max(0, Math.min(1, Number(e.target.value) || 0)),
              })
            }
            style={{
              width: 100,
              padding: '6px 10px',
              border: '1px solid var(--border)',
              borderRadius: 6,
              background: 'var(--bg)',
              color: 'var(--fg)',
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
            }}
          />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div
          style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border)',
            fontSize: 13,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span>Auto-rollback</span>
          <span
            className="tag"
            style={{
              background: 'var(--bg-muted)',
              color: 'var(--fg-muted)',
              fontSize: 11,
            }}
            title="Values persist to YAML; metric-trigger fires once production telemetry is wired."
          >
            persisted; not yet firing
          </span>
        </div>
        <div className="policy-row">
          <div>
            <div className="pname">Trigger threshold</div>
            <div className="pdesc">How much regression before the agent rolls itself back</div>
          </div>
          {editingThreshold ? (
            <div
              style={{
                display: 'flex',
                gap: 6,
                alignItems: 'center',
                fontSize: 12.5,
                fontFamily: 'var(--font-mono)',
              }}
            >
              CSAT drop ≥
              <input
                type="number"
                step="0.05"
                min="0"
                max="5"
                value={draft.auto_rollback.csat_drop}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    auto_rollback: {
                      ...draft.auto_rollback,
                      csat_drop: Math.max(0, Math.min(5, Number(e.target.value) || 0)),
                    },
                  })
                }
                style={{
                  width: 60,
                  padding: '3px 6px',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  background: 'var(--bg)',
                  fontFamily: 'inherit',
                  fontSize: 12.5,
                }}
              />
              within
              <input
                type="number"
                min="1"
                max="720"
                value={draft.auto_rollback.window_hours}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    auto_rollback: {
                      ...draft.auto_rollback,
                      window_hours: Math.max(1, Math.min(720, Number(e.target.value) || 24)),
                    },
                  })
                }
                style={{
                  width: 56,
                  padding: '3px 6px',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  background: 'var(--bg)',
                  fontFamily: 'inherit',
                  fontSize: 12.5,
                }}
              />
              h, OR resolution drop ≥
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={draft.auto_rollback.resolution_drop}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    auto_rollback: {
                      ...draft.auto_rollback,
                      resolution_drop: Math.max(
                        0,
                        Math.min(1, Number(e.target.value) || 0),
                      ),
                    },
                  })
                }
                style={{
                  width: 60,
                  padding: '3px 6px',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  background: 'var(--bg)',
                  fontFamily: 'inherit',
                  fontSize: 12.5,
                }}
              />
            </div>
          ) : (
            <div className="dim mono" style={{ fontSize: 12.5 }}>
              CSAT drop ≥ {draft.auto_rollback.csat_drop} within {draft.auto_rollback.window_hours}
              h, OR resolution rate drop ≥ {(draft.auto_rollback.resolution_drop * 100).toFixed(0)}%
            </div>
          )}
          <button className="btn sm" onClick={() => setEditingThreshold((v) => !v)}>
            {editingThreshold ? 'Done' : 'Edit'}
          </button>
        </div>
        <div className="policy-row">
          <div>
            <div className="pname">Notify on rollback</div>
            <div className="pdesc">Where the agent posts a heads-up</div>
          </div>
          {editingNotify ? (
            <input
              type="text"
              value={draft.auto_rollback.notify_channels.join(', ')}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  auto_rollback: {
                    ...draft.auto_rollback,
                    notify_channels: e.target.value
                      .split(',')
                      .map((s) => s.trim())
                      .filter(Boolean),
                  },
                })
              }
              placeholder="email, slack:#agent-evolution"
              style={{
                width: 260,
                padding: '6px 10px',
                border: '1px solid var(--border)',
                borderRadius: 6,
                background: 'var(--bg)',
                fontSize: 12.5,
              }}
            />
          ) : (
            <div className="dim" style={{ fontSize: 12.5 }}>
              {draft.auto_rollback.notify_channels.length === 0
                ? '(none)'
                : draft.auto_rollback.notify_channels.join(' + ')}
            </div>
          )}
          <button className="btn sm" onClick={() => setEditingNotify((v) => !v)}>
            {editingNotify ? 'Done' : 'Edit'}
          </button>
        </div>
      </div>

      <div
        className="card card-pad"
        style={{ display: 'flex', gap: 14, alignItems: 'flex-start', marginBottom: 24 }}
      >
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
          <div style={{ fontWeight: 500, marginBottom: 4, fontSize: 13.5 }}>
            How auto-promote works
          </div>
          <div className="dim" style={{ fontSize: 13, lineHeight: 1.6 }}>
            The agent runs each candidate change against your eval set offline before shipping.
            Auto mode promotes only when projected lift exceeds the threshold and no critic blocks.
            Per-kind overrides win over the global default. Production traffic ramps gradually
            once it's on. Rollback is automatic if any auto-rollback rule fires.
          </div>
        </div>
      </div>

      <div
        style={{
          position: 'sticky',
          bottom: 16,
          display: 'flex',
          gap: 10,
          alignItems: 'center',
          padding: '12px 16px',
          background: 'var(--bg-elev)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          boxShadow: '0 4px 16px rgb(0 0 0 / 8%)',
        }}
      >
        <button className="btn primary" onClick={save} disabled={!dirty || saving}>
          <Icon name="check" size={14} /> {saving ? 'Saving…' : 'Save changes'}
        </button>
        <button
          className="btn ghost"
          onClick={() => {
            if (policy) setDraft(policy);
            setEditingThreshold(false);
            setEditingNotify(false);
          }}
          disabled={!dirty || saving}
        >
          Reset
        </button>
        {dirty && (
          <span className="dim" style={{ fontSize: 12.5 }}>Unsaved changes</span>
        )}
        {!dirty && savedAt && !saving && (
          <span className="dim" style={{ fontSize: 12.5 }}>Saved.</span>
        )}
      </div>
    </div>
  );
};
