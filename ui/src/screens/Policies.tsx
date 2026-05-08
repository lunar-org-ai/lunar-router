/**
 * Policies — the operator's trust dial.
 *
 * AutoHarness paper (arxiv 2603.03329) frames self-improving systems as a
 * loop of: propose → eval → promote / hold. Policies is the gate between
 * "eval said this is better" and "ship it without me". Three modes
 * (auto/review/off) plus a minimum-lift threshold give the operator a
 * single coarse dial to bias the system toward speed or caution.
 *
 * Per-kind overrides (e.g. auto for prompt edits, review for tool wrappers)
 * are flagged in the YAML as future work — the harness approver doesn't
 * read them yet, so we don't pretend they work in the UI.
 */

import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { ApiError, getPolicy, updatePolicy, type PolicyView } from '../api';

type Mode = 'auto' | 'review' | 'off';

const MODE_DESC: Record<Mode, string> = {
  auto: 'Ship if Δoverall meets threshold and no critic blocks.',
  review: 'Always wait for your approval — every promotion lands in Review.',
  off: 'Freeze promotions entirely. Candidates run for evals only.',
};

export const Policies = () => {
  const [policy, setPolicy] = useState<PolicyView | null>(null);
  const [draft, setDraft] = useState<PolicyView | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

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
    !!draft &&
    !!policy &&
    (draft.mode !== policy.mode || draft.auto_min_lift !== policy.auto_min_lift);

  const onSave = async () => {
    if (!draft) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await updatePolicy({
        mode: draft.mode,
        auto_min_lift: draft.auto_min_lift,
      });
      setPolicy(updated);
      setDraft(updated);
      setSavedAt(Date.now());
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

  const onReset = () => {
    if (policy) setDraft(policy);
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
        <p className="page-sub" style={{ color: 'var(--bad)' }}>{error || 'Policy unavailable.'}</p>
        <button className="btn" onClick={load}>Retry</button>
      </div>
    );
  }

  const liftPct = (draft.auto_min_lift * 100).toFixed(1);

  return (
    <div className="content">
      <h1 className="page-title">Policies</h1>
      <p className="page-sub">
        Set how the agent decides what to ship without you. Edits write to{' '}
        <span className="mono">policies/auto_approve.yaml</span> and the approver picks them up on
        the next harness loop.
      </p>

      {error && (
        <div className="card card-pad" style={{ borderColor: 'var(--bad)', marginBottom: 16 }}>
          <p className="dim" style={{ color: 'var(--bad)', margin: 0 }}>
            {error}
          </p>
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
          Approval mode
        </div>
        <div className="policy-row">
          <div>
            <div className="pname">Default for every change</div>
            <div className="pdesc">{MODE_DESC[draft.mode as Mode] || '—'}</div>
          </div>
          <div className="dim mono" style={{ fontSize: 12.5 }}>
            mode: {draft.mode}
          </div>
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
        <div className="policy-row">
          <div>
            <div className="pname">Auto-promote threshold</div>
            <div className="pdesc">
              Minimum Δoverall on the eval suite before auto-promote fires (only used in mode=auto).
            </div>
          </div>
          <div className="dim mono" style={{ fontSize: 12.5 }}>
            auto_min_lift: {draft.auto_min_lift.toFixed(4)} ({liftPct} pp)
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
            disabled={draft.mode !== 'auto'}
            style={{
              width: 100,
              padding: '6px 10px',
              border: '1px solid var(--border)',
              borderRadius: 6,
              background: 'var(--bg)',
              color: 'var(--fg)',
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
              opacity: draft.mode === 'auto' ? 1 : 0.5,
            }}
          />
        </div>
        <div
          style={{
            padding: '12px 16px',
            borderTop: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <button
            className="btn primary"
            onClick={onSave}
            disabled={!dirty || saving}
          >
            <Icon name="check" size={14} /> {saving ? 'Saving…' : 'Save'}
          </button>
          <button className="btn ghost" onClick={onReset} disabled={!dirty || saving}>
            Reset
          </button>
          {savedAt && !dirty && !saving && (
            <span className="dim" style={{ fontSize: 12.5 }}>
              Saved.
            </span>
          )}
          {dirty && (
            <span className="dim" style={{ fontSize: 12.5 }}>
              Unsaved changes
            </span>
          )}
        </div>
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
          Per-kind overrides
        </div>
        <div className="policy-row">
          <div>
            <div className="pname">Different mode for different change kinds</div>
            <div className="pdesc">
              e.g. auto for prompt edits, review for tool wrappers. Listed in the YAML as future
              work — the approver doesn't honor overrides yet.
            </div>
          </div>
          <span
            className="tag"
            style={{ background: 'var(--bg-muted)', color: 'var(--fg-muted)', fontSize: 11.5 }}
          >
            Coming soon
          </span>
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
          <div style={{ fontWeight: 500, marginBottom: 4, fontSize: 13.5 }}>
            How the trust dial works
          </div>
          <div className="dim" style={{ fontSize: 13, lineHeight: 1.6 }}>
            Every candidate runs through the eval suite before reaching the approver. In{' '}
            <span className="mono">auto</span> mode, the approver promotes if Δoverall meets the
            threshold and every critic passed. In <span className="mono">review</span> mode each
            candidate lands in the Review queue regardless. In{' '}
            <span className="mono">off</span> mode candidates run for telemetry only — nothing is
            promoted. Auto-rollback on regression is independent of this dial.
          </div>
        </div>
      </div>
    </div>
  );
};
