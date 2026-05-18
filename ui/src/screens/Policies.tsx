/**
 * Policies — operator's trust dial.
 *
 * Per-kind overrides are honored by harness.approver.policy.decide() — the
 * approver looks up policy.mode_for(kind) before falling back to the global
 * mode. Auto-rollback values persist to YAML and are visible to the approver,
 * but the metric watcher that triggers them on production telemetry doesn't
 * exist yet.
 */

import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { ToggleGroup, ToggleGroupItem } from '../components/ui/toggle-group';
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

const ModeToggle = ({
  value,
  onChange,
}: {
  value: PolicyMode | string | undefined;
  onChange: (mode: PolicyMode) => void;
}) => (
  <ToggleGroup
    type="single"
    variant="outline"
    size="sm"
    value={value ?? ''}
    onValueChange={(v) => v && onChange(v as PolicyMode)}
  >
    <ToggleGroupItem value="auto">Auto</ToggleGroupItem>
    <ToggleGroupItem value="review">Review</ToggleGroupItem>
    <ToggleGroupItem value="off">Off</ToggleGroupItem>
  </ToggleGroup>
);

const SectionHeader = ({ children }: { children: React.ReactNode }) => (
  <div className="flex items-center justify-between border-b border-border px-4 py-3.5 text-[13px] font-semibold">
    {children}
  </div>
);

const Row = ({
  name,
  desc,
  hint,
  control,
  muted,
}: {
  name: string;
  desc: React.ReactNode;
  hint?: React.ReactNode;
  control: React.ReactNode;
  muted?: boolean;
}) => (
  <div
    className={`grid grid-cols-[1fr_auto_auto] items-center gap-4 px-4 py-3.5 border-b border-border last:border-b-0 ${
      muted ? 'bg-muted' : ''
    }`}
  >
    <div className="min-w-0">
      <div className="text-[13.5px] font-medium">{name}</div>
      <div className="dim text-xs leading-snug">{desc}</div>
    </div>
    {hint && <div className="dim text-xs">{hint}</div>}
    <div>{control}</div>
  </div>
);

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

  const dirty = !!policy && !!draft && JSON.stringify(policy) !== JSON.stringify(draft);

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
        <p className="page-sub text-destructive">{error || 'Policy unavailable.'}</p>
        <Button onClick={load}>Retry</Button>
      </div>
    );
  }

  const setKindMode = (kind: string, mode: PolicyMode) => {
    setDraft((d) => (d ? { ...d, overrides: { ...d.overrides, [kind]: mode } } : d));
  };
  const clearKindMode = (kind: string) => {
    setDraft((d) => {
      if (!d) return d;
      const next = { ...d.overrides };
      delete next[kind];
      return { ...d, overrides: next };
    });
  };
  const effectiveMode = (kind: string): string => draft.overrides[kind] || draft.mode;

  const autoLiftPct = (draft.auto_min_lift * 100).toFixed(1);

  return (
    <div className="content">
      <h1 className="page-title">Policies</h1>
      <p className="page-sub">
        Set how the agent decides what to ship without you. Anything set to auto-promote will roll
        back automatically if metrics regress.
      </p>

      {error && (
        <Card className="mb-4 border-destructive p-4">
          <p className="dim m-0 text-destructive">{error}</p>
        </Card>
      )}

      <Card className="mb-6 gap-0 py-0">
        <SectionHeader>Approval mode by change type</SectionHeader>
        <Row
          name="Default for everything"
          desc="Falls through to this mode for any change kind without an override below"
          hint={modeDesc(draft.mode, autoLiftPct)}
          control={
            <ModeToggle
              value={draft.mode}
              onChange={(m) => setDraft({ ...draft, mode: m })}
            />
          }
        />
        {KIND_ROWS.map((r) => {
          const overrideMode = draft.overrides[r.key];
          const eff = effectiveMode(r.key);
          const usingOverride = !!overrideMode;
          return (
            <Row
              key={r.key}
              name={r.name}
              desc={
                <>
                  {r.desc}
                  {!usingOverride && (
                    <span className="dim ml-2 text-[11px]">(using default)</span>
                  )}
                </>
              }
              hint={modeDesc(eff, autoLiftPct)}
              control={
                <div className="flex items-center gap-1">
                  <ModeToggle
                    value={overrideMode}
                    onChange={(m) => setKindMode(r.key, m)}
                  />
                  {usingOverride && (
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Clear override"
                      onClick={() => clearKindMode(r.key)}
                    >
                      ×
                    </Button>
                  )}
                </div>
              }
            />
          );
        })}
        <Row
          muted
          name="Auto-promote threshold"
          desc={
            <>
              Minimum Δoverall on the eval suite before <span className="mono">auto</span>{' '}
              promotes. Only used in auto mode.
            </>
          }
          hint={
            <span className="mono">
              auto_min_lift: {draft.auto_min_lift.toFixed(4)} ({autoLiftPct} pp)
            </span>
          }
          control={
            <Input
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
              className="mono w-[100px]"
            />
          }
        />
      </Card>

      <Card className="mb-6 gap-0 py-0">
        <SectionHeader>
          <span>Auto-rollback</span>
          <Badge
            variant="neutral"
            title="Values persist to YAML; metric-trigger fires once production telemetry is wired."
          >
            persisted; not yet firing
          </Badge>
        </SectionHeader>
        <Row
          name="Trigger threshold"
          desc="How much regression before the agent rolls itself back"
          hint={
            editingThreshold ? (
              <div className="mono flex items-center gap-1.5 text-xs">
                CSAT drop ≥
                <Input
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
                  className="h-7 w-[64px] px-2 text-xs"
                />
                within
                <Input
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
                  className="h-7 w-[60px] px-2 text-xs"
                />
                h, OR resolution drop ≥
                <Input
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
                        resolution_drop: Math.max(0, Math.min(1, Number(e.target.value) || 0)),
                      },
                    })
                  }
                  className="h-7 w-[64px] px-2 text-xs"
                />
              </div>
            ) : (
              <span className="mono">
                CSAT drop ≥ {draft.auto_rollback.csat_drop} within {draft.auto_rollback.window_hours}
                h, OR resolution rate drop ≥ {(draft.auto_rollback.resolution_drop * 100).toFixed(0)}%
              </span>
            )
          }
          control={
            <Button variant="outline" size="sm" onClick={() => setEditingThreshold((v) => !v)}>
              {editingThreshold ? 'Done' : 'Edit'}
            </Button>
          }
        />
        <Row
          name="Notify on rollback"
          desc="Where the agent posts a heads-up"
          hint={
            editingNotify ? (
              <Input
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
                className="w-[260px]"
              />
            ) : draft.auto_rollback.notify_channels.length === 0 ? (
              '(none)'
            ) : (
              draft.auto_rollback.notify_channels.join(' + ')
            )
          }
          control={
            <Button variant="outline" size="sm" onClick={() => setEditingNotify((v) => !v)}>
              {editingNotify ? 'Done' : 'Edit'}
            </Button>
          }
        />
      </Card>

      <Card className="mb-6 flex-row items-start gap-3.5 px-4 py-4">
        <div
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md"
          style={{ background: 'var(--info-soft)', color: 'var(--info-fg)' }}
        >
          <Icon name="info" size={16} />
        </div>
        <div>
          <div className="mb-1 text-[13.5px] font-medium">How auto-promote works</div>
          <div className="dim text-[13px] leading-relaxed">
            The agent runs each candidate change against your eval set offline before shipping.
            Auto mode promotes only when projected lift exceeds the threshold and no critic blocks.
            Per-kind overrides win over the global default. Production traffic ramps gradually
            once it's on. Rollback is automatic if any auto-rollback rule fires.
          </div>
        </div>
      </Card>

      <div
        className="sticky bottom-4 flex items-center gap-2.5 rounded-[var(--radius)] border border-border bg-card px-4 py-3 shadow-lg"
      >
        <Button onClick={save} disabled={!dirty || saving}>
          <Icon name="check" size={14} /> {saving ? 'Saving…' : 'Save changes'}
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            if (policy) setDraft(policy);
            setEditingThreshold(false);
            setEditingNotify(false);
          }}
          disabled={!dirty || saving}
        >
          Reset
        </Button>
        {dirty && <span className="dim text-xs">Unsaved changes</span>}
        {!dirty && savedAt && !saving && <span className="dim text-xs">Saved.</span>}
      </div>
    </div>
  );
};
