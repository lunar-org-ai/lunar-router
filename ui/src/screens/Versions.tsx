import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag } from '../components/Tag';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { ApiError, listVersions, rollbackVersion, type VersionInfo } from '../api';
import { CompareWithLive } from './CompareWithLive';
import { ReplayTraces } from './ReplayTraces';

const formatDate = (iso: string | null): string => {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
};

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div className="dim mb-2.5 text-[11px] font-medium uppercase tracking-[0.05em]">{children}</div>
);

export const Versions = () => {
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rollingBack, setRollingBack] = useState(false);
  const [overlay, setOverlay] = useState<'compare' | 'replay' | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await listVersions();
      setVersions(items);
      setActiveId((current) => {
        if (current && items.some((v) => v.id === current)) return current;
        return items.find((v) => v.is_live)?.id ?? items[0]?.id ?? null;
      });
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
    void refresh();
  }, [refresh]);

  const onRollback = async (version: string) => {
    if (!confirm(`Roll back live agent to ${version}?`)) return;
    setRollingBack(true);
    try {
      await rollbackVersion(version, 'rollback from Versions UI');
      await refresh();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Rollback failed: ${e.status} — ${e.message}`
          : `Rollback failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setRollingBack(false);
    }
  };

  if (loading && versions.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Versions</h1>
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  if (error && versions.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Versions</h1>
        <p className="page-sub text-destructive">{error}</p>
        <Button onClick={refresh}>Retry</Button>
      </div>
    );
  }

  if (versions.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Versions</h1>
        <p className="page-sub">
          No versions yet. Promote a candidate via the harness to create the first one.
        </p>
      </div>
    );
  }

  const active = versions.find((v) => v.id === activeId) ?? versions[0];
  const lesson = active.lesson;

  return (
    <div className="content">
      <h1 className="page-title">Versions</h1>
      <p className="page-sub">
        Every version of your agent. Inspect, compare, or roll back to any point in history.
      </p>

      {error && (
        <Card className="mb-4 border-destructive p-4">
          <p className="dim m-0 text-destructive">{error}</p>
        </Card>
      )}

      <div className="grid min-h-[520px] grid-cols-[220px_1fr] overflow-hidden rounded-[var(--radius)] border border-border bg-card">
        <div className="overflow-y-auto border-r border-border bg-muted py-3">
          {versions.map((v) => {
            const isActive = v.id === active.id;
            return (
              <button
                key={v.id}
                onClick={() => setActiveId(v.id)}
                className={`flex w-full items-center gap-2.5 border-l-2 px-4 py-2.5 text-left text-[13px] transition-colors ${
                  isActive
                    ? 'border-l-foreground bg-card font-medium'
                    : 'border-l-transparent hover:bg-[var(--bg-sunken)]'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="mono text-xs">{v.id}</div>
                  <div className="truncate text-[11px] text-muted-foreground">
                    {v.lesson?.title ?? '(no lesson)'}
                  </div>
                </div>
                <div className="ml-auto flex items-center gap-1.5">
                  {v.is_live && (
                    <span className="rounded bg-primary px-1.5 py-px text-[10px] font-semibold tracking-wider text-primary-foreground">
                      LIVE
                    </span>
                  )}
                  {v.status === 'rolled_back' && (
                    <Tag kind="bad">
                      <span className="dot" />
                    </Tag>
                  )}
                </div>
              </button>
            );
          })}
        </div>

        <div className="overflow-auto p-7">
          <div className="mb-2 flex items-center gap-3">
            <span className="mono text-[18px] font-semibold">{active.id}</span>
            {active.is_live && (
              <Tag kind="success">
                <span className="dot" /> Live now
              </Tag>
            )}
            {active.status === 'rolled_back' && (
              <Tag kind="bad">
                <span className="dot" /> Rolled back
              </Tag>
            )}
            {active.status === 'archived' && <Tag>Archived</Tag>}
            <span className="dim ml-auto text-[13px]">
              Promoted {formatDate(active.promoted_at)}
            </span>
          </div>
          <div className="mb-6 text-[18px] font-medium leading-tight tracking-tight">
            {lesson?.title ?? '(no lesson attached to this version)'}
          </div>

          <div className="mb-6 grid grid-cols-2 gap-3">
            <Card className="px-4 py-3.5">
              <SectionLabel>Parent version</SectionLabel>
              <div className="mono text-[14px] font-medium">{lesson?.parent_version ?? 'initial'}</div>
            </Card>
            <Card className="px-4 py-3.5">
              <SectionLabel>{active.status === 'rolled_back' ? 'Rolled back' : 'Status'}</SectionLabel>
              <div className="text-[14px] font-medium">
                {active.status === 'rolled_back'
                  ? formatDate(active.rolled_back_at)
                  : active.is_live
                  ? 'Live'
                  : 'Archived'}
              </div>
            </Card>
          </div>

          {lesson && (
            <>
              <SectionLabel>Change in this version</SectionLabel>
              <Card className="mb-4 px-4 py-3.5">
                {lesson.voice ? (
                  <div className="text-[14px] italic leading-relaxed text-foreground">
                    "{lesson.voice}"
                  </div>
                ) : (
                  <div className="text-[14px] leading-relaxed text-foreground">{lesson.summary}</div>
                )}
                {lesson.mutations.length > 0 && (
                  <div className="dim mono mt-3 text-xs">
                    {lesson.mutations.map((m) => (
                      <div key={m}>{m}</div>
                    ))}
                  </div>
                )}
                {lesson.delta?.overall_score !== undefined && (
                  <div className="dim mt-2 text-xs">
                    Δoverall: {lesson.delta.overall_score! >= 0 ? '+' : ''}
                    {lesson.delta.overall_score!.toFixed(4)}
                  </div>
                )}
              </Card>
            </>
          )}

          <div className="flex items-center gap-2 border-t border-border pt-5">
            <Button variant="outline" onClick={() => setOverlay('compare')}>
              <Icon name="eye" size={14} /> Compare with live
            </Button>
            <Button variant="outline" onClick={() => setOverlay('replay')}>
              <Icon name="play" size={14} /> Replay traces
            </Button>
            {!active.is_live && (
              <Button
                className="ml-auto"
                onClick={() => onRollback(active.id)}
                disabled={rollingBack}
              >
                <Icon name="rollback" size={14} />{' '}
                {rollingBack ? 'Rolling back…' : `Roll back to ${active.id}`}
              </Button>
            )}
          </div>
        </div>
      </div>

      {overlay === 'compare' && (
        <CompareWithLive
          version={active}
          live={versions.find((v) => v.is_live) ?? active}
          onClose={() => setOverlay(null)}
        />
      )}
      {overlay === 'replay' && (
        <ReplayTraces
          version={active}
          live={versions.find((v) => v.is_live) ?? active}
          onClose={() => setOverlay(null)}
        />
      )}
    </div>
  );
};
