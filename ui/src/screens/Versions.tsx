import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag } from '../components/Tag';
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
        <p className="page-sub" style={{ color: 'var(--bad)' }}>
          {error}
        </p>
        <button className="btn" onClick={refresh}>
          Retry
        </button>
      </div>
    );
  }

  if (versions.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Versions</h1>
        <p className="page-sub">No versions yet. Promote a candidate via the harness to create the first one.</p>
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
        <div className="card card-pad" style={{ borderColor: 'var(--bad)', marginBottom: 16 }}>
          <p className="dim" style={{ color: 'var(--bad)', margin: 0 }}>
            {error}
          </p>
        </div>
      )}

      <div className="versions">
        <div className="version-list">
          {versions.map((v) => (
            <button
              key={v.id}
              className={`version-item ${v.id === active.id ? 'active' : ''}`}
              onClick={() => setActiveId(v.id)}
              style={{ border: 'none', width: '100%' }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="vname">{v.id}</div>
                <div
                  className="vlabel"
                  style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                >
                  {v.lesson?.title ?? '(no lesson)'}
                </div>
              </div>
              <div className="vmeta">
                {v.is_live && <span className="live">LIVE</span>}
                {v.status === 'rolled_back' && (
                  <Tag kind="bad">
                    <span className="dot" />
                  </Tag>
                )}
              </div>
            </button>
          ))}
        </div>
        <div style={{ padding: 28, overflow: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span className="mono" style={{ fontSize: 18, fontWeight: 600 }}>
              {active.id}
            </span>
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
            {active.status === 'archived' && (
              <Tag kind="">
                <span className="dot" /> Archived
              </Tag>
            )}
            <span className="dim" style={{ marginLeft: 'auto', fontSize: 13 }}>
              Promoted {formatDate(active.promoted_at)}
            </span>
          </div>
          <div
            style={{ fontSize: 18, fontWeight: 500, letterSpacing: '-0.01em', marginBottom: 24 }}
          >
            {lesson?.title ?? '(no lesson attached to this version)'}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 24 }}>
            <div className="card card-pad">
              <div
                className="dim"
                style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}
              >
                Parent version
              </div>
              <div className="mono" style={{ fontSize: 14, fontWeight: 500 }}>
                {lesson?.parent_version ?? 'initial'}
              </div>
            </div>
            <div className="card card-pad">
              <div
                className="dim"
                style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}
              >
                {active.status === 'rolled_back' ? 'Rolled back' : 'Status'}
              </div>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                {active.status === 'rolled_back'
                  ? formatDate(active.rolled_back_at)
                  : active.is_live
                  ? 'Live'
                  : 'Archived'}
              </div>
            </div>
          </div>

          {lesson && (
            <>
              <div
                className="dim"
                style={{
                  fontSize: 11,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  fontWeight: 500,
                  marginBottom: 10,
                }}
              >
                Change in this version
              </div>
              <div className="card card-pad" style={{ marginBottom: 16 }}>
                {lesson.voice && (
                  <div style={{ fontStyle: 'italic', color: 'var(--fg)', fontSize: 14, lineHeight: 1.55 }}>
                    “{lesson.voice}”
                  </div>
                )}
                {!lesson.voice && (
                  <div style={{ color: 'var(--fg)', fontSize: 14, lineHeight: 1.55 }}>
                    {lesson.summary}
                  </div>
                )}
                {lesson.mutations.length > 0 && (
                  <div className="dim" style={{ fontSize: 12, marginTop: 12, fontFamily: 'var(--font-mono)' }}>
                    {lesson.mutations.map((m) => (
                      <div key={m}>{m}</div>
                    ))}
                  </div>
                )}
                {lesson.delta?.overall_score !== undefined && (
                  <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                    Δoverall: {lesson.delta.overall_score! >= 0 ? '+' : ''}
                    {lesson.delta.overall_score!.toFixed(4)}
                  </div>
                )}
              </div>
            </>
          )}

          <div style={{ display: 'flex', gap: 8, paddingTop: 20, borderTop: '1px solid var(--border)' }}>
            <button className="btn" onClick={() => setOverlay('compare')}>
              <Icon name="eye" size={14} /> Compare with live
            </button>
            <button className="btn" onClick={() => setOverlay('replay')}>
              <Icon name="play" size={14} /> Replay traces
            </button>
            {!active.is_live && (
              <button
                className="btn primary"
                style={{ marginLeft: 'auto' }}
                onClick={() => onRollback(active.id)}
                disabled={rollingBack}
              >
                <Icon name="rollback" size={14} /> {rollingBack ? 'Rolling back…' : `Roll back to ${active.id}`}
              </button>
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
