import { useState } from 'react';
import { Icon } from '../components/Icon';
import { Tag } from '../components/Tag';
import { versions, lessons, fmtDay } from '../data';

export const Versions = () => {
  const [activeId, setActiveId] = useState(versions.find((v) => v.live)?.id || versions[0].id);
  const active = versions.find((v) => v.id === activeId);
  const lesson = lessons.find((l) => l.version === activeId);
  if (!active) return null;

  return (
    <div className="content">
      <h1 className="page-title">Versions</h1>
      <p className="page-sub">Every version of your agent. Inspect, compare, or roll back to any point in history.</p>

      <div className="versions">
        <div className="version-list">
          {versions.map((v) => (
            <button
              key={v.id}
              className={`version-item ${v.id === activeId ? 'active' : ''}`}
              onClick={() => setActiveId(v.id)}
              style={{ border: 'none', width: '100%' }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="vname">{v.id}</div>
                <div
                  className="vlabel"
                  style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                >
                  {v.label}
                </div>
              </div>
              <div className="vmeta">
                {v.live && <span className="live">LIVE</span>}
                {v.status === 'rolled_back' && (
                  <Tag kind="bad">
                    <span className="dot" />
                  </Tag>
                )}
                {v.status === 'review' && (
                  <Tag kind="warn">
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
            {active.live && (
              <Tag kind="success">
                <span className="dot" /> Live now
              </Tag>
            )}
            {active.status === 'rolled_back' && (
              <Tag kind="bad">
                <span className="dot" /> Rolled back
              </Tag>
            )}
            {active.status === 'review' && (
              <Tag kind="warn">
                <span className="dot" /> In review
              </Tag>
            )}
            <span className="dim" style={{ marginLeft: 'auto', fontSize: 13 }}>
              Created {fmtDay(active.date)}
            </span>
          </div>
          <div style={{ fontSize: 18, fontWeight: 500, letterSpacing: '-0.01em', marginBottom: 24 }}>
            {active.label}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 24 }}>
            <div className="card card-pad">
              <div
                className="dim"
                style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}
              >
                Branched from
              </div>
              <div className="mono" style={{ fontSize: 14, fontWeight: 500 }}>
                {active.from || 'initial'}
              </div>
            </div>
            <div className="card card-pad">
              <div
                className="dim"
                style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}
              >
                Active in production
              </div>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                {active.live
                  ? '12 hours and counting'
                  : active.status === 'rolled_back'
                  ? '24 hours (then rolled back)'
                  : 'Never'}
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
              <div className="card card-pad" style={{ marginBottom: 24 }}>
                <div style={{ fontStyle: 'italic', color: 'var(--fg)', fontSize: 14, lineHeight: 1.55 }}>
                  {lesson.voice}”
                </div>
              </div>
            </>
          )}

          <div style={{ display: 'flex', gap: 8, paddingTop: 20, borderTop: '1px solid var(--border)' }}>
            <button className="btn">
              <Icon name="eye" size={14} /> Compare with live
            </button>
            <button className="btn">
              <Icon name="play" size={14} /> Replay traces against this
            </button>
            {!active.live && (
              <button className="btn primary" style={{ marginLeft: 'auto' }}>
                <Icon name="rollback" size={14} /> Roll back to {active.id}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
