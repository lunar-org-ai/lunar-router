/**
 * Review — human-in-the-loop queue.
 *
 * Lists lessons with status="awaiting_review" (provisional lessons the harness
 * loop wrote when policy.mode=review). Approving runs promote_queued() on the
 * backend; rejecting writes a rejected entry. Either way the lesson is mutated
 * on disk and the timeline reflects it.
 */

import { useCallback, useEffect, useState } from 'react';
import { Link } from '@tanstack/react-router';
import { Icon } from '../components/Icon';
import { Tag, KindIcon, KindLabel } from '../components/Tag';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { preserveSearch } from '../router';
import {
  ApiError,
  approveLesson,
  listLessons,
  rejectLesson,
  requeueLesson,
  type LessonSummary,
} from '../api';

const fmtRubric = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(4)}`;

const DECIDED_STATUSES = new Set(['approved', 'auto_promoted', 'human_rejected']);

const decidedKind = (status: string): 'approved' | 'auto' | 'rejected' =>
  status === 'human_rejected' ? 'rejected' : status === 'auto_promoted' ? 'auto' : 'approved';

// Lesson IDs are L-YYYYMMDD-HHMMSS-xxxx — parse the timestamp prefix.
const fmtLessonTime = (id: string): string => {
  const m = id.match(/^L-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/);
  if (!m) return '';
  const [, y, mo, d, h, mi] = m;
  return `${y}-${mo}-${d} ${h}:${mi}`;
};

const HISTORY_LIMIT = 20;

const deltaColor = (v: number): string =>
  v > 0 ? 'var(--accent-fg)' : v < 0 ? 'var(--bad-fg)' : 'var(--muted-foreground)';

const SectionLabel = ({ children, hint }: { children: React.ReactNode; hint?: React.ReactNode }) => (
  <div className="dim mb-2 flex items-center gap-2 text-[12px] font-medium uppercase tracking-[0.05em]">
    <span>{children}</span>
    {hint && <span className="mono opacity-70">{hint}</span>}
  </div>
);

export const Review = () => {
  const [queue, setQueue] = useState<LessonSummary[]>([]);
  const [history, setHistory] = useState<LessonSummary[]>([]);
  const [decided, setDecided] = useState<
    { id: string; title: string; action: 'approve' | 'reject' }[]
  >([]);
  const [acting, setActing] = useState<Record<string, 'approve' | 'reject' | null>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const all = await listLessons();
      setQueue(all.filter((l) => l.status === 'awaiting_review'));
      setHistory(
        all
          .filter((l) => DECIDED_STATUSES.has(l.status))
          .sort((a, b) => (a.id < b.id ? 1 : a.id > b.id ? -1 : 0)),
      );
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

  const onAction = async (l: LessonSummary, action: 'approve' | 'reject') => {
    setActing((s) => ({ ...s, [l.id]: action }));
    try {
      if (action === 'approve') {
        await approveLesson(l.id, 'ui');
      } else {
        await rejectLesson(l.id);
      }
      setDecided((d) => [...d, { id: l.id, title: l.title, action }]);
      await refresh();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${action} failed: ${e.status} — ${e.message}`
          : `${action} failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setActing((s) => ({ ...s, [l.id]: null }));
    }
  };

  const requeue = async (id: string) => {
    setActing((s) => ({ ...s, [id]: 'approve' }));
    try {
      await requeueLesson(id);
      setDecided((d) => d.filter((x) => x.id !== id));
      await refresh();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Undo failed: ${e.status} — ${e.message}`
          : `Undo failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setActing((s) => ({ ...s, [id]: null }));
    }
  };

  if (loading && queue.length === 0 && decided.length === 0 && history.length === 0) {
    return (
      <div className="content">
        <h1 className="page-title">Pending review</h1>
        <p className="page-sub">Loading…</p>
      </div>
    );
  }

  return (
    <div className="content">
      <h1 className="page-title">Pending review</h1>
      <p className="page-sub">
        The agent has proposed changes based on what it learned. Approve to ship them, reject to
        discard, or wait — anything untouched in 24h follows your default policy.
      </p>

      {error && (
        <Card className="mb-4 border-destructive p-4">
          <p className="dim m-0 text-destructive">{error}</p>
        </Card>
      )}

      {queue.length === 0 && (
        <Card className="flex flex-col items-center justify-center gap-3 p-16 text-center">
          <Icon name="check" size={32} />
          <div className="text-base font-medium">Inbox zero.</div>
          <div className="dim">
            The agent will check in again when it has something new to suggest.
          </div>
        </Card>
      )}

      <div className="flex flex-col gap-4">
        {queue.map((l) => {
          const perRubric = l.delta?.per_rubric ?? {};
          const rubricKeys = Object.keys(perRubric);
          const overall = l.delta?.overall_score;
          const busy = acting[l.id];

          return (
            <Card key={l.id} className="gap-0 py-0">
              <div className="flex items-center gap-2 border-b border-border px-4 py-3">
                <Tag>
                  <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
                </Tag>
                <span className="text-[14px] font-medium">{l.title}</span>
                <span className="dim mono ml-auto text-[11.5px]">{l.id}</span>
              </div>
              <div className="grid grid-cols-[1fr_320px] gap-6 px-4 py-4">
                <div>
                  {l.voice && (
                    <div className="mb-3 italic text-foreground leading-relaxed">"{l.voice}"</div>
                  )}
                  <SectionLabel>What changes</SectionLabel>
                  {l.mutations.length === 0 ? (
                    <div className="dim text-[13px]">No mutations recorded.</div>
                  ) : (
                    <div className="mono rounded-[var(--radius)] bg-muted px-3 py-2.5 text-[12.5px] leading-[1.7]">
                      {l.mutations.map((m, i) => (
                        <div key={i}>{m}</div>
                      ))}
                    </div>
                  )}
                  <div className="dim mt-2.5 text-xs leading-snug">
                    Branched from <span className="mono">{l.parent_version || '—'}</span>; candidate
                    ID <span className="mono">{l.id}</span>
                  </div>
                </div>
                <div>
                  <SectionLabel>Eval movement</SectionLabel>
                  {typeof overall === 'number' && (
                    <div className="mb-2.5 flex items-center justify-between rounded-[var(--radius)] bg-muted px-3 py-2.5 text-[13px]">
                      <span>Δ overall</span>
                      <span
                        className="mono font-medium"
                        style={{ color: deltaColor(overall) }}
                      >
                        {fmtRubric(overall)}
                      </span>
                    </div>
                  )}
                  {rubricKeys.length > 0 && (
                    <div className="mb-3.5 overflow-hidden rounded-[var(--radius)] border border-border">
                      {rubricKeys.map((k, i) => {
                        const v = perRubric[k] ?? 0;
                        return (
                          <div
                            key={k}
                            className={`flex items-center justify-between px-2.5 py-2 text-[12.5px] ${
                              i < rubricKeys.length - 1 ? 'border-b border-border' : ''
                            }`}
                          >
                            <span>{k}</span>
                            <span className="mono font-medium" style={{ color: deltaColor(v) }}>
                              {fmtRubric(v)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <Button asChild variant="ghost" size="sm" className="px-2.5">
                    <Link to="/lesson/$id" params={{ id: l.id }} search={preserveSearch}>
                      <Icon name="eye" size={12} /> See full reasoning
                    </Link>
                  </Button>
                </div>
              </div>
              <div className="flex items-center gap-2 border-t border-border bg-[var(--bg-sunken)] px-4 py-3">
                <span className="dim text-xs flex-1">
                  Auto-decision in <span className="mono">24h</span> per policy
                </span>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => onAction(l, 'reject')}
                  disabled={!!busy}
                >
                  <Icon name="x" size={14} /> {busy === 'reject' ? 'Rejecting…' : 'Reject'}
                </Button>
                <Button
                  size="sm"
                  onClick={() => onAction(l, 'approve')}
                  disabled={!!busy}
                >
                  <Icon name="check" size={14} />{' '}
                  {busy === 'approve' ? 'Promoting…' : 'Approve & ship'}
                </Button>
              </div>
            </Card>
          );
        })}
      </div>

      {decided.length > 0 && (
        <div className="mt-6">
          <SectionLabel>Decided just now</SectionLabel>
          <div className="flex flex-col gap-2">
            {decided.map(({ id, title, action }) => (
              <Card
                key={id}
                className="flex flex-row items-center gap-3 px-4 py-3 text-[13.5px]"
              >
                <Tag kind={action === 'approve' ? 'success' : 'bad'}>
                  {action === 'approve' ? 'Approved' : 'Rejected'}
                </Tag>
                <span className="flex-1 truncate">{title}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!!acting[id]}
                  onClick={() => requeue(id)}
                >
                  {acting[id] ? 'Undoing…' : 'Undo'}
                </Button>
              </Card>
            ))}
          </div>
        </div>
      )}

      {(() => {
        const decidedNowIds = new Set(decided.map((d) => d.id));
        const earlier = history.filter((l) => !decidedNowIds.has(l.id));
        if (earlier.length === 0) return null;
        const shown = earlier.slice(0, HISTORY_LIMIT);
        return (
          <div className="mt-6">
            <SectionLabel
              hint={
                earlier.length > HISTORY_LIMIT
                  ? `showing ${HISTORY_LIMIT} of ${earlier.length}`
                  : `${earlier.length}`
              }
            >
              Earlier decisions
            </SectionLabel>
            <div className="flex flex-col gap-2">
              {shown.map((l) => {
                const kind = decidedKind(l.status);
                const tagKind = kind === 'rejected' ? 'bad' : 'success';
                const label =
                  kind === 'rejected' ? 'Rejected' : kind === 'auto' ? 'Auto-promoted' : 'Approved';
                const when = (kind !== 'rejected' && l.promoted_at) || fmtLessonTime(l.id);
                return (
                  <Card
                    key={l.id}
                    className="flex flex-row items-center gap-3 px-4 py-3 text-[13.5px]"
                  >
                    <Tag kind={tagKind}>{label}</Tag>
                    <Tag>
                      <KindIcon kind={l.kind} /> <KindLabel kind={l.kind} />
                    </Tag>
                    <span className="flex-1 truncate">{l.title}</span>
                    {when && <span className="dim mono text-[11.5px]">{when}</span>}
                    <Button asChild variant="ghost" size="sm">
                      <Link to="/lesson/$id" params={{ id: l.id }} search={preserveSearch}>
                        <Icon name="eye" size={12} /> Details
                      </Link>
                    </Button>
                    {kind === 'rejected' && (
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={!!acting[l.id]}
                        onClick={() => requeue(l.id)}
                      >
                        {acting[l.id] ? 'Requeuing…' : 'Requeue'}
                      </Button>
                    )}
                  </Card>
                );
              })}
            </div>
          </div>
        );
      })()}
    </div>
  );
};
