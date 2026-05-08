import type { ReactNode } from 'react';
import { Badge } from './ui/badge';
import { Icon, type IconName } from './Icon';

export type TagKind = '' | 'success' | 'warn' | 'bad' | 'info' | 'solid';

const KIND_TO_VARIANT = {
  '': 'neutral',
  success: 'success',
  warn: 'warn',
  bad: 'bad',
  info: 'info',
  solid: 'solid',
} as const;

export const Tag = ({ kind, children }: { kind?: TagKind; children: ReactNode }) => (
  <Badge variant={KIND_TO_VARIANT[kind ?? '']} className="gap-1">
    {children}
  </Badge>
);

export type LessonStatus = 'pending' | 'approved' | 'rolled_back' | 'auto_promoted' | 'awaiting_review' | 'human_rejected';

const STATUS_DOT_COLOR: Record<TagKind, string> = {
  '': 'var(--muted-foreground)',
  success: 'var(--accent-fg)',
  warn: 'var(--warn-fg)',
  bad: 'var(--bad-fg)',
  info: 'var(--info-fg)',
  solid: 'var(--primary-foreground)',
};

const StatusDot = ({ kind }: { kind: TagKind }) => (
  <span
    aria-hidden
    style={{
      width: 6,
      height: 6,
      borderRadius: '50%',
      background: STATUS_DOT_COLOR[kind],
      display: 'inline-block',
    }}
  />
);

export const StatusTag = ({ status }: { status: LessonStatus | string }) => {
  const map: Record<string, { kind: TagKind; label: string; dot: boolean }> = {
    pending: { kind: 'warn', label: 'Awaiting review', dot: true },
    awaiting_review: { kind: 'warn', label: 'Awaiting review', dot: true },
    approved: { kind: 'success', label: 'Approved', dot: true },
    rolled_back: { kind: 'bad', label: 'Rolled back', dot: true },
    auto_promoted: { kind: 'success', label: 'Auto-promoted', dot: true },
    human_rejected: { kind: 'bad', label: 'Rejected', dot: true },
  };
  const m = map[status] || { kind: '' as TagKind, label: status, dot: false };
  return (
    <Tag kind={m.kind}>
      {m.dot && <StatusDot kind={m.kind} />}
      {m.label}
    </Tag>
  );
};

// LessonKind covers both legacy mock kinds (prompt/policy/eval/rollback) and
// the kinds the harness actually emits today (rag/rerank/router/memory/other).
export type LessonKind =
  | 'prompt'
  | 'policy'
  | 'router'
  | 'eval'
  | 'rollback'
  | 'rag'
  | 'rerank'
  | 'memory'
  | 'other';

const KIND_ICON: Record<string, IconName> = {
  prompt: 'sparkles',
  policy: 'shield',
  router: 'route',
  eval: 'flask',
  rollback: 'rollback',
  rag: 'book',
  rerank: 'sliders',
  memory: 'inbox',
  other: 'settings',
};

const KIND_LABEL: Record<string, string> = {
  prompt: 'Prompt change',
  policy: 'Behavior change',
  router: 'Routing change',
  eval: 'New self-test',
  rollback: 'Rolled back',
  rag: 'Retrieval change',
  rerank: 'Reranking change',
  memory: 'Memory change',
  other: 'Adjustment',
};

export const KindIcon = ({ kind }: { kind: LessonKind | string }) => (
  <Icon name={KIND_ICON[kind] || 'sparkles'} size={14} />
);

export const KindLabel = ({ kind }: { kind: LessonKind | string }) => (
  <span>{KIND_LABEL[kind] || kind}</span>
);
