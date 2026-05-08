import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

export type TagKind = '' | 'success' | 'warn' | 'bad' | 'info' | 'solid';

export const Tag = ({ kind, children }: { kind?: TagKind; children: ReactNode }) => (
  <span className={`tag ${kind || ''}`}>{children}</span>
);

export type LessonStatus = 'pending' | 'approved' | 'rolled_back' | 'auto_promoted' | 'awaiting_review' | 'human_rejected';

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
      {m.dot && <span className="dot" />}
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
